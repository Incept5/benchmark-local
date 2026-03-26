"""Quality evaluation: perplexity, MMLU-Pro accuracy, output similarity."""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Default location of the MMLU-Pro test parquet (downloaded from TIGER-Lab/MMLU-Pro)
MMLU_PRO_PATH = Path(__file__).resolve().parent.parent.parent / "evals" / "mmlu_pro" / "data" / "test-00000-of-00001.parquet"


@dataclass
class QualityResults:
    perplexity: float | None = None
    mmlu_accuracy: float | None = None
    mmlu_correct: int = 0
    mmlu_total: int = 0
    mmlu_per_category: dict[str, float] = field(default_factory=dict)
    output_similarity: dict[str, float] | None = None  # prompt_id -> similarity


def compute_perplexity(
    model: Any,
    tokenizer: Any,
    text: str,
    max_tokens: int = 8192,
    window_size: int = 2048,
    stride: int = 1024,
) -> float:
    """Compute perplexity using a sliding-window approach.

    Processes up to *max_tokens* of the input text.  A window of
    *window_size* tokens slides forward by *stride* tokens each step.
    Only the new (non-overlapping) tokens in each window contribute to
    the loss, while earlier tokens in the window provide context.
    This gives every scored token the benefit of prior context, matching
    the standard HuggingFace ``evaluate`` methodology.

    Args:
        max_tokens: Cap on how many tokens of *text* to evaluate.
        window_size: Context window fed to the model each step.
        stride: How far the window advances each step.  Tokens in
            ``[prev_end .. prev_end+stride)`` are the ones scored.
    """
    # Tokenize
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text)
    elif hasattr(tokenizer, "tokenizer"):
        tokens = tokenizer.tokenizer.encode(text)
    else:
        raise ValueError("Cannot find encode method on tokenizer")

    if len(tokens) < 2:
        return float("inf")

    # Cap length
    tokens = tokens[:max_tokens]
    seq_len = len(tokens)
    tokens_array = mx.array([tokens])

    total_loss = 0.0
    num_tokens = 0

    # Slide a window across the sequence
    prev_end = 0
    for begin in range(0, seq_len - 1, stride):
        end = min(begin + window_size, seq_len)
        chunk = tokens_array[:, begin:end]

        logits = model(chunk)
        mx.eval(logits)

        # Only score the *new* tokens (those past prev_end)
        # Within the chunk, the target at position i predicts token i+1.
        # We want to score targets whose actual sequence position >= prev_end.
        score_start = max(prev_end - begin, 0)

        shift_logits = logits[:, score_start:-1, :]
        shift_labels = chunk[:, score_start + 1:]

        if shift_labels.size == 0:
            prev_end = end
            continue

        loss = nn.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        mx.eval(loss)
        total_loss += loss.item()
        num_tokens += shift_labels.size

        prev_end = end
        if end >= seq_len:
            break

    avg_loss = total_loss / num_tokens if num_tokens > 0 else float("inf")
    return math.exp(avg_loss)


def _get_answer_token_ids(tokenizer: Any, num_options: int = 10) -> dict[str, int]:
    """Get token IDs for answer letters (A through J for MMLU-Pro)."""
    answer_ids = {}
    for i in range(num_options):
        letter = chr(65 + i)  # A, B, C, ... J
        ids = tokenizer.encode(letter)
        # Use the last token (in case BOS/prefix tokens are added)
        answer_ids[letter] = ids[-1]
    return answer_ids


def _ensure_mmlu_pro(parquet_path: Path) -> Path:
    """Download MMLU-Pro from HuggingFace if not already present."""
    if parquet_path.exists():
        return parquet_path
    logger.info("MMLU-Pro dataset not found — downloading from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="TIGER-Lab/MMLU-Pro",
            filename="data/test-00000-of-00001.parquet",
            repo_type="dataset",
            local_dir=parquet_path.parent.parent,  # evals/mmlu_pro/
        )
        logger.info("MMLU-Pro downloaded to %s", parquet_path)
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is required to download MMLU-Pro. "
            "Install it with: pip install huggingface_hub"
        )
    return parquet_path


def _load_mmlu_pro(
    parquet_path: str | Path,
    fraction: float = 1.0,
    seed: int = 42,
) -> Any:
    """Load MMLU-Pro dataset, optionally sampling a fraction.

    Returns a pandas DataFrame with columns:
        question, options (list[str]), answer (letter), category
    Sampling is stratified by category to maintain balance.
    """
    import pandas as pd

    path = _ensure_mmlu_pro(Path(parquet_path))
    df = pd.read_parquet(path)

    if fraction < 1.0:
        # Stratified sample: maintain category proportions
        sampled_parts = []
        for _cat, group in df.groupby("category"):
            sampled_parts.append(group.sample(frac=fraction, random_state=seed))
        df = pd.concat(sampled_parts).reset_index(drop=True)

    return df


def eval_mmlu(
    model: Any, tokenizer: Any,
    parquet_path: str | Path | None = None,
    fraction: float = 0.05,
    temperature: float = 0.0,
) -> tuple[float, int, int, dict[str, float]]:
    """Evaluate on MMLU-Pro using logprob scoring.

    Computes next-token logits and picks the answer choice (A-J) with
    the highest probability.  Robust to thinking models, verbose models,
    and broken chat templates.

    Args:
        parquet_path: Path to MMLU-Pro test parquet. Defaults to bundled copy.
        fraction: Fraction of questions to use (0.05=tiny, 0.10=small, 0.25=medium, 1.0=full).

    Returns (accuracy, correct, total, per_category_accuracy).
    """
    if parquet_path is None:
        parquet_path = MMLU_PRO_PATH

    df = _load_mmlu_pro(parquet_path, fraction=fraction)
    if df.empty:
        return 0.0, 0, 0, {}

    answer_ids = _get_answer_token_ids(tokenizer, num_options=10)
    from bench.models import _is_qwen35_model

    correct = 0
    total = len(df)
    cat_correct: dict[str, int] = {}
    cat_total: dict[str, int] = {}

    for _, row in df.iterrows():
        question_text = row["question"]
        options = row["options"]
        expected = row["answer"]
        category = row["category"]

        cat_total[category] = cat_total.get(category, 0) + 1

        # Format as multiple-choice prompt
        prompt_text = f"{question_text}\n"
        for i, choice in enumerate(options):
            letter = chr(65 + i)
            prompt_text += f"{letter}. {choice}\n"
        num_opts = len(options)
        last_letter = chr(64 + num_opts)  # e.g. "J" for 10 options
        prompt_text += f"Answer with just the letter (A through {last_letter}):"

        # Apply chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            template_kwargs: dict[str, Any] = {}
            if _is_qwen35_model(tokenizer):
                template_kwargs["enable_thinking"] = False
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )
            # Strip hardcoded <think> from broken fine-tune templates
            if formatted.endswith("<think>\n"):
                formatted = formatted[:-len("<think>\n")]
        else:
            formatted = prompt_text

        # Tokenize and get logits for the next token
        if hasattr(tokenizer, "encode"):
            tokens = tokenizer.encode(formatted)
        else:
            tokens = tokenizer.tokenizer.encode(formatted)

        tokens_array = mx.array([tokens])
        logits = model(tokens_array)
        mx.eval(logits)

        # Pick the answer with highest logit among valid options
        next_logits = logits[0, -1, :]
        best_letter = ""
        best_logit = float("-inf")
        for i in range(num_opts):
            letter = chr(65 + i)
            tid = answer_ids.get(letter)
            if tid is None:
                continue
            logit_val = next_logits[tid].item()
            if logit_val > best_logit:
                best_logit = logit_val
                best_letter = letter

        if best_letter == expected:
            correct += 1
            cat_correct[category] = cat_correct.get(category, 0) + 1

    accuracy = correct / total if total > 0 else 0.0
    per_category = {
        cat: cat_correct.get(cat, 0) / cat_total[cat]
        for cat in sorted(cat_total)
    }
    return accuracy, correct, total, per_category


@dataclass
class MMLUGenResult:
    """Result of a single MMLU-Pro generation-based evaluation."""
    question_id: int
    category: str
    expected: str
    predicted: str
    correct: bool
    ttft_ms: float
    tokens_generated: int
    generation_time_s: float
    decode_tps: float
    peak_memory_mb: float
    thinking: bool
    response_text: str


def _extract_answer_from_response(response: str) -> str:
    """Extract A-J answer letter from a generated response.

    Handles thinking blocks, verbose explanations, and various formats.
    """
    import re

    text = response.strip()

    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # "The answer is X"
    m = re.search(r"(?:answer\s*(?:is|:)\s*)([A-Ja-j])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Standalone letter at start: "B", "B.", "(B)", "**B**"
    m = re.match(r"^[\s*(*]*([A-Ja-j])[\s*).,:]*(?:$|\s)", text)
    if m:
        return m.group(1).upper()

    # "X." pattern
    m = re.search(r"\b([A-Ja-j])\.\s", text)
    if m:
        return m.group(1).upper()

    # Last standalone letter (thinking models often put the answer at the end)
    matches = re.findall(r"\b([A-Ja-j])\b", text)
    if matches:
        return matches[-1].upper()

    return ""


def eval_mmlu_generate(
    model: Any, tokenizer: Any,
    parquet_path: str | Path | None = None,
    fraction: float = 0.10,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    max_tokens: int = 2048,
    on_progress: Any = None,
) -> tuple[float, int, int, dict[str, float], list[MMLUGenResult]]:
    """Evaluate MMLU-Pro using text generation (supports thinking mode).

    Unlike logprob scoring, this generates a full response and extracts
    the answer letter.  Captures timing metrics (TTFT, tok/s, memory)
    for each question.

    Args:
        enable_thinking: If True, allow the model to use chain-of-thought
            reasoning (Qwen3.5 thinking mode).  If False, disable it.
        max_tokens: Maximum tokens to generate per question.
        on_progress: Optional callback(current, total) for progress updates.

    Returns (accuracy, correct, total, per_category, results_list).
    """
    import time

    if parquet_path is None:
        parquet_path = MMLU_PRO_PATH

    df = _load_mmlu_pro(parquet_path, fraction=fraction)
    if df.empty:
        return 0.0, 0, 0, {}, []

    from mlx_lm import stream_generate
    from mlx_lm.generate import make_sampler
    from bench.models import _is_qwen35_model

    sampler = make_sampler(temp=temperature)
    is_qwen35 = _is_qwen35_model(tokenizer)

    correct = 0
    total = len(df)
    cat_correct: dict[str, int] = {}
    cat_total: dict[str, int] = {}
    results: list[MMLUGenResult] = []

    for idx, (_, row) in enumerate(df.iterrows()):
        question_text = row["question"]
        options = row["options"]
        expected = row["answer"]
        category = row["category"]
        q_id = row.get("question_id", idx)

        cat_total[category] = cat_total.get(category, 0) + 1

        # Format prompt
        num_opts = len(options)
        last_letter = chr(64 + num_opts)
        prompt_text = f"{question_text}\n"
        for i, choice in enumerate(options):
            prompt_text += f"{chr(65 + i)}. {choice}\n"
        prompt_text += f"Answer with just the letter (A through {last_letter}):"

        # Apply chat template
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            template_kwargs: dict[str, Any] = {}
            if is_qwen35:
                template_kwargs["enable_thinking"] = enable_thinking
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )
            # Don't strip <think> here — we want it if thinking is enabled
            if not enable_thinking and formatted.endswith("<think>\n"):
                formatted = formatted[:-len("<think>\n")]
        else:
            formatted = prompt_text

        # Generate with timing
        mx.reset_peak_memory()
        chunks = []
        ttft_ms = 0.0
        start = time.perf_counter()

        for i, response in enumerate(stream_generate(
            model, tokenizer, prompt=formatted,
            max_tokens=max_tokens, sampler=sampler,
        )):
            if i == 0:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            chunks.append(response)

        end = time.perf_counter()
        gen_time = end - start

        output_text = "".join(c.text for c in chunks)
        num_tokens = len(chunks)
        last_chunk = chunks[-1] if chunks else None
        decode_tps = last_chunk.generation_tps if last_chunk else 0.0
        peak_mem_mb = mx.get_peak_memory() / 1024**2

        # Extract answer
        predicted = _extract_answer_from_response(output_text)
        is_correct = predicted == expected

        if is_correct:
            correct += 1
            cat_correct[category] = cat_correct.get(category, 0) + 1

        results.append(MMLUGenResult(
            question_id=q_id,
            category=category,
            expected=expected,
            predicted=predicted,
            correct=is_correct,
            ttft_ms=ttft_ms,
            tokens_generated=num_tokens,
            generation_time_s=gen_time,
            decode_tps=decode_tps,
            peak_memory_mb=peak_mem_mb,
            thinking=enable_thinking,
            response_text=output_text,
        ))

        if on_progress:
            on_progress(idx + 1, total)

    accuracy = correct / total if total > 0 else 0.0
    per_category = {
        cat: cat_correct.get(cat, 0) / cat_total[cat]
        for cat in sorted(cat_total)
    }
    return accuracy, correct, total, per_category, results


def compute_output_similarity(output_text: str, reference_text: str) -> float:
    """Compute token-level F1 similarity between two texts.

    Uses unigram token overlap (bag-of-words F1) — no external embedding model needed.
    """
    if not output_text or not reference_text:
        return 0.0

    output_tokens = output_text.lower().split()
    reference_tokens = reference_text.lower().split()

    if not output_tokens or not reference_tokens:
        return 0.0

    output_counts = Counter(output_tokens)
    reference_counts = Counter(reference_tokens)

    # Compute overlap
    overlap = 0
    for token, count in output_counts.items():
        overlap += min(count, reference_counts.get(token, 0))

    precision = overlap / len(output_tokens) if output_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1
