"""Quality evaluation: perplexity, MMLU accuracy, output similarity."""

from __future__ import annotations

import logging
import math
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class QualityResults:
    perplexity: float | None = None
    mmlu_accuracy: float | None = None
    mmlu_correct: int = 0
    mmlu_total: int = 0
    output_similarity: dict[str, float] | None = None  # prompt_id -> similarity


def compute_perplexity(model: Any, tokenizer: Any, text: str) -> float:
    """Compute perplexity on a text sample using rolling log-likelihood.

    Uses the model's forward pass to compute cross-entropy loss over the sample.
    """
    # Tokenize the text
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text)
    elif hasattr(tokenizer, "tokenizer"):
        tokens = tokenizer.tokenizer.encode(text)
    else:
        raise ValueError("Cannot find encode method on tokenizer")

    if len(tokens) < 2:
        return float("inf")

    tokens_array = mx.array([tokens])
    total_loss = 0.0
    num_tokens = 0

    # Process in chunks to manage memory
    chunk_size = 512
    for i in range(0, len(tokens) - 1, chunk_size):
        end = min(i + chunk_size + 1, len(tokens))
        chunk = tokens_array[:, i:end]

        logits = model(chunk)
        # Shift: predict next token from current
        shift_logits = logits[:, :-1, :]
        shift_labels = chunk[:, 1:]

        # Cross-entropy loss
        loss = nn.losses.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        mx.eval(loss)
        total_loss += loss.item()
        num_tokens += shift_labels.size

    avg_loss = total_loss / num_tokens if num_tokens > 0 else float("inf")
    return math.exp(avg_loss)


def eval_mmlu(model: Any, tokenizer: Any, questions_path: str | Path) -> tuple[float, int, int]:
    """Evaluate on MMLU subset. Returns (accuracy, correct, total)."""
    path = Path(questions_path)
    with open(path, "rb") as f:
        data = tomllib.load(f)

    questions = data.get("question", [])
    if not questions:
        return 0.0, 0, 0

    correct = 0
    total = len(questions)

    for q in questions:
        question_text = q["question"]
        choices = q["choices"]
        expected = q["answer"]

        # Format as multiple-choice prompt
        prompt_text = f"{question_text}\n"
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D
            prompt_text += f"{letter}. {choice}\n"
        prompt_text += "Answer with just the letter (A, B, C, or D):"

        # Apply chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt_text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt_text

        # Generate a short response
        from mlx_lm import generate
        from mlx_lm.generate import make_sampler

        response = generate(
            model, tokenizer, prompt=formatted, max_tokens=5,
            sampler=make_sampler(temp=0.0), verbose=False,
        )

        # Extract the answer letter from response
        answer = _extract_answer(response)
        if answer == expected:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def _extract_answer(response: str) -> str:
    """Extract A/B/C/D answer from model response."""
    response = response.strip().upper()
    for char in response:
        if char in "ABCD":
            return char
    return ""


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
