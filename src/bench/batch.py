"""Batch throughput measurement using mlx_lm.batch_generate."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of one batch throughput measurement."""
    batch_size: int
    total_prompt_tokens: int
    total_generated_tokens: int
    prefill_time_s: float
    generation_time_s: float
    prefill_tps: float  # prompt tokens / prefill time
    generation_tps: float  # generated tokens / generation time
    peak_memory_bytes: int


def measure_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> BatchResult:
    """Run batch_generate with *batch_size* concurrent prompts and measure throughput.

    Args:
        model: MLX model
        tokenizer: tokenizer with apply_chat_template
        prompts: raw prompt texts (will be formatted and tokenized)
        batch_size: number of concurrent prompts
        max_tokens: max tokens to generate per prompt
        temperature: sampling temperature

    Returns:
        BatchResult with aggregate throughput metrics.
    """
    from mlx_lm import batch_generate
    from mlx_lm.generate import make_sampler

    # Select and cycle prompts to fill the batch
    selected: list[str] = []
    for i in range(batch_size):
        selected.append(prompts[i % len(prompts)])

    # Format with chat template and tokenize
    tokenized_prompts: list[list[int]] = []
    for text in selected:
        messages = [{"role": "user", "content": text}]
        if hasattr(tokenizer, "apply_chat_template"):
            from bench.models import _is_qwen35_model
            template_kwargs: dict[str, Any] = {}
            if _is_qwen35_model(tokenizer):
                template_kwargs["enable_thinking"] = False
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )
        else:
            formatted = text
        tokens = tokenizer.encode(formatted)
        tokenized_prompts.append(tokens)

    total_prompt_tokens = sum(len(t) for t in tokenized_prompts)

    # Reset memory tracking
    mx.reset_peak_memory()

    sampler = make_sampler(temp=temperature)

    start = time.perf_counter()
    result = batch_generate(
        model,
        tokenizer,
        prompts=tokenized_prompts,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )
    end = time.perf_counter()

    total_time = end - start
    peak_memory = mx.get_peak_memory()

    # Extract stats from BatchResponse
    stats = result.stats
    prefill_time = getattr(stats, "prompt_time", 0.0) or 0.0
    gen_time = getattr(stats, "generation_time", 0.0) or 0.0
    gen_tokens = getattr(stats, "generation_tokens", 0) or 0

    # Fallback: estimate from texts if stats are incomplete
    if gen_tokens == 0:
        gen_tokens = sum(
            len(tokenizer.encode(t)) for t in result.texts
        )
    if prefill_time == 0.0:
        # Rough split: assume prefill is ~10% of total for small batches
        prefill_time = total_time * 0.1
        gen_time = total_time * 0.9

    prefill_tps = total_prompt_tokens / prefill_time if prefill_time > 0 else 0.0
    generation_tps = gen_tokens / gen_time if gen_time > 0 else 0.0

    return BatchResult(
        batch_size=batch_size,
        total_prompt_tokens=total_prompt_tokens,
        total_generated_tokens=gen_tokens,
        prefill_time_s=prefill_time,
        generation_time_s=gen_time,
        prefill_tps=prefill_tps,
        generation_tps=generation_tps,
        peak_memory_bytes=peak_memory,
    )
