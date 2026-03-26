"""Thin wrappers around mlx_lm / mlx_vlm for model loading and generation."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from bench.config import ModelVariant
from bench.prompts import Prompt

logger = logging.getLogger(__name__)


@dataclass
class GenerationChunk:
    """One step of generation output."""

    text: str
    # These are populated on every chunk for text models (from mlx_lm).
    # For vision models, only the final chunk has meaningful values.
    prompt_tokens: int = 0
    prompt_tps: float = 0.0  # prefill tokens/sec
    generation_tokens: int = 0
    generation_tps: float = 0.0  # decode tokens/sec
    peak_memory_gb: float = 0.0


def load_model(variant: ModelVariant, kind: str) -> tuple[Any, Any]:
    """Load a model and its tokenizer/processor.

    Returns (model, tokenizer) for text models or (model, processor) for vision models.
    """
    if kind == "vision":
        from mlx_vlm import load

        model, processor = load(variant.repo)
        return model, processor
    else:
        from mlx_lm import load

        model, tokenizer = load(variant.repo)
        return model, tokenizer


def generate_stream(
    variant: ModelVariant,
    kind: str,
    model: Any,
    tokenizer_or_processor: Any,
    prompt: Prompt,
    max_tokens: int,
    temperature: float = 0.0,
) -> Iterator[GenerationChunk]:
    """Unified streaming interface for text and vision models.

    Yields GenerationChunk per generation step with prefill/decode stats.
    """
    if kind == "vision" and prompt.is_vision:
        yield from _generate_vision(
            model, tokenizer_or_processor, prompt, max_tokens, temperature
        )
    elif kind == "vision" and not prompt.is_vision:
        yield from _generate_vision_text_only(
            model, tokenizer_or_processor, prompt, max_tokens, temperature
        )
    else:
        yield from _generate_text(
            model, tokenizer_or_processor, prompt, max_tokens, temperature
        )


def _is_qwen35_model(tokenizer: Any) -> bool:
    """Detect Qwen3.5 models by checking tokenizer config for model_type or special tokens."""
    # Check name_or_path on the tokenizer
    name = getattr(tokenizer, "name_or_path", "") or ""
    if "qwen3.5" in name.lower() or "qwen3_5" in name.lower():
        return True
    # Check for Qwen3.5 special tokens (DeltaNet hybrid architecture)
    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    return "<tool_call>" in vocab and "<think>" in vocab and len(vocab) > 240000


def _generate_text(
    model: Any,
    tokenizer: Any,
    prompt: Prompt,
    max_tokens: int,
    temperature: float,
) -> Iterator[GenerationChunk]:
    """Stream text generation using mlx_lm."""
    from mlx_lm import stream_generate
    from mlx_lm.generate import make_sampler

    messages = [{"role": "user", "content": prompt.text}]

    if hasattr(tokenizer, "apply_chat_template"):
        # Qwen3.5 has thinking mode enabled by default which inflates output
        # tokens and pollutes throughput measurements.  Keep it on for
        # reasoning prompts so we actually benchmark the thinking capability.
        template_kwargs: dict[str, Any] = {}
        if _is_qwen35_model(tokenizer):
            is_reasoning = prompt.id in ("reasoning",)
            template_kwargs["enable_thinking"] = is_reasoning
            logger.debug("Qwen3.5 detected — thinking mode %s for [%s]",
                         "enabled" if is_reasoning else "disabled", prompt.id)
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            **template_kwargs,
        )
    else:
        formatted = prompt.text

    sampler = make_sampler(temp=temperature)
    for response in stream_generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        yield GenerationChunk(
            text=response.text,
            prompt_tokens=response.prompt_tokens,
            prompt_tps=response.prompt_tps,
            generation_tokens=response.generation_tokens,
            generation_tps=response.generation_tps,
            peak_memory_gb=response.peak_memory,
        )


def _generate_vision(
    model: Any,
    processor: Any,
    prompt: Prompt,
    max_tokens: int,
    temperature: float,
) -> Iterator[GenerationChunk]:
    """Stream vision model generation using mlx_vlm."""
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.utils import load_image

    image = load_image(prompt.image)

    output = vlm_generate(
        model,
        processor,
        image=image,
        prompt=prompt.text,
        max_tokens=max_tokens,
        temp=temperature,
        verbose=False,
    )
    # mlx_vlm.generate returns the full string; yield as one chunk
    yield GenerationChunk(text=output)


def _generate_vision_text_only(
    model: Any,
    processor: Any,
    prompt: Prompt,
    max_tokens: int,
    temperature: float,
) -> Iterator[GenerationChunk]:
    """Vision model generating text-only prompt (no image)."""
    from mlx_vlm import generate as vlm_generate

    output = vlm_generate(
        model,
        processor,
        prompt=prompt.text,
        max_tokens=max_tokens,
        temp=temperature,
        verbose=False,
    )
    yield GenerationChunk(text=output)
