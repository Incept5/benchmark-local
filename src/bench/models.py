"""Thin wrappers around mlx_lm / mlx_vlm for model loading and generation."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from bench.config import ModelVariant
from bench.prompts import Prompt

logger = logging.getLogger(__name__)


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
) -> Iterator[str]:
    """Unified streaming interface for text and vision models.

    Yields one string chunk per generation step (typically one token).
    """
    if kind == "vision" and prompt.is_vision:
        yield from _generate_vision(
            model, tokenizer_or_processor, prompt, max_tokens, temperature
        )
    elif kind == "vision" and not prompt.is_vision:
        # Vision model on text-only prompt: use text generation path
        yield from _generate_vision_text_only(
            model, tokenizer_or_processor, prompt, max_tokens, temperature
        )
    else:
        yield from _generate_text(
            model, tokenizer_or_processor, prompt, max_tokens, temperature
        )


def _generate_text(
    model: Any,
    tokenizer: Any,
    prompt: Prompt,
    max_tokens: int,
    temperature: float,
) -> Iterator[str]:
    """Stream text generation using mlx_lm."""
    from mlx_lm import stream_generate
    from mlx_lm.generate import make_sampler

    messages = [{"role": "user", "content": prompt.text}]

    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
        yield response.text


def _generate_vision(
    model: Any,
    processor: Any,
    prompt: Prompt,
    max_tokens: int,
    temperature: float,
) -> Iterator[str]:
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
    # mlx_vlm.generate returns the full string; yield it as one chunk
    # For token counting, we'll post-process with tokenizer
    yield output


def _generate_vision_text_only(
    model: Any,
    processor: Any,
    prompt: Prompt,
    max_tokens: int,
    temperature: float,
) -> Iterator[str]:
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
    yield output
