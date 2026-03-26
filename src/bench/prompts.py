"""Prompt suite loader."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Filler passage used to pad prompts to the target context size.
# This is a factual passage about computer science history that any LLM
# tokenizer will handle predictably (~1 token per word).
_FILLER_PASSAGE = (
    "The history of computing spans centuries of innovation. Charles Babbage designed the "
    "Analytical Engine in 1837, a mechanical general-purpose computer that was never completed "
    "but anticipated modern computer architecture with its mill (processor), store (memory), "
    "and input/output mechanisms. Ada Lovelace wrote what is considered the first computer "
    "program for this machine, an algorithm to compute Bernoulli numbers. "
    "In the 1930s, Alan Turing formalized the concept of computation with his theoretical "
    "Turing machine, proving that certain problems are undecidable. Meanwhile, Alonzo Church "
    "developed lambda calculus, establishing an equivalent model of computation. "
    "The first electronic general-purpose computer, ENIAC, was completed in 1945 at the "
    "University of Pennsylvania. It weighed 30 tons, occupied 1,800 square feet, and consumed "
    "150 kilowatts of power. It could perform 5,000 additions per second. "
    "John von Neumann proposed the stored-program architecture in 1945, where instructions "
    "and data share the same memory. This design remains the foundation of virtually all "
    "modern computers. The first implementation was the Manchester Baby in 1948. "
    "The invention of the transistor at Bell Labs in 1947 by Bardeen, Brattain, and Shockley "
    "revolutionized electronics. The integrated circuit followed in 1958, independently "
    "invented by Jack Kilby at Texas Instruments and Robert Noyce at Fairchild Semiconductor. "
    "Gordon Moore observed in 1965 that the number of transistors on a chip doubled "
    "approximately every two years, a trend that held for decades and drove exponential "
    "improvements in computing power. "
    "The development of high-level programming languages began with Fortran in 1957, designed "
    "by John Backus at IBM for scientific computing. COBOL followed in 1959 for business "
    "applications. Lisp, created by John McCarthy in 1958, introduced garbage collection, "
    "recursion, and the concept of programs as data. C, developed by Dennis Ritchie at Bell "
    "Labs in 1972, became the lingua franca of systems programming and remains widely used. "
    "The ARPANET, precursor to the internet, transmitted its first message in 1969 between "
    "UCLA and Stanford Research Institute. TCP/IP was standardized in 1983, and Tim Berners-Lee "
    "invented the World Wide Web at CERN in 1989. The Mosaic browser in 1993 brought the web "
    "to the general public, triggering the dot-com boom. "
    "Parallel computing evolved from vector processors in the 1970s through symmetric "
    "multiprocessing in the 1990s to the massively parallel GPU computing that powers modern "
    "AI. NVIDIA's CUDA platform, released in 2006, made GPU programming accessible to "
    "researchers and catalyzed the deep learning revolution. "
    "Machine learning has roots in the 1950s with Frank Rosenblatt's perceptron. After the "
    "AI winters of the 1970s and 1990s, the field was revitalized by deep learning. "
    "AlexNet won the ImageNet competition in 2012 using a deep convolutional neural network "
    "trained on GPUs, achieving a dramatic improvement over traditional methods. The "
    "transformer architecture, introduced in 2017, enabled the development of large language "
    "models that can generate coherent text, translate languages, write code, and reason "
    "about complex problems. These models are trained on vast amounts of text data using "
    "self-supervised learning objectives like next-token prediction. "
)


@dataclass
class Prompt:
    id: str
    category: str
    text: str
    image: str | None = None
    max_tokens: int | None = None  # per-prompt override; None = use config default

    @property
    def is_vision(self) -> bool:
        return self.image is not None


def load_suite(path: str | Path) -> list[Prompt]:
    """Load prompt suite from a TOML file."""
    path = Path(path)
    with open(path, "rb") as f:
        data = tomllib.load(f)

    prompts = []
    for p in data.get("prompt", []):
        prompts.append(
            Prompt(
                id=p["id"],
                category=p["category"],
                text=p["text"],
                image=p.get("image"),
                max_tokens=p.get("max_tokens"),
            )
        )
    return prompts


def pad_prompt_to_context(prompt: Prompt, target_tokens: int, tokenizer: Any) -> Prompt:
    """Return a copy of the prompt with text padded to approximately *target_tokens*.

    The padding is added as a clearly-delimited "background context" block before
    the original prompt, so the model still sees the real instruction last.
    Vision prompts are returned unchanged (context is dominated by the image).

    The tokenizer is used to measure the current token count so we know how much
    filler to add. We overshoot slightly and then trim, since token boundaries
    don't align perfectly with word boundaries.
    """
    if prompt.is_vision:
        return prompt

    # Measure current prompt size in tokens
    current_tokens = _count_tokens(prompt.text, tokenizer)
    if current_tokens >= target_tokens:
        return prompt

    needed = target_tokens - current_tokens
    filler = _generate_filler(needed, tokenizer)

    padded_text = (
        "The following background context is provided for reference. "
        "After reading it, answer the question at the end.\n\n"
        "---BEGIN CONTEXT---\n"
        f"{filler}\n"
        "---END CONTEXT---\n\n"
        f"{prompt.text}"
    )

    actual = _count_tokens(padded_text, tokenizer)
    logger.debug(
        "Padded prompt %s from %d to %d tokens (target %d)",
        prompt.id, current_tokens, actual, target_tokens,
    )

    return replace(prompt, text=padded_text)


def _count_tokens(text: str, tokenizer: Any) -> int:
    """Count tokens using whatever tokenizer/processor we have."""
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text))
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "encode"):
        return len(tokenizer.tokenizer.encode(text))
    # Rough fallback: ~1.3 tokens per word
    return int(len(text.split()) * 1.3)


def _generate_filler(target_tokens: int, tokenizer: Any) -> str:
    """Repeat the filler passage until we reach approximately target_tokens."""
    # Measure one copy of the filler
    filler_tokens = _count_tokens(_FILLER_PASSAGE, tokenizer)
    if filler_tokens <= 0:
        return ""

    # How many full copies do we need, plus a partial
    full_copies = target_tokens // filler_tokens
    remainder_tokens = target_tokens - (full_copies * filler_tokens)

    parts = [_FILLER_PASSAGE] * full_copies

    # Add a partial copy for the remainder
    if remainder_tokens > 0:
        words = _FILLER_PASSAGE.split()
        # Approximate: take (remainder_tokens / filler_tokens) fraction of words
        frac = remainder_tokens / filler_tokens
        num_words = max(1, int(len(words) * frac))
        parts.append(" ".join(words[:num_words]))

    return "\n\n".join(parts)
