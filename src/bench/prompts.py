"""Prompt suite loader."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


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
