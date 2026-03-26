"""Auto-discover MLX models from LM Studio and generate benchmark configs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Default LM Studio models directory
LMSTUDIO_MODELS = Path(os.path.expanduser("~/.lmstudio/models"))

# Known quantization suffixes (order matters — match longer first)
_QUANT_PATTERNS = [
    ("bf16", re.compile(r"[-_]bf16$", re.IGNORECASE)),
    ("mxfp8", re.compile(r"[-_]mxfp8$", re.IGNORECASE)),
    ("8bit", re.compile(r"[-_]8[- _]?bit$", re.IGNORECASE)),
    ("4bit", re.compile(r"[-_]4[- _]?bit$", re.IGNORECASE)),
    ("4bit-dwq", re.compile(r"[-_]4bit[-_]?dwq", re.IGNORECASE)),
]

# Known base model patterns → (base_family, size_str)
# Order: most specific first
_BASE_PATTERNS = [
    (re.compile(r"Qwen3\.5[-_]([\d.]+B(?:-A[\d.]+B)?)", re.IGNORECASE), "Qwen3.5"),
    (re.compile(r"Qwen3[-_]([\d.]+B(?:-A[\d.]+B)?)", re.IGNORECASE), "Qwen3"),
    (re.compile(r"Llama[-_]?3\.?2?[-_]([\d.]+B)", re.IGNORECASE), "Llama3"),
    (re.compile(r"Mistral[-_]([\d.]+B)", re.IGNORECASE), "Mistral"),
    (re.compile(r"DeepSeek[-_]R1.*[-_]([\d.]+B)", re.IGNORECASE), "DeepSeek-R1"),
]

# Derivative classification rules — checked in order, first full match wins.
# Each rule: (keywords_all_required, label)
# If multiple rules match, earlier rule wins.
_DERIVATIVE_RULES: list[tuple[list[str], str]] = [
    (["claude", "reasoning"], "Claude Reasoning Distilled"),
    (["claude", "highiq"], "Claude HighIQ"),
    (["claude", "distill"], "Claude Distilled"),
    (["abliterat"], "Abliterated"),
    (["uncensor"], "Uncensored"),
    (["heretic"], "Uncensored"),
    (["coder"], "Coder"),
    (["thinking"], "Thinking"),
    (["guard"], "Guard"),
]


@dataclass
class DiscoveredModel:
    """A single MLX model discovered on disk."""
    path: str
    provider: str       # e.g. "mlx-community", "Jackrong"
    dir_name: str       # e.g. "Qwen3.5-9B-MLX-8bit"
    base_family: str    # e.g. "Qwen3.5"
    size: str           # e.g. "9B"
    quant: str          # e.g. "8bit", "bf16", "mxfp8"
    is_derivative: bool
    derivative_tag: str  # e.g. "Claude Distilled", "Abliterated", "" for standard
    display_name: str    # human-friendly name for grouping


@dataclass
class ModelGroup:
    """A group of models that should be compared together."""
    name: str           # e.g. "Qwen3.5 9B"
    base_family: str
    size: str
    is_standard: bool
    derivative_tag: str
    models: list[DiscoveredModel] = field(default_factory=list)


def _detect_quant(name: str) -> str:
    """Extract quantization level from model directory name."""
    for quant, pattern in _QUANT_PATTERNS:
        if pattern.search(name):
            return quant
    return "unknown"


def _detect_base(name: str) -> tuple[str, str]:
    """Extract base model family and size from directory name."""
    for pattern, family in _BASE_PATTERNS:
        m = pattern.search(name)
        if m:
            return family, m.group(1)
    return "unknown", "unknown"


def _detect_derivative(name: str) -> tuple[bool, str]:
    """Detect if model is a derivative and classify it."""
    name_lower = name.lower()
    for keywords, label in _DERIVATIVE_RULES:
        if all(kw in name_lower for kw in keywords):
            return True, label
    return False, ""


def discover_models(
    models_dir: str | Path = LMSTUDIO_MODELS,
    base_family: str | None = None,
    size: str | None = None,
) -> list[DiscoveredModel]:
    """Scan LM Studio models directory for MLX models.

    Args:
        models_dir: Root models directory (default: ~/.lmstudio/models)
        base_family: Filter to this family (e.g. "Qwen3.5")
        size: Filter to this size (e.g. "9B")
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    results = []
    for provider_dir in sorted(models_dir.iterdir()):
        if not provider_dir.is_dir():
            continue
        for model_dir in sorted(provider_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            # Skip GGUF models
            if "gguf" in model_dir.name.lower():
                continue
            # Must have config.json (MLX model)
            if not (model_dir / "config.json").exists():
                continue

            name = model_dir.name
            family, sz = _detect_base(name)
            if family == "unknown":
                continue

            if base_family and family.lower() != base_family.lower():
                continue
            if size and sz.lower() != size.lower():
                continue

            quant = _detect_quant(name)
            is_deriv, deriv_tag = _detect_derivative(name)

            # Build display name
            if is_deriv:
                display = f"{family} {sz} ({deriv_tag})"
            else:
                display = f"{family} {sz}"

            results.append(DiscoveredModel(
                path=str(model_dir),
                provider=provider_dir.name,
                dir_name=name,
                base_family=family,
                size=sz,
                quant=quant,
                is_derivative=is_deriv,
                derivative_tag=deriv_tag,
                display_name=display,
            ))

    return results


def group_models(models: list[DiscoveredModel]) -> list[ModelGroup]:
    """Group discovered models into benchmark families.

    Standard models (same base_family + size) are grouped together.
    Each derivative gets its own group for comparison.
    """
    groups: dict[str, ModelGroup] = {}

    for m in models:
        if m.is_derivative:
            # Group derivatives by family+size+tag (merge across providers)
            key = f"{m.base_family}_{m.size}_{m.derivative_tag}"
            if key not in groups:
                groups[key] = ModelGroup(
                    name=f"{m.base_family} {m.size} ({m.derivative_tag})",
                    base_family=m.base_family,
                    size=m.size,
                    is_standard=False,
                    derivative_tag=m.derivative_tag,
                )
            groups[key].models.append(m)
        else:
            # All standard models of same family+size together
            key = f"{m.base_family}_{m.size}_standard"
            if key not in groups:
                groups[key] = ModelGroup(
                    name=f"{m.base_family} {m.size}",
                    base_family=m.base_family,
                    size=m.size,
                    is_standard=True,
                    derivative_tag="",
                )
            groups[key].models.append(m)

    # Deduplicate: if multiple models in a group share the same quant,
    # prefer mlx-community, then first alphabetically.
    for group in groups.values():
        seen: dict[str, DiscoveredModel] = {}
        for m in group.models:
            existing = seen.get(m.quant)
            if existing is None:
                seen[m.quant] = m
            elif m.provider == "mlx-community" and existing.provider != "mlx-community":
                seen[m.quant] = m
        group.models = list(seen.values())

    return sorted(groups.values(), key=lambda g: (g.base_family, g.size, not g.is_standard, g.name))


def _pick_reference(models: list[DiscoveredModel]) -> str:
    """Pick the best reference variant (prefer bf16 > 8bit > mxfp8 > 4bit)."""
    pref = {"bf16": 0, "fp16": 1, "mxfp8": 2, "8bit": 3, "4bit": 4, "unknown": 5}
    best = min(models, key=lambda m: pref.get(m.quant, 99))
    return best.quant


def generate_toml(
    groups: list[ModelGroup],
    max_tokens: int = 2048,
    mmlu_size: str = "small",
) -> str:
    """Generate a TOML config from model groups."""
    lines = [
        "# Auto-generated config from LM Studio model discovery",
        "",
        "[benchmark]",
        "warmup_runs = 2",
        "measured_runs = 10",
        f"max_tokens = {max_tokens}",
        "temperature = 0.0",
        "randomize_order = true",
        'prompt_suite = "prompts/suite.toml"',
        'output_dir = "results"',
        "batch_sizes = []",
        "batch_runs = 3",
        f'mmlu_size = "{mmlu_size}"',
        "",
    ]

    for group in groups:
        if not group.models:
            continue

        tag = ""
        if group.is_standard:
            tag = "# Standard"
        else:
            tag = f"# Derivative: {group.derivative_tag}"

        lines.append(f"{tag}")
        lines.append("[[model_family]]")
        lines.append(f'name = "{group.name}"')
        lines.append('kind = "text"')
        lines.append(f'size = "{group.size}"')

        variant_lines = []
        for m in sorted(group.models, key=lambda x: {"bf16": 0, "8bit": 1, "mxfp8": 2, "4bit": 3}.get(x.quant, 9)):
            variant_lines.append(f'    {{ repo = "{m.path}", quant = "{m.quant}" }}')

        lines.append("variants = [")
        lines.append(",\n".join(variant_lines))
        lines.append("]")

        ref = _pick_reference(group.models)
        lines.append(f'reference = "{ref}"')
        lines.append("")

    return "\n".join(lines)


def print_discovery(models: list[DiscoveredModel]) -> None:
    """Print a human-readable summary of discovered models."""
    groups = group_models(models)

    for group in groups:
        marker = "  [standard]" if group.is_standard else f"  [{group.derivative_tag}]"
        print(f"\n{group.name}{marker}")
        for m in sorted(group.models, key=lambda x: x.quant):
            print(f"  {m.quant:8s}  {m.path}")
