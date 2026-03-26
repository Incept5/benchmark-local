"""Benchmark configuration loaded from TOML."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelVariant:
    repo: str
    quant: str


def _default_context_tokens(size: str) -> int:
    """Derive a default context token count from the model size label.

    Rules:
      < 2.5B  → 4096 tokens
      < 5B    → 8192 tokens
      >= 5B   → 16384 tokens

    The size string can be e.g. "0.8B", "3B", "35B-A3B", "70B".
    For MoE labels like "35B-A3B" we parse the first number.
    """
    m = re.search(r"([\d.]+)", size)
    if not m:
        return 4096
    param_b = float(m.group(1))
    if param_b < 2.5:
        return 4096
    if param_b < 5:
        return 8192
    return 16384


def _default_quality_temperature(name: str) -> float:
    """Return recommended temperature for quality evals based on model name.

    Qwen3.5 recommends temp=0.7 for non-thinking general tasks.
    Most other models work well at temp=0.0 for deterministic eval.
    """
    if "qwen3.5" in name.lower() or "qwen3_5" in name.lower():
        return 0.7
    return 0.0


@dataclass
class ModelFamily:
    name: str
    kind: str  # "text" or "vision"
    size: str
    variants: list[ModelVariant]
    reference: str  # quant label of the reference variant
    context_tokens: int = 0  # 0 = auto from size
    quality_temperature: float = -1.0  # -1 = auto from model name

    def __post_init__(self) -> None:
        if self.context_tokens <= 0:
            self.context_tokens = _default_context_tokens(self.size)
        if self.quality_temperature < 0:
            self.quality_temperature = _default_quality_temperature(self.name)

    def get_reference_variant(self) -> ModelVariant | None:
        for v in self.variants:
            if v.quant == self.reference:
                return v
        return None


def _effective_measured_runs(
    default_runs: int,
    large_runs: int,
    large_threshold_b: float,
    size: str,
) -> int:
    """Return fewer measured runs for large models to save time."""
    m = re.search(r"([\d.]+)", size)
    if m and float(m.group(1)) >= large_threshold_b:
        return large_runs
    return default_runs


# Prompts kept in quick mode — covers short input (short-qa) and
# medium/code output (code-gen).  These two span the extremes of
# prefill/decode behaviour without redundancy.  One vision prompt
# (vision-chart) is included so vision models aren't skipped entirely.
QUICK_PROMPT_IDS = ("short-qa", "code-gen", "vision-chart")

# Fixed context size in quick mode so all models are compared at the same input length.
QUICK_CONTEXT_TOKENS = 8192

# MMLU-Pro subset sizes — fraction of the full 12,032 question set.
MMLU_PRO_SIZES = {
    "tiny": 0.05,    # ~600 questions  (~30s on 9B)
    "small": 0.10,   # ~1200 questions (~60s on 9B)
    "medium": 0.25,  # ~3000 questions (~2.5m on 9B)
    "full": 1.0,     # all 12,032      (~10m on 9B)
}


@dataclass
class BenchmarkConfig:
    warmup_runs: int = 2
    measured_runs: int = 10
    large_model_measured_runs: int = 3
    large_model_threshold_b: float = 12.0
    max_tokens: int = 256
    temperature: float = 0.0
    randomize_order: bool = True
    prompt_suite: str = "prompts/suite.toml"
    output_dir: str = "results"
    batch_sizes: list[int] = field(default_factory=list)  # e.g. [1, 4, 8, 16]
    batch_runs: int = 3  # measured runs per batch size
    quick: bool = False  # --quick mode: 1 warmup, 1 run, 2 prompts
    mmlu_size: str = "small"  # tiny/small/medium/full
    config_name: str = ""  # stem of the config file (e.g. "qwen3.5-9b")
    model_families: list[ModelFamily] = field(default_factory=list)

    def effective_measured_runs(self, family: ModelFamily) -> int:
        """Return measured_runs, reduced for large models or quick mode."""
        if self.quick:
            return 1
        return _effective_measured_runs(
            self.measured_runs,
            self.large_model_measured_runs,
            self.large_model_threshold_b,
            family.size,
        )

    def effective_warmup_runs(self) -> int:
        """Return warmup_runs, reduced in quick mode."""
        if self.quick:
            return 1
        return self.warmup_runs

    @classmethod
    def from_toml(cls, path: str | Path) -> BenchmarkConfig:
        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        bench = data.get("benchmark", {})
        families = []
        for fam in data.get("model_family", []):
            variants = [
                ModelVariant(repo=v["repo"], quant=v["quant"])
                for v in fam["variants"]
            ]
            families.append(
                ModelFamily(
                    name=fam["name"],
                    kind=fam["kind"],
                    size=fam["size"],
                    variants=variants,
                    reference=fam["reference"],
                    context_tokens=fam.get("context_tokens", 0),
                    quality_temperature=fam.get("quality_temperature", -1.0),
                )
            )

        # Extract config name from filename, stripping common prefixes
        stem = path.stem
        for prefix in ("discovered-", "bench-", "config-"):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break

        return cls(
            warmup_runs=bench.get("warmup_runs", 2),
            measured_runs=bench.get("measured_runs", 10),
            large_model_measured_runs=bench.get("large_model_measured_runs", 3),
            large_model_threshold_b=bench.get("large_model_threshold_b", 8.0),
            max_tokens=bench.get("max_tokens", 256),
            temperature=bench.get("temperature", 0.0),
            randomize_order=bench.get("randomize_order", True),
            prompt_suite=bench.get("prompt_suite", "prompts/suite.toml"),
            output_dir=bench.get("output_dir", "results"),
            batch_sizes=bench.get("batch_sizes", []),
            batch_runs=bench.get("batch_runs", 3),
            mmlu_size=bench.get("mmlu_size", "tiny"),
            config_name=stem,
            model_families=families,
        )
