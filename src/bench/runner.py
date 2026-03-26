"""Orchestrates a full benchmark session."""

from __future__ import annotations

import gc
import logging
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from bench.config import BenchmarkConfig, ModelFamily, ModelVariant
from bench.batch import BatchResult, measure_batch
from bench.measure import RunResult, measure_one
from bench.models import load_model
from bench.power import PowerMonitor, PowerReading
from bench.prompts import Prompt, load_suite, pad_prompt_to_context
from bench.quality import (
    QualityResults,
    compute_output_similarity,
    compute_perplexity,
    eval_mmlu,
)
from bench.report import generate_report
from bench.report_md import generate_markdown_report
from bench.stats import AggregatedMetric, aggregate
from bench.store import SessionResult, SystemInfo, save_session

logger = logging.getLogger(__name__)


@dataclass
class VariantResult:
    variant: ModelVariant
    family_name: str
    kind: str
    runs: list[RunResult] = field(default_factory=list)
    batch_results: list[BatchResult] = field(default_factory=list)
    quality: QualityResults | None = None
    tool_calling: Any = None  # ToolCallEvalResults
    power: PowerReading | None = None
    aggregated: dict[str, AggregatedMetric] = field(default_factory=dict)
    reference_outputs: dict[str, str] = field(default_factory=dict)  # prompt_id -> text


@dataclass
class ProgressEvent:
    stage: str  # "loading", "warmup", "measuring", "quality", "done"
    family_name: str = ""
    variant_repo: str = ""
    variant_quant: str = ""
    prompt_id: str = ""
    run_index: int = 0
    total_runs: int = 0
    current_result: RunResult | None = None
    message: str = ""
    overall_progress: float = 0.0  # 0.0 to 1.0
    error: str = ""


ProgressCallback = Callable[[ProgressEvent], None]


def _default_progress(event: ProgressEvent) -> None:
    if event.error:
        logger.error("[%s] %s: %s", event.stage, event.family_name, event.error)
    elif event.message:
        logger.info("[%s] %s", event.stage, event.message)


def run_benchmark(
    config: BenchmarkConfig,
    on_progress: ProgressCallback | None = None,
) -> SessionResult:
    """Run a full benchmark session."""
    if on_progress is None:
        on_progress = _default_progress

    system_info = SystemInfo.detect()
    power_monitor = PowerMonitor()
    prompts = load_suite(config.prompt_suite)

    # In quick mode, keep only the representative subset of prompts
    if config.quick:
        from bench.config import QUICK_PROMPT_IDS
        prompts = [p for p in prompts if p.id in QUICK_PROMPT_IDS]
        logger.info("Quick mode: using %d prompts (%s)",
                     len(prompts), ", ".join(p.id for p in prompts))

    # Count total work units for progress tracking
    total_variants = sum(len(f.variants) for f in config.model_families)
    completed_variants = 0

    all_variant_results: list[VariantResult] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    bench_start = time.perf_counter()

    # Helper to save incremental results at any point
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_md_path = output_dir / "results-tmp.md"

    last_session: dict[str, Any] = {}  # mutable container for closure

    def _save_incremental(partial: bool = False) -> Path:
        bench_duration_s = time.perf_counter() - bench_start
        session = _build_session(
            timestamp=timestamp,
            system_info=system_info,
            config=config,
            variant_results=all_variant_results,
            duration_s=bench_duration_s,
        )
        last_session["value"] = session
        output_path = save_session(session, config.output_dir, config.config_name)
        html_path = output_path.with_suffix(".html")
        generate_report(session, html_path)
        if partial:
            # Write in-progress markdown only
            generate_markdown_report(session, tmp_md_path)
            logger.info("Incremental results saved to %s and %s", output_path, tmp_md_path)
        else:
            # Final: write timestamped markdown, clean up temp
            md_path = output_path.with_suffix(".md")
            generate_markdown_report(session, md_path)
            if tmp_md_path.exists():
                tmp_md_path.unlink()
            logger.info("Final results saved to %s (.json, .html, .md)", output_path)
        return output_path

    interrupted = False
    try:
        for family in config.model_families:
            # Filter prompts: text models skip vision prompts
            family_prompts = [
                p for p in prompts
                if family.kind == "vision" or not p.is_vision
            ]

            # Determine variant order
            variants = list(family.variants)
            if config.randomize_order:
                random.shuffle(variants)

            # Collect reference outputs for similarity comparison
            reference_outputs: dict[str, str] = {}
            family_variant_results: list[VariantResult] = []
            # Padded prompts are computed once per family using the first
            # variant's tokenizer, ensuring identical inputs across variants
            family_padded_prompts: list[Prompt] | None = None

            for variant in variants:
                vr = _run_variant(
                    config=config,
                    family=family,
                    variant=variant,
                    prompts=family_prompts,
                    padded_prompts=family_padded_prompts,
                    power_monitor=power_monitor,
                    on_progress=on_progress,
                    overall_progress=completed_variants / total_variants if total_variants > 0 else 0.0,
                )

                if vr is not None:
                    # Capture padded prompts from first variant for reuse
                    if family_padded_prompts is None and hasattr(vr, "_padded_prompts"):
                        family_padded_prompts = vr._padded_prompts

                    # If this is the reference variant, store its outputs
                    if variant.quant == family.reference:
                        for run in vr.runs:
                            if not run.is_warmup:
                                reference_outputs[run.prompt_id] = run.output_text

                    family_variant_results.append(vr)
                    all_variant_results.append(vr)

                    # Save after each completed variant
                    _save_incremental(partial=True)

                completed_variants += 1

            # Compute output similarity for non-reference variants
            if reference_outputs:
                for vr in family_variant_results:
                    if vr.variant.quant != family.reference:
                        similarities: dict[str, float] = {}
                        for run in vr.runs:
                            if not run.is_warmup and run.prompt_id in reference_outputs:
                                sim = compute_output_similarity(
                                    run.output_text, reference_outputs[run.prompt_id]
                                )
                                similarities[run.prompt_id] = sim
                        if vr.quality is None:
                            vr.quality = QualityResults()
                        vr.quality.output_similarity = similarities

    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Benchmark interrupted — saving partial results...")

    # Final save (or partial if interrupted)
    if all_variant_results:
        output_path = _save_incremental(partial=interrupted)

        md_info = f"\nMarkdown report: {tmp_md_path}" if interrupted else f"\nMarkdown report: {output_path.with_suffix('.md')}"
        on_progress(ProgressEvent(
            stage="done",
            message=(
                f"{'Partial results' if interrupted else 'Results'} saved to {output_path}\n"
                f"HTML report: {output_path.with_suffix('.html')}"
                f"{md_info}"
            ),
            overall_progress=1.0 if not interrupted else completed_variants / total_variants,
        ))
    else:
        on_progress(ProgressEvent(
            stage="done",
            message="No results were collected.",
            overall_progress=0.0,
        ))

    # Return the last saved session (avoid rebuilding)
    if "value" in last_session:
        return last_session["value"]
    # Nothing was saved — build a minimal session
    return _build_session(
        timestamp=timestamp,
        system_info=system_info,
        config=config,
        variant_results=[],
        duration_s=time.perf_counter() - bench_start,
    )


def _run_variant(
    config: BenchmarkConfig,
    family: ModelFamily,
    variant: ModelVariant,
    prompts: list[Prompt],
    padded_prompts: list[Prompt] | None,
    power_monitor: PowerMonitor,
    on_progress: ProgressCallback,
    overall_progress: float,
) -> VariantResult | None:
    """Run all benchmarks for a single variant."""
    on_progress(ProgressEvent(
        stage="loading",
        family_name=family.name,
        variant_repo=variant.repo,
        variant_quant=variant.quant,
        message=f"Loading {variant.repo}",
        overall_progress=overall_progress,
    ))

    try:
        model, tokenizer = load_model(variant, family.kind)
    except Exception as e:
        on_progress(ProgressEvent(
            stage="loading",
            family_name=family.name,
            variant_repo=variant.repo,
            variant_quant=variant.quant,
            error=f"Failed to load {variant.repo}: {e}",
            overall_progress=overall_progress,
        ))
        return None

    # Pad prompts to the family's target context size.
    # Reuse pre-padded prompts if provided (ensures identical inputs across variants).
    if padded_prompts is None:
        padded_prompts = [
            pad_prompt_to_context(p, family.context_tokens, tokenizer)
            for p in prompts
        ]
        logger.info(
            "Context target for %s (%s): %d tokens (padded with this variant's tokenizer)",
            family.name, family.size, family.context_tokens,
        )
    else:
        logger.info(
            "Context target for %s (%s): %d tokens (reusing family-level padded prompts)",
            family.name, family.size, family.context_tokens,
        )

    vr = VariantResult(
        variant=variant,
        family_name=family.name,
        kind=family.kind,
    )

    warmup_runs = config.effective_warmup_runs()
    measured_runs = config.effective_measured_runs(family)
    if measured_runs != config.measured_runs:
        logger.info("Large model (%s) — using %d measured runs instead of %d",
                     family.size, measured_runs, config.measured_runs)
    total_prompt_runs = len(padded_prompts) * (warmup_runs + measured_runs)
    current_run = 0

    # Start power measurement window for this variant
    power_window_name = f"{variant.repo}_{variant.quant}"
    power_monitor.begin_window(power_window_name)

    for prompt in padded_prompts:
        # Warmup
        for i in range(warmup_runs):
            current_run += 1
            on_progress(ProgressEvent(
                stage="warmup",
                family_name=family.name,
                variant_repo=variant.repo,
                variant_quant=variant.quant,
                prompt_id=prompt.id,
                run_index=i + 1,
                total_runs=warmup_runs,
                overall_progress=overall_progress,
            ))
            try:
                result = measure_one(
                    variant, family.kind, model, tokenizer, prompt,
                    prompt.max_tokens or config.max_tokens, config.temperature, is_warmup=True,
                )
                vr.runs.append(result)
                on_progress(ProgressEvent(
                    stage="warmup",
                    family_name=family.name,
                    variant_repo=variant.repo,
                    variant_quant=variant.quant,
                    prompt_id=prompt.id,
                    run_index=i + 1,
                    total_runs=warmup_runs,
                    current_result=result,
                    overall_progress=overall_progress,
                ))
            except Exception as e:
                logger.warning("Warmup run failed for %s on %s: %s", variant.repo, prompt.id, e)

        # Measured runs
        for i in range(measured_runs):
            current_run += 1
            on_progress(ProgressEvent(
                stage="measuring",
                family_name=family.name,
                variant_repo=variant.repo,
                variant_quant=variant.quant,
                prompt_id=prompt.id,
                run_index=i + 1,
                total_runs=measured_runs,
                overall_progress=overall_progress,
            ))
            try:
                result = measure_one(
                    variant, family.kind, model, tokenizer, prompt,
                    prompt.max_tokens or config.max_tokens, config.temperature, is_warmup=False,
                )
                vr.runs.append(result)
                on_progress(ProgressEvent(
                    stage="measuring",
                    family_name=family.name,
                    variant_repo=variant.repo,
                    variant_quant=variant.quant,
                    prompt_id=prompt.id,
                    run_index=i + 1,
                    total_runs=measured_runs,
                    current_result=result,
                    overall_progress=overall_progress,
                ))
            except Exception as e:
                logger.warning("Measured run failed for %s on %s: %s", variant.repo, prompt.id, e)

    # End power window
    vr.power = power_monitor.end_window(power_window_name)

    # Batch throughput benchmarking (text models only; mlx_vlm has no batch API)
    if family.kind == "text" and config.batch_sizes:
        prompt_texts = [p.text for p in padded_prompts]
        for batch_size in config.batch_sizes:
            on_progress(ProgressEvent(
                stage="measuring",
                family_name=family.name,
                variant_repo=variant.repo,
                variant_quant=variant.quant,
                message=f"Batch throughput (batch_size={batch_size})...",
                overall_progress=overall_progress,
            ))
            try:
                # Run multiple iterations and keep the best (most representative)
                best_result: BatchResult | None = None
                for run_i in range(config.batch_runs):
                    mx.reset_peak_memory()
                    br = measure_batch(
                        model, tokenizer, prompt_texts,
                        batch_size=batch_size,
                        max_tokens=config.max_tokens,
                        temperature=config.temperature,
                    )
                    if best_result is None or br.generation_tps > best_result.generation_tps:
                        best_result = br
                if best_result is not None:
                    vr.batch_results.append(best_result)
                    logger.info(
                        "Batch %dx: %.1f gen tok/s (prefill %.1f tok/s, %d MB peak)",
                        batch_size, best_result.generation_tps, best_result.prefill_tps,
                        best_result.peak_memory_bytes // (1024 ** 2),
                    )
            except Exception as e:
                logger.warning(
                    "Batch measurement (size=%d) failed for %s: %s",
                    batch_size, variant.repo, e,
                )

    # Quality evaluations (text models only for perplexity/MMLU)
    if family.kind == "text":
        on_progress(ProgressEvent(
            stage="quality",
            family_name=family.name,
            variant_repo=variant.repo,
            variant_quant=variant.quant,
            message="Computing perplexity...",
            overall_progress=overall_progress,
        ))
        try:
            wikitext_path = Path("evals/wikitext2_test.txt")
            if not wikitext_path.exists():
                logger.info("WikiText-2 test set not found — downloading...")
                try:
                    from datasets import load_dataset
                    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                    text = "\n".join(t for t in ds["text"] if t.strip())
                    wikitext_path.write_text(text)
                except Exception as dl_err:
                    logger.warning("Failed to download WikiText-2: %s — using bundled sample", dl_err)
                    wikitext_path = Path("evals/wikitext_sample.txt")
            wikitext = wikitext_path.read_text()
            ppl = compute_perplexity(model, tokenizer, wikitext)
        except Exception as e:
            logger.warning("Perplexity computation failed for %s: %s", variant.repo, e)
            ppl = None

        from bench.config import MMLU_PRO_SIZES
        mmlu_fraction = MMLU_PRO_SIZES.get(config.mmlu_size, 0.05)
        on_progress(ProgressEvent(
            stage="quality",
            family_name=family.name,
            variant_repo=variant.repo,
            variant_quant=variant.quant,
            message=f"Running MMLU-Pro evaluation ({config.mmlu_size})...",
            overall_progress=overall_progress,
        ))
        try:
            mmlu_acc, mmlu_correct, mmlu_total, mmlu_per_cat = eval_mmlu(
                model, tokenizer,
                fraction=mmlu_fraction,
                temperature=family.quality_temperature,
            )
        except Exception as e:
            logger.warning("MMLU-Pro evaluation failed for %s: %s", variant.repo, e)
            mmlu_acc, mmlu_correct, mmlu_total, mmlu_per_cat = None, 0, 0, {}

        # Tool calling evaluation
        on_progress(ProgressEvent(
            stage="quality",
            family_name=family.name,
            variant_repo=variant.repo,
            variant_quant=variant.quant,
            message="Running tool calling evaluation...",
            overall_progress=overall_progress,
        ))
        tool_call_results = None
        try:
            from bench.tool_calling import eval_tool_calling
            tc_path = Path("evals/tool_calling.toml")
            if tc_path.exists():
                tool_call_results = eval_tool_calling(model, tokenizer, tc_path)
        except Exception as e:
            logger.warning("Tool calling evaluation failed for %s: %s", variant.repo, e)

        vr.quality = QualityResults(
            perplexity=ppl,
            mmlu_accuracy=mmlu_acc,
            mmlu_correct=mmlu_correct,
            mmlu_total=mmlu_total,
            mmlu_per_category=mmlu_per_cat,
        )

        # Attach tool calling results
        if tool_call_results:
            vr.tool_calling = tool_call_results

    # Aggregate performance metrics per-prompt first, then combine.
    # Aggregating across prompts directly inflates CV% because different
    # prompts have different input lengths and output lengths.
    measured_runs = [r for r in vr.runs if not r.is_warmup]
    if measured_runs:
        # Group by prompt
        by_prompt: dict[str, list[RunResult]] = defaultdict(list)
        for r in measured_runs:
            by_prompt[r.prompt_id].append(r)

        # Per-prompt medians
        prompt_ttft_medians = []
        prompt_tps_medians = []
        prompt_prefill_medians = []
        prompt_decode_medians = []
        for prompt_id, runs in by_prompt.items():
            prompt_ttft = aggregate([r.ttft_ms for r in runs])
            prompt_tps = aggregate([r.tokens_per_sec for r in runs])
            prompt_prefill = aggregate([r.prefill_tps for r in runs if r.prefill_tps > 0])
            prompt_decode = aggregate([r.decode_tps for r in runs if r.decode_tps > 0])
            prompt_ttft_medians.append(prompt_ttft.median)
            prompt_tps_medians.append(prompt_tps.median)
            if prompt_prefill.n > 0:
                prompt_prefill_medians.append(prompt_prefill.median)
            if prompt_decode.n > 0:
                prompt_decode_medians.append(prompt_decode.median)

        vr.aggregated["ttft_ms"] = aggregate(prompt_ttft_medians)
        vr.aggregated["tokens_per_sec"] = aggregate(prompt_tps_medians)
        if prompt_prefill_medians:
            vr.aggregated["prefill_tps"] = aggregate(prompt_prefill_medians)
        if prompt_decode_medians:
            vr.aggregated["decode_tps"] = aggregate(prompt_decode_medians)
        vr.aggregated["peak_memory_bytes"] = aggregate(
            [float(r.peak_memory_bytes) for r in measured_runs]
        )

        # Tokens per watt
        if vr.power and vr.power.avg_watts > 0:
            median_tps = vr.aggregated["tokens_per_sec"].median
            tokens_per_watt = median_tps / vr.power.avg_watts
            vr.aggregated["tokens_per_watt"] = aggregate([tokens_per_watt])

    # Stash padded prompts for reuse by subsequent variants in the same family
    vr._padded_prompts = padded_prompts  # type: ignore[attr-defined]

    # Unload model to free memory
    del model, tokenizer
    gc.collect()
    mx.reset_peak_memory()

    return vr


def _build_session(
    timestamp: str,
    system_info: SystemInfo,
    config: BenchmarkConfig,
    variant_results: list[VariantResult],
    duration_s: float = 0.0,
) -> SessionResult:
    """Build the final session result."""
    runs_data = []
    quality_data: dict[str, Any] = {}
    aggregated_data: dict[str, Any] = {}
    power_data: dict[str, Any] = {}
    tool_calling_data: dict[str, Any] = {}
    batch_data: dict[str, list[dict]] = {}

    for vr in variant_results:
        key = f"{vr.variant.repo}|{vr.variant.quant}"

        for run in vr.runs:
            runs_data.append(asdict(run))

        if vr.quality:
            quality_data[key] = asdict(vr.quality)

        if vr.tool_calling:
            tool_calling_data[key] = {
                "total": vr.tool_calling.total,
                "json_valid_rate": vr.tool_calling.json_valid_rate,
                "function_accuracy": vr.tool_calling.function_accuracy,
                "param_accuracy": vr.tool_calling.param_accuracy,
                "refusal_accuracy": vr.tool_calling.refusal_accuracy,
                "overall_accuracy": vr.tool_calling.overall_accuracy,
                "results": [asdict(r) for r in vr.tool_calling.results],
            }

        if vr.aggregated:
            agg = {}
            for metric_name, metric_val in vr.aggregated.items():
                agg[metric_name] = asdict(metric_val)
            aggregated_data[key] = agg

        if vr.power:
            power_data[key] = asdict(vr.power)

        if vr.batch_results:
            batch_data[key] = [asdict(br) for br in vr.batch_results]

    config_snapshot = {
        "warmup_runs": config.warmup_runs,
        "quick": config.quick,
        "measured_runs": config.measured_runs,
        "large_model_measured_runs": config.large_model_measured_runs,
        "large_model_threshold_b": config.large_model_threshold_b,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "randomize_order": config.randomize_order,
        "batch_sizes": config.batch_sizes,
        "batch_runs": config.batch_runs,
        "mmlu_size": config.mmlu_size,
        "model_families": [
            {
                "name": f.name,
                "kind": f.kind,
                "size": f.size,
                "variants": [{"repo": v.repo, "quant": v.quant} for v in f.variants],
                "reference": f.reference,
                "context_tokens": f.context_tokens,
                "quality_temperature": f.quality_temperature,
            }
            for f in config.model_families
        ],
    }

    return SessionResult(
        timestamp=timestamp,
        duration_s=duration_s,
        system_info=asdict(system_info),
        config_snapshot=config_snapshot,
        runs=runs_data,
        quality=quality_data,
        tool_calling=tool_calling_data,
        aggregated=aggregated_data,
        power=power_data,
        batch=batch_data,
    )
