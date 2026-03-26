"""Generate a Markdown report from benchmark results."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from bench.store import SessionResult

logger = logging.getLogger(__name__)


def generate_markdown_report(session: SessionResult, output_path: str | Path) -> Path:
    """Generate a Markdown report and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = _build_markdown(session)
    output_path.write_text(doc)
    return output_path


def _fmt(val: Any, fmt: str = ".1f", suffix: str = "") -> str:
    if val is None:
        return "—"
    try:
        return f"{val:{fmt}}{suffix}"
    except (ValueError, TypeError):
        return str(val)


def _short_name(key: str, variant_info: dict[str, dict]) -> str:
    info = variant_info.get(key, {})
    repo = info.get("repo", key.split("|")[0])
    return repo.split("/")[-1]


def _build_markdown(session: SessionResult) -> str:
    sys_info = session.system_info
    cfg = session.config_snapshot
    agg = session.aggregated
    tc = session.tool_calling
    qual = session.quality
    pwr = session.power
    families = cfg.get("model_families", [])

    # Build variant lookup
    variant_info: dict[str, dict] = {}
    family_variants: dict[str, list[str]] = defaultdict(list)
    for fam in families:
        for var in fam.get("variants", []):
            key = f"{var['repo']}|{var['quant']}"
            variant_info[key] = {
                "family": fam["name"],
                "size": fam["size"],
                "kind": fam["kind"],
                "quant": var["quant"],
                "repo": var["repo"],
                "reference": fam["reference"],
                "context_tokens": fam.get("context_tokens", 0),
            }
            family_variants[fam["name"]].append(key)

    lines: list[str] = []

    # --- Header ---
    lines.append("# MLX Benchmark Results")
    lines.append("")

    # Timestamp
    ts = session.timestamp
    if ts:
        try:
            dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
            ts_display = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except ValueError:
            ts_display = ts
        duration = session.duration_s
        if duration > 0:
            mins, secs = divmod(int(duration), 60)
            hours, mins = divmod(mins, 60)
            if hours > 0:
                dur_str = f"{hours}h {mins}m {secs}s"
            elif mins > 0:
                dur_str = f"{mins}m {secs}s"
            else:
                dur_str = f"{secs}s"
            ts_display += f" ({dur_str})"
        lines.append(f"**Run:** {ts_display}")
        lines.append("")

    # --- System Info ---
    lines.append("## System")
    lines.append("")
    lines.append(f"| | |")
    lines.append(f"|---|---|")
    lines.append(f"| **Chip** | {sys_info.get('chip', 'Unknown')} |")
    lines.append(f"| **Memory** | {sys_info.get('memory_gb', '?')} GB |")
    lines.append(f"| **OS** | {sys_info.get('os_version', 'Unknown')} |")
    lines.append(f"| **Python** | {sys_info.get('python_version', '?')} |")
    lines.append(f"| **MLX** | {sys_info.get('mlx_version', '?') or '—'} |")
    lines.append(f"| **mlx_lm** | {sys_info.get('mlx_lm_version', '?') or '—'} |")
    lines.append(f"| **mlx_vlm** | {sys_info.get('mlx_vlm_version', '?') or '—'} |")
    lines.append("")

    lines.append("### Benchmark Parameters")
    lines.append("")
    lines.append(f"| | |")
    lines.append(f"|---|---|")
    lines.append(f"| Warmup runs | {cfg.get('warmup_runs', '?')} |")
    lines.append(f"| Measured runs | {cfg.get('measured_runs', '?')} |")
    lines.append(f"| Max tokens | {cfg.get('max_tokens', '?')} |")
    lines.append(f"| Temperature | {cfg.get('temperature', '?')} |")
    lines.append(f"| MMLU-Pro | {cfg.get('mmlu_size', 'tiny')} |")
    lines.append("")

    # --- Architecture Notes ---
    arch_notes = _architecture_notes(families)
    if arch_notes:
        lines.append("## Architecture Notes")
        lines.append("")
        lines.append(arch_notes)
        lines.append("")

    # --- Summary Table ---
    if agg:
        lines.append("## Summary")
        lines.append("")
        lines.append(_summary_intro())
        lines.append("")
        lines.append(_summary_table(agg, qual, pwr, tc, variant_info))
        lines.append("")

    # --- Quantization Comparison by Family ---
    has_family_data = any(
        any(k in agg for k in keys) for keys in family_variants.values()
    )
    if has_family_data:
        lines.append("## Quantization Comparison")
        lines.append("")
        lines.append(_quantization_intro())
        lines.append("")

        for fam_name, keys in family_variants.items():
            valid_keys = [k for k in keys if k in agg]
            if not valid_keys:
                continue
            lines.append(f"### {fam_name}")
            lines.append("")
            lines.append(_family_table(fam_name, valid_keys, agg, qual, variant_info))
            lines.append("")
            commentary = _family_commentary(fam_name, valid_keys, agg, qual, variant_info)
            if commentary:
                lines.append(commentary)
                lines.append("")

    # --- Derivative Comparison ---
    deriv_section = _derivative_comparison(agg, qual, variant_info, family_variants)
    if deriv_section:
        lines.append("## Derivative Comparison")
        lines.append("")
        lines.append(
            "> Compares fine-tuned, distilled, and abliterated derivatives against "
            "the standard base model at matched quantization levels. "
            "Differences in MMLU-Pro and perplexity show quality impact; "
            "speed differences reveal architecture or tokenizer changes."
        )
        lines.append("")
        lines.append(deriv_section)
        lines.append("")

    # --- Batch Throughput ---
    batch = session.batch
    if batch:
        lines.append("## Batch Throughput")
        lines.append("")
        lines.append(
            "> Measures aggregate throughput with multiple concurrent requests using "
            "`mlx_lm.batch_generate`. Higher batch sizes move more data through the "
            "memory bus per cycle, increasing total tokens/sec — but per-request latency "
            "may increase. This reflects server-style workloads, not single-user chat."
        )
        lines.append("")
        lines.append(_batch_table(batch, agg, variant_info))
        lines.append("")

    # --- Power ---
    if pwr:
        lines.append("## Power Consumption")
        lines.append("")
        lines.append(
            "> Energy measured via Apple Silicon power counters (zeus-ml). "
            "Lower watts at the same throughput = better efficiency and longer battery life."
        )
        lines.append("")
        lines.append(_power_table(pwr, variant_info))
        lines.append("")

    # --- Tool Calling ---
    if tc:
        lines.append("## Tool Calling")
        lines.append("")
        lines.append(
            "> Tests whether models can act as AI agents — selecting the right "
            "function and extracting correct parameters from natural language."
        )
        lines.append("")
        lines.append(_tool_calling_table(tc, variant_info))
        lines.append("")

    # --- Overall Analysis ---
    if agg and len(family_variants) > 0:
        lines.append("## Analysis: Impact of Quantization")
        lines.append("")
        lines.append(_quantization_analysis(agg, qual, pwr, variant_info, family_variants))
        lines.append("")

    # --- Key ---
    lines.append("## Metric Definitions")
    lines.append("")
    lines.append("| Metric | Description |")
    lines.append("|--------|-------------|")
    lines.append("| **TTFT** | Time to first token in milliseconds — latency before the model starts responding (lower is better) |")
    lines.append("| **Prefill** | Prompt processing speed in tokens/sec — how fast the model reads your input (higher is better) |")
    lines.append("| **Decode** | Token generation speed in tokens/sec — the streaming speed you experience (higher is better) |")
    lines.append("| **tok/W** | Tokens per watt — energy efficiency (higher is better) |")
    lines.append("| **Mem** | Peak GPU memory usage (lower = fits on more devices) |")
    lines.append("| **Perplexity** | Language model quality on WikiText-2 — how well it predicts text (lower is better) |")
    lines.append("| **MMLU-Pro** | Accuracy on MMLU-Pro 10-choice knowledge benchmark across 14 categories (harder than MMLU; higher is better) |")
    lines.append("| **Output Sim.** | Token-level F1 vs reference variant output (1.0 = identical, measures quantization drift) |")
    lines.append("| **Batch Gen tok/s** | Aggregate generation throughput with concurrent requests via `batch_generate` (higher is better) |")
    lines.append("| **vs single** | Speedup of batch generation compared to single-request decode — shows the benefit of batching |")
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated by [MacOS-MLX-Benchmark](https://github.com/Incept5/MacOS-MLX-Benchmark)*")
    lines.append("")

    return "\n".join(lines)


def _summary_intro() -> str:
    return (
        "> Overall performance and quality for each model variant. "
        "**TTFT** = time before first word appears. "
        "**Prefill** = prompt reading speed. "
        "**Decode** = response generation speed (the speed you feel). "
        "**Perplexity** = language understanding (lower = smarter). "
        "**MMLU-Pro** = knowledge accuracy (higher = smarter)."
    )


def _architecture_notes(families: list[dict]) -> str:
    """Generate architecture notes for non-standard models."""
    notes: list[str] = []

    qwen35_families = [f for f in families if "qwen3.5" in f.get("name", "").lower()]
    if qwen35_families:
        names = ", ".join(f["name"] for f in qwen35_families)
        notes.append(
            f"> **{names}**: These use a **hybrid DeltaNet + Attention architecture** — "
            "not a standard transformer. Each block of 4 layers has 3 Gated DeltaNet "
            "(linear attention) layers and 1 full attention layer. This means:"
        )
        notes.append(">")
        notes.append(
            "> - **Faster inference at long contexts** — linear attention scales O(n) "
            "vs O(n\u00b2) for full attention"
        )
        notes.append(
            "> - **Different quantization behaviour** — DeltaNet layers may respond differently "
            "to quantization than standard attention, so quality/speed trade-offs are not directly "
            "comparable to pure transformer models"
        )
        notes.append(
            "> - **Built-in vision encoder** — Qwen3.5 models are natively multimodal "
            "(image-text-to-text) even when used for text-only tasks"
        )
        notes.append(">")
        notes.append(
            "> Thinking mode has been **disabled** for these benchmarks to ensure "
            "throughput and quality measurements reflect actual output, not internal "
            "reasoning chains."
        )

    return "\n".join(notes)


def _quantization_intro() -> str:
    return (
        "> Quantization compresses model weights from full precision (bf16/fp16) to fewer bits (8-bit or 4-bit). "
        "This **reduces memory** and typically **increases speed** because less data moves through the memory bus. "
        "The trade-off is quality loss — the model becomes slightly less accurate. "
        "The **reference** variant (usually bf16) is the quality baseline. "
        "**vs ref** columns show the relative change."
    )


def _summary_table(
    agg: dict, qual: dict, pwr: dict, tc: dict, variant_info: dict[str, dict]
) -> str:
    lines: list[str] = []

    has_tools = bool(tc)
    header = "| Model | Quant | Context | TTFT (ms) | Prefill | Decode | tok/W | Mem (MB) | Perplexity | MMLU-Pro |"
    sep = "|-------|-------|--------:|----------:|--------:|-------:|------:|---------:|-----------:|---------:|"
    if has_tools:
        header += " Tools |"
        sep += "------:|"

    lines.append(header)
    lines.append(sep)

    for key, metrics in agg.items():
        info = variant_info.get(key, {})
        name = _short_name(key, variant_info)
        quant = info.get("quant", key.split("|")[-1])
        ctx = info.get("context_tokens", 0)
        ctx_str = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)

        ttft = _v(metrics, "ttft_ms", "median")
        prefill = _v(metrics, "prefill_tps", "median")
        decode = _v(metrics, "decode_tps", "median")
        tpw = _v(metrics, "tokens_per_watt", "median")
        mem = _v(metrics, "peak_memory_bytes", "median")
        mem_mb = mem / (1024 ** 2) if mem else None

        q = qual.get(key, {})
        ppl = q.get("perplexity")
        mmlu = q.get("mmlu_accuracy")

        t = tc.get(key, {}) if tc else {}
        tool_acc = t.get("overall_accuracy")

        row = (
            f"| {name} | {quant} "
            f"| {ctx_str} "
            f"| {_fmt(ttft, '.1f')} "
            f"| {_fmt(prefill, '.1f')} "
            f"| {_fmt(decode, '.1f')} "
            f"| {_fmt(tpw, '.2f')} "
            f"| {_fmt(mem_mb, '.0f')} "
            f"| {_fmt(ppl, '.2f')} "
            f"| {_fmt(mmlu * 100 if mmlu is not None else None, '.1f', '%')} |"
        )
        if has_tools:
            row += f" {_fmt(tool_acc * 100 if tool_acc is not None else None, '.1f', '%')} |"
        lines.append(row)

    return "\n".join(lines)


def _v(metrics: dict, metric_name: str, field: str = "median") -> Any:
    """Safe nested access: metrics[metric_name][field]."""
    m = metrics.get(metric_name)
    if isinstance(m, dict):
        return m.get(field)
    return None


def _family_table(
    fam_name: str,
    keys: list[str],
    agg: dict,
    qual: dict,
    variant_info: dict[str, dict],
) -> str:
    ref_quant = variant_info.get(keys[0], {}).get("reference", "")
    ref_key = next((k for k in keys if variant_info.get(k, {}).get("quant") == ref_quant), None)

    ref_tps = _v(agg.get(ref_key, {}), "tokens_per_sec", "median") if ref_key else None
    ref_mem = _v(agg.get(ref_key, {}), "peak_memory_bytes", "median") if ref_key else None
    ref_ppl = qual.get(ref_key, {}).get("perplexity") if ref_key else None

    lines: list[str] = []
    lines.append("| Quant | tok/s | vs ref | Mem (MB) | vs ref | Perplexity | PPL delta | Output Sim. |")
    lines.append("|-------|------:|-------:|---------:|-------:|-----------:|----------:|------------:|")

    for key in keys:
        info = variant_info.get(key, {})
        quant = info.get("quant", "?")
        metrics = agg.get(key, {})
        q = qual.get(key, {})

        tps = _v(metrics, "tokens_per_sec", "median")
        mem = _v(metrics, "peak_memory_bytes", "median")
        ppl = q.get("perplexity")

        speed_vs = f"{tps / ref_tps:.2f}x" if tps and ref_tps else "—"
        mem_vs = f"{mem / ref_mem:.2f}x" if mem and ref_mem else "—"

        ppl_delta = "—"
        if ppl and ref_ppl and ref_ppl > 0:
            change = ((ppl - ref_ppl) / ref_ppl) * 100
            sign = "+" if change >= 0 else ""
            ppl_delta = f"{sign}{change:.1f}%"
        elif quant == ref_quant:
            ppl_delta = "ref"

        sim = q.get("output_similarity", {})
        avg_sim = f"{sum(sim.values()) / len(sim):.3f}" if sim else "—"

        is_ref = quant == ref_quant
        quant_label = f"**{quant}** (ref)" if is_ref else quant

        lines.append(
            f"| {quant_label} "
            f"| {_fmt(tps, '.1f')} "
            f"| {speed_vs} "
            f"| {_fmt(mem / (1024**2) if mem else None, '.0f')} "
            f"| {mem_vs} "
            f"| {_fmt(ppl, '.2f')} "
            f"| {ppl_delta} "
            f"| {avg_sim} |"
        )

    return "\n".join(lines)


def _family_commentary(
    fam_name: str,
    keys: list[str],
    agg: dict,
    qual: dict,
    variant_info: dict[str, dict],
) -> str:
    """Generate a short per-family insight about quantization trade-offs."""
    if len(keys) < 2:
        return ""

    ref_quant = variant_info.get(keys[0], {}).get("reference", "")
    ref_key = next((k for k in keys if variant_info.get(k, {}).get("quant") == ref_quant), None)
    if not ref_key:
        return ""

    ref_tps = _v(agg.get(ref_key, {}), "tokens_per_sec", "median")
    ref_mem = _v(agg.get(ref_key, {}), "peak_memory_bytes", "median")
    ref_ppl = qual.get(ref_key, {}).get("perplexity")

    insights: list[str] = []
    for key in keys:
        info = variant_info.get(key, {})
        quant = info.get("quant", "?")
        if quant == ref_quant:
            continue

        tps = _v(agg.get(key, {}), "tokens_per_sec", "median")
        mem = _v(agg.get(key, {}), "peak_memory_bytes", "median")
        ppl = qual.get(key, {}).get("perplexity")

        parts: list[str] = []
        if tps and ref_tps:
            speedup = ((tps - ref_tps) / ref_tps) * 100
            if speedup > 0:
                parts.append(f"{speedup:.0f}% faster")
            else:
                parts.append(f"{abs(speedup):.0f}% slower")

        if mem and ref_mem:
            savings = ((ref_mem - mem) / ref_mem) * 100
            if savings > 0:
                parts.append(f"{savings:.0f}% less memory")
            else:
                parts.append(f"{abs(savings):.0f}% more memory")

        if ppl and ref_ppl and ref_ppl > 0:
            ppl_change = ((ppl - ref_ppl) / ref_ppl) * 100
            if ppl_change < 2:
                parts.append("negligible quality loss")
            elif ppl_change < 5:
                parts.append(f"moderate quality loss ({ppl_change:.1f}% PPL increase)")
            else:
                parts.append(f"significant quality loss ({ppl_change:.1f}% PPL increase)")

        if parts:
            insights.append(f"- **{quant}** vs **{ref_quant}**: {', '.join(parts)}")

    if not insights:
        return ""
    return "\n".join(insights)


def _derivative_comparison(
    agg: dict, qual: dict,
    variant_info: dict[str, dict],
    family_variants: dict[str, list[str]],
) -> str:
    """Build a comparison table of derivatives vs standard base model.

    Groups families by size, identifies the standard model as baseline,
    and shows deltas for derivatives at each matching quant level.
    """
    import re

    # Group families by size
    size_families: dict[str, list[str]] = defaultdict(list)
    for fam_name in family_variants:
        # Get size from any variant in this family
        for key in family_variants[fam_name]:
            info = variant_info.get(key, {})
            if info.get("size"):
                size_families[info["size"]].append(fam_name)
                break

    # Only proceed for sizes that have both standard and derivative families
    sections: list[str] = []
    for size, fam_names in sorted(size_families.items()):
        if len(fam_names) < 2:
            continue

        # Find standard family (no parenthetical tag in name)
        standard_fam = None
        derivative_fams = []
        for fn in fam_names:
            if "(" in fn:
                derivative_fams.append(fn)
            else:
                standard_fam = fn

        if not standard_fam or not derivative_fams:
            continue

        # Build quant→key lookup for standard
        std_by_quant: dict[str, str] = {}
        for key in family_variants.get(standard_fam, []):
            info = variant_info.get(key, {})
            if key in agg:
                std_by_quant[info.get("quant", "")] = key

        if not std_by_quant:
            continue

        lines: list[str] = []
        lines.append(f"### {size} Models")
        lines.append("")
        lines.append(f"Baseline: **{standard_fam}**")
        lines.append("")
        lines.append("| Variant | Quant | Decode | vs Std | MMLU-Pro | vs Std | Perplexity | vs Std | Mem (MB) |")
        lines.append("|---------|-------|-------:|-------:|---------:|-------:|-----------:|-------:|---------:|")

        # Standard rows first
        for quant in ["bf16", "fp16", "mxfp8", "8bit", "4bit"]:
            key = std_by_quant.get(quant)
            if not key:
                continue
            m = agg.get(key, {})
            q = qual.get(key, {})
            decode = m.get("decode_tps", {}).get("median")
            mmlu = q.get("mmlu_accuracy")
            ppl = q.get("perplexity")
            mem = m.get("peak_memory_bytes", {}).get("median")
            lines.append(
                f"| {standard_fam} | {quant} "
                f"| {_fmt(decode, '.1f', ' t/s')} | — "
                f"| {_fmt(mmlu * 100 if mmlu is not None else None, '.1f', '%')} | — "
                f"| {_fmt(ppl, '.2f')} | — "
                f"| {_fmt(mem / 1024**2 if mem else None, '.0f')} |"
            )

        # Derivative rows with deltas
        for deriv_fam in sorted(derivative_fams):
            for key in family_variants.get(deriv_fam, []):
                if key not in agg:
                    continue
                info = variant_info.get(key, {})
                quant = info.get("quant", "")
                m = agg.get(key, {})
                q = qual.get(key, {})
                decode = m.get("decode_tps", {}).get("median")
                mmlu = q.get("mmlu_accuracy")
                ppl = q.get("perplexity")
                mem = m.get("peak_memory_bytes", {}).get("median")

                # Find matching standard quant for delta
                std_key = std_by_quant.get(quant)
                decode_delta = ""
                mmlu_delta = ""
                ppl_delta = ""
                if std_key:
                    sm = agg.get(std_key, {})
                    sq = qual.get(std_key, {})
                    std_decode = sm.get("decode_tps", {}).get("median")
                    std_mmlu = sq.get("mmlu_accuracy")
                    std_ppl = sq.get("perplexity")

                    if decode and std_decode and std_decode > 0:
                        pct = ((decode - std_decode) / std_decode) * 100
                        decode_delta = f"{pct:+.1f}%"
                    if mmlu is not None and std_mmlu is not None:
                        diff = (mmlu - std_mmlu) * 100
                        decode_delta_str = decode_delta or "—"
                        mmlu_delta = f"{diff:+.1f}pp"
                    if ppl and std_ppl and std_ppl > 0:
                        pct = ((ppl - std_ppl) / std_ppl) * 100
                        ppl_delta = f"{pct:+.1f}%"

                # Extract short derivative name
                m_tag = re.search(r"\((.+?)\)", deriv_fam)
                short_name = m_tag.group(1) if m_tag else deriv_fam

                lines.append(
                    f"| {short_name} | {quant} "
                    f"| {_fmt(decode, '.1f', ' t/s')} | {decode_delta or '—'} "
                    f"| {_fmt(mmlu * 100 if mmlu is not None else None, '.1f', '%')} | {mmlu_delta or '—'} "
                    f"| {_fmt(ppl, '.2f')} | {ppl_delta or '—'} "
                    f"| {_fmt(mem / 1024**2 if mem else None, '.0f')} |"
                )

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _batch_table(batch: dict, agg: dict, variant_info: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("| Model | Quant | Batch | Gen tok/s | Prefill tok/s | vs single | Mem (MB) |")
    lines.append("|-------|-------|------:|----------:|--------------:|----------:|---------:|")

    for key, results in batch.items():
        name = _short_name(key, variant_info)
        info = variant_info.get(key, {})
        quant = info.get("quant", key.split("|")[-1])

        # Get single-request decode speed for comparison
        single_tps = _v(agg.get(key, {}), "decode_tps", "median") if agg else None

        for br in results:
            bs = br.get("batch_size", 0)
            gen_tps = br.get("generation_tps", 0)
            pre_tps = br.get("prefill_tps", 0)
            mem = br.get("peak_memory_bytes", 0)
            mem_mb = mem / (1024 ** 2) if mem else None

            if single_tps and single_tps > 0:
                speedup = f"{gen_tps / single_tps:.2f}x"
            else:
                speedup = "—"

            lines.append(
                f"| {name} | {quant} "
                f"| {bs} "
                f"| {_fmt(gen_tps, '.1f')} "
                f"| {_fmt(pre_tps, '.1f')} "
                f"| {speedup} "
                f"| {_fmt(mem_mb, '.0f')} |"
            )

    return "\n".join(lines)


def _power_table(pwr: dict, variant_info: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("| Model | Quant | Avg W | Total J | Duration (s) | CPU (W) | GPU (W) |")
    lines.append("|-------|-------|------:|--------:|-------------:|--------:|--------:|")

    for key, data in pwr.items():
        name = _short_name(key, variant_info)
        info = variant_info.get(key, {})
        quant = info.get("quant", key.split("|")[-1])
        comps = data.get("components", {})

        lines.append(
            f"| {name} | {quant} "
            f"| {_fmt(data.get('avg_watts'), '.1f')} "
            f"| {_fmt(data.get('total_joules'), '.1f')} "
            f"| {_fmt(data.get('duration_s'), '.1f')} "
            f"| {_fmt(comps.get('cpu'), '.1f')} "
            f"| {_fmt(comps.get('gpu'), '.1f')} |"
        )

    return "\n".join(lines)


def _tool_calling_table(tc: dict, variant_info: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("| Model | Quant | JSON Valid | Function Acc. | Param Acc. | Refusal Acc. | Overall | N |")
    lines.append("|-------|-------|----------:|--------------:|-----------:|-------------:|--------:|--:|")

    for key, data in tc.items():
        name = _short_name(key, variant_info)
        info = variant_info.get(key, {})
        quant = info.get("quant", key.split("|")[-1])

        lines.append(
            f"| {name} | {quant} "
            f"| {_fmt(data.get('json_valid_rate', 0) * 100, '.1f', '%')} "
            f"| {_fmt(data.get('function_accuracy', 0) * 100, '.1f', '%')} "
            f"| {_fmt(data.get('param_accuracy', 0) * 100, '.1f', '%')} "
            f"| {_fmt(data.get('refusal_accuracy', 0) * 100, '.1f', '%')} "
            f"| {_fmt(data.get('overall_accuracy', 0) * 100, '.1f', '%')} "
            f"| {data.get('total', 0)} |"
        )

    return "\n".join(lines)


def _quantization_analysis(
    agg: dict,
    qual: dict,
    pwr: dict,
    variant_info: dict[str, dict],
    family_variants: dict[str, list[str]],
) -> str:
    """Generate an overall analysis section about quantization trade-offs."""
    lines: list[str] = []

    lines.append(
        "Quantization reduces model weights from full precision (16-bit floats) to "
        "fewer bits per weight. This is the single most impactful optimisation for "
        "running LLMs locally on Apple Silicon."
    )
    lines.append("")

    # Collect aggregate statistics across all families
    speed_gains_8bit: list[float] = []
    speed_gains_4bit: list[float] = []
    mem_savings_8bit: list[float] = []
    mem_savings_4bit: list[float] = []
    ppl_changes_8bit: list[float] = []
    ppl_changes_4bit: list[float] = []

    for fam_name, keys in family_variants.items():
        ref_quant = variant_info.get(keys[0], {}).get("reference", "")
        ref_key = next((k for k in keys if variant_info.get(k, {}).get("quant") == ref_quant), None)
        if not ref_key or ref_key not in agg:
            continue

        ref_tps = _v(agg.get(ref_key, {}), "tokens_per_sec", "median")
        ref_mem = _v(agg.get(ref_key, {}), "peak_memory_bytes", "median")
        ref_ppl = qual.get(ref_key, {}).get("perplexity")

        for key in keys:
            info = variant_info.get(key, {})
            quant = info.get("quant", "")
            if quant == ref_quant:
                continue
            if key not in agg:
                continue

            tps = _v(agg.get(key, {}), "tokens_per_sec", "median")
            mem = _v(agg.get(key, {}), "peak_memory_bytes", "median")
            ppl = qual.get(key, {}).get("perplexity")

            if tps and ref_tps and ref_tps > 0:
                gain = ((tps - ref_tps) / ref_tps) * 100
                if "8" in quant:
                    speed_gains_8bit.append(gain)
                elif "4" in quant:
                    speed_gains_4bit.append(gain)

            if mem and ref_mem and ref_mem > 0:
                saving = ((ref_mem - mem) / ref_mem) * 100
                if "8" in quant:
                    mem_savings_8bit.append(saving)
                elif "4" in quant:
                    mem_savings_4bit.append(saving)

            if ppl and ref_ppl and ref_ppl > 0:
                change = ((ppl - ref_ppl) / ref_ppl) * 100
                if "8" in quant:
                    ppl_changes_8bit.append(change)
                elif "4" in quant:
                    ppl_changes_4bit.append(change)

    lines.append("### Why quantize?")
    lines.append("")
    lines.append(
        "Apple Silicon uses **unified memory** shared between CPU and GPU. A model "
        "that doesn't fit in memory can't run at all — there's no swap-to-disk fallback "
        "that preserves usable speed. Quantization directly determines which models fit "
        "on your machine:"
    )
    lines.append("")
    lines.append("| Precision | Bits/weight | ~Memory for 7B model |")
    lines.append("|-----------|-------------|----------------------|")
    lines.append("| bf16/fp16 | 16 bits | ~14 GB |")
    lines.append("| 8-bit | 8 bits | ~7 GB |")
    lines.append("| 4-bit | 4 bits | ~3.5 GB |")
    lines.append("")

    lines.append("### What this benchmark found")
    lines.append("")

    if speed_gains_8bit:
        avg = sum(speed_gains_8bit) / len(speed_gains_8bit)
        lines.append(
            f"**8-bit quantization** — averaged {avg:+.0f}% speed change vs reference across "
            f"{len(speed_gains_8bit)} model(s)."
        )
        if mem_savings_8bit:
            avg_mem = sum(mem_savings_8bit) / len(mem_savings_8bit)
            lines.append(f"  Memory savings averaged {avg_mem:.0f}%.")
        if ppl_changes_8bit:
            avg_ppl = sum(ppl_changes_8bit) / len(ppl_changes_8bit)
            if avg_ppl < 2:
                lines.append(f"  Quality impact was negligible (average {avg_ppl:+.1f}% perplexity change).")
            else:
                lines.append(f"  Quality impact: {avg_ppl:+.1f}% perplexity change.")
        lines.append("")

    if speed_gains_4bit:
        avg = sum(speed_gains_4bit) / len(speed_gains_4bit)
        lines.append(
            f"**4-bit quantization** — averaged {avg:+.0f}% speed change vs reference across "
            f"{len(speed_gains_4bit)} model(s)."
        )
        if mem_savings_4bit:
            avg_mem = sum(mem_savings_4bit) / len(mem_savings_4bit)
            lines.append(f"  Memory savings averaged {avg_mem:.0f}%.")
        if ppl_changes_4bit:
            avg_ppl = sum(ppl_changes_4bit) / len(ppl_changes_4bit)
            if avg_ppl < 5:
                lines.append(f"  Quality impact was moderate (average {avg_ppl:+.1f}% perplexity change).")
            else:
                lines.append(f"  Quality impact was significant (average {avg_ppl:+.1f}% perplexity change).")
        lines.append("")

    lines.append("### Recommendations")
    lines.append("")
    lines.append(
        "- **Best quality**: Use the highest precision that fits in your memory. "
        "bf16 is the gold standard but requires 2x the memory of 8-bit."
    )
    lines.append(
        "- **Best balance**: 8-bit quantization typically preserves nearly all quality "
        "while halving memory. This is the sweet spot for most use cases."
    )
    lines.append(
        "- **Maximum speed/minimum memory**: 4-bit gets you the largest models on "
        "constrained hardware, but expect measurable quality degradation — especially "
        "on reasoning, code generation, and tasks requiring precise knowledge recall."
    )
    lines.append(
        "- **Larger models at lower precision often beat smaller models at full precision**: "
        "e.g., a 7B-4bit model may outperform a 3B-bf16 model on quality benchmarks "
        "despite using similar memory."
    )

    return "\n".join(lines)
