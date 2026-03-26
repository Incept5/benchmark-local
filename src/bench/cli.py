"""Entry point: `uv run bench`."""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="MLX inference benchmarking tool for Apple Silicon",
    )
    parser.add_argument(
        "--config",
        default="configs/default.toml",
        help="Path to benchmark config TOML (default: configs/default.toml)",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Run in CLI mode without the TUI",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (-v for INFO, -vv for DEBUG)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 1 warmup, 1 measured run, 2 representative prompts",
    )
    parser.add_argument(
        "--mmlu-size",
        choices=["tiny", "small", "medium", "full"],
        default=None,
        help="MMLU-Pro subset: tiny=5%% (~600q), small=10%%, medium=25%%, full=100%% (default: from config or tiny)",
    )
    parser.add_argument(
        "--discover",
        metavar="FILTER",
        nargs="?",
        const="all",
        help='Discover LM Studio MLX models and generate a config. '
             'Filter by "family:size" e.g. "Qwen3.5:9B", "Qwen3.5", or "all"',
    )
    args = parser.parse_args()

    # Handle --discover before anything else
    if args.discover is not None:
        _run_discover(args.discover)
        return

    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose >= 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from bench.config import BenchmarkConfig

    try:
        config = BenchmarkConfig.from_toml(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    if args.quick:
        config.quick = True
        from bench.config import QUICK_CONTEXT_TOKENS
        for family in config.model_families:
            family.context_tokens = QUICK_CONTEXT_TOKENS
        # Default to tiny MMLU-Pro in quick mode unless explicitly overridden
        if not args.mmlu_size:
            config.mmlu_size = "tiny"
    if args.mmlu_size:
        config.mmlu_size = args.mmlu_size

    if args.no_tui:
        _run_cli(config)
    else:
        _run_tui(config)


def _run_discover(filter_str: str) -> None:
    """Discover models and generate/print config."""
    from bench.discover import discover_models, group_models, generate_toml, print_discovery

    # Parse filter
    base_family = None
    size = None
    if filter_str and filter_str != "all":
        parts = filter_str.split(":")
        base_family = parts[0]
        if len(parts) > 1:
            size = parts[1]

    models = discover_models(base_family=base_family, size=size)
    if not models:
        print(f"No MLX models found matching filter '{filter_str}'")
        print("Searched: ~/.lmstudio/models/")
        return

    # Print summary
    print_discovery(models)

    # Generate and save config
    groups = group_models(models)
    toml_content = generate_toml(groups)

    if size:
        filename = f"configs/discovered-{base_family.lower()}-{size.lower()}.toml"
    elif base_family:
        filename = f"configs/discovered-{base_family.lower()}.toml"
    else:
        filename = "configs/discovered-all.toml"

    from pathlib import Path
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml_content)
    print(f"\nConfig written to: {filename}")
    print(f"Run with: bench --config {filename} --no-tui --quick")


def _run_tui(config: BenchmarkConfig) -> None:
    """Run benchmark with the Textual TUI."""
    from bench.tui.app import BenchApp

    app = BenchApp(config)
    app.run()


def _run_cli(config: BenchmarkConfig) -> None:
    """Run benchmark in CLI mode with progress printed to stdout."""
    from bench.runner import ProgressEvent, run_benchmark

    def on_progress(event: ProgressEvent) -> None:
        if event.error:
            print(f"  ERROR: {event.error}")
            return
        if event.stage == "loading":
            print(f"\n>>> Loading {event.variant_repo} ({event.variant_quant})")
        elif event.stage == "warmup":
            if event.current_result:
                r = event.current_result
                print(
                    f"  Warmup {event.run_index}/{event.total_runs} "
                    f"[{event.prompt_id}] "
                    f"TTFT={r.ttft_ms:.1f}ms "
                    f"prefill={r.prefill_tps:.1f}t/s "
                    f"decode={r.decode_tps:.1f}t/s"
                )
            else:
                print(
                    f"  Warmup {event.run_index}/{event.total_runs} "
                    f"[{event.prompt_id}]"
                )
        elif event.stage == "measuring" and event.current_result:
            r = event.current_result
            print(
                f"  Run {event.run_index}/{event.total_runs} "
                f"[{event.prompt_id}] "
                f"TTFT={r.ttft_ms:.1f}ms "
                f"prefill={r.prefill_tps:.1f}t/s "
                f"decode={r.decode_tps:.1f}t/s "
                f"mem={r.peak_memory_bytes / 1024**2:.0f}MB"
            )
        elif event.stage == "quality":
            print(f"  {event.message}")
        elif event.stage == "done":
            print(f"\n{event.message}")

    mode_label = " [QUICK]" if config.quick else ""
    print(f"MacOS-MLX-Benchmark{mode_label} — {len(config.model_families)} model families")
    if config.quick:
        from bench.config import QUICK_PROMPT_IDS, QUICK_CONTEXT_TOKENS
        print(f"  warmup=1 measured=1 context={QUICK_CONTEXT_TOKENS} "
              f"prompts={','.join(QUICK_PROMPT_IDS)} "
              f"max_tokens={config.max_tokens} temp={config.temperature}")
    else:
        print(f"  warmup={config.warmup_runs} measured={config.measured_runs} "
              f"(large models ≥{config.large_model_threshold_b:.0f}B: {config.large_model_measured_runs}) "
              f"max_tokens={config.max_tokens} temp={config.temperature}")
    print()

    session = run_benchmark(config, on_progress=on_progress)

    # Print duration and summary
    if session.duration_s > 0:
        mins, secs = divmod(int(session.duration_s), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            print(f"\nTotal benchmark duration: {hours}h {mins}m {secs}s")
        elif mins > 0:
            print(f"\nTotal benchmark duration: {mins}m {secs}s")
        else:
            print(f"\nTotal benchmark duration: {secs}s")
    _print_summary(session)


def _print_summary(session) -> None:
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    agg = session.aggregated
    quality = session.quality

    if not agg:
        print("No results collected.")
        return

    # Header
    print(
        f"{'Model':<40} {'TTFT(ms)':>9} {'Prefill':>9} {'Decode':>9} {'tok/W':>8} "
        f"{'Mem(MB)':>9} {'PPL':>8} {'MMLU-Pro%':>9}"
    )
    print("-" * 110)

    for key, metrics in agg.items():
        repo, quant = key.split("|", 1)
        short_name = repo.split("/")[-1][:30]
        label = f"{short_name} ({quant})"

        ttft = metrics.get("ttft_ms", {})
        prefill = metrics.get("prefill_tps", {})
        decode = metrics.get("decode_tps", {})
        tpw = metrics.get("tokens_per_watt", {})
        mem = metrics.get("peak_memory_bytes", {})

        ttft_val = f"{ttft.get('median', 0):.1f}" if ttft else "—"
        prefill_val = f"{prefill.get('median', 0):.1f}" if prefill else "—"
        decode_val = f"{decode.get('median', 0):.1f}" if decode else "—"
        tpw_val = f"{tpw.get('median', 0):.2f}" if tpw else "—"
        mem_val = f"{mem.get('median', 0) / 1024**2:.0f}" if mem else "—"

        q = quality.get(key, {})
        ppl_val = f"{q['perplexity']:.2f}" if q.get("perplexity") else "—"
        mmlu_val = f"{q['mmlu_accuracy'] * 100:.1f}" if q.get("mmlu_accuracy") is not None else "—"

        print(f"{label:<40} {ttft_val:>9} {prefill_val:>9} {decode_val:>9} {tpw_val:>8} {mem_val:>9} {ppl_val:>8} {mmlu_val:>7}")

    print()
