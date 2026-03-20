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
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
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

    if args.no_tui:
        _run_cli(config)
    else:
        _run_tui(config)


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
                    f"tok/s={r.tokens_per_sec:.1f}"
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
                f"tok/s={r.tokens_per_sec:.1f} "
                f"mem={r.peak_memory_bytes / 1024**2:.0f}MB"
            )
        elif event.stage == "quality":
            print(f"  {event.message}")
        elif event.stage == "done":
            print(f"\n{event.message}")

    print(f"benchmark-local — {len(config.model_families)} model families")
    print(f"  warmup={config.warmup_runs} measured={config.measured_runs} "
          f"max_tokens={config.max_tokens} temp={config.temperature}")
    print()

    session = run_benchmark(config, on_progress=on_progress)

    # Print summary
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
        f"{'Model':<45} {'TTFT(ms)':>10} {'tok/s':>8} {'tok/W':>8} "
        f"{'Mem(MB)':>9} {'PPL':>8} {'MMLU%':>7}"
    )
    print("-" * 100)

    for key, metrics in agg.items():
        repo, quant = key.split("|", 1)
        short_name = repo.split("/")[-1][:35]
        label = f"{short_name} ({quant})"

        ttft = metrics.get("ttft_ms", {})
        tps = metrics.get("tokens_per_sec", {})
        tpw = metrics.get("tokens_per_watt", {})
        mem = metrics.get("peak_memory_bytes", {})

        ttft_val = f"{ttft.get('median', 0):.1f}" if ttft else "—"
        tps_val = f"{tps.get('median', 0):.1f}" if tps else "—"
        tpw_val = f"{tpw.get('median', 0):.2f}" if tpw else "—"
        mem_val = f"{mem.get('median', 0) / 1024**2:.0f}" if mem else "—"

        q = quality.get(key, {})
        ppl_val = f"{q['perplexity']:.2f}" if q.get("perplexity") else "—"
        mmlu_val = f"{q['mmlu_accuracy'] * 100:.1f}" if q.get("mmlu_accuracy") is not None else "—"

        print(f"{label:<45} {ttft_val:>10} {tps_val:>8} {tpw_val:>8} {mem_val:>9} {ppl_val:>8} {mmlu_val:>7}")

    print()
