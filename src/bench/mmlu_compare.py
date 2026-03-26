"""Compare MMLU-Pro scores with thinking on vs off.

Usage:
    python -m bench.mmlu_compare <model_path> [--size tiny|small|medium|full] [--temp 0.7]

Example:
    python -m bench.mmlu_compare /Users/jdavies/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit
    python -m bench.mmlu_compare /Users/jdavies/.lmstudio/models/mlx-community/Qwen3.5-4B-MLX-4bit --size small
"""

from __future__ import annotations

import argparse
import statistics
import time

from bench.config import MMLU_PRO_SIZES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MMLU-Pro with thinking on vs off",
    )
    parser.add_argument("model", help="Path to MLX model directory")
    parser.add_argument("--size", choices=list(MMLU_PRO_SIZES.keys()), default="small",
                        help="MMLU-Pro subset size (default: small)")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Temperature for generation (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens per question (default: 2048)")
    args = parser.parse_args()

    fraction = MMLU_PRO_SIZES[args.size]

    print(f"Loading {args.model.split('/')[-1]}...")
    from mlx_lm import load
    model, tokenizer = load(args.model)

    from bench.quality import eval_mmlu, eval_mmlu_generate

    # --- Logprob baseline ---
    print(f"\n{'='*70}")
    print(f"LOGPROB SCORING (baseline, no generation)")
    print(f"{'='*70}")
    start = time.perf_counter()
    lp_acc, lp_correct, lp_total, lp_cats = eval_mmlu(
        model, tokenizer, fraction=fraction, temperature=args.temp,
    )
    lp_time = time.perf_counter() - start
    print(f"Accuracy: {lp_acc*100:.1f}% ({lp_correct}/{lp_total}) in {lp_time:.1f}s")
    _print_categories(lp_cats)

    # --- Generation: thinking OFF ---
    print(f"\n{'='*70}")
    print(f"GENERATION — thinking OFF (temp={args.temp})")
    print(f"{'='*70}")
    start = time.perf_counter()
    off_acc, off_correct, off_total, off_cats, off_results = eval_mmlu_generate(
        model, tokenizer, fraction=fraction, temperature=args.temp,
        enable_thinking=False, max_tokens=args.max_tokens,
        on_progress=_progress,
    )
    off_time = time.perf_counter() - start
    print()
    print(f"Accuracy: {off_acc*100:.1f}% ({off_correct}/{off_total}) in {off_time:.1f}s")
    _print_metrics(off_results)
    _print_categories(off_cats)

    # --- Generation: thinking ON ---
    print(f"\n{'='*70}")
    print(f"GENERATION — thinking ON (temp={args.temp})")
    print(f"{'='*70}")
    start = time.perf_counter()
    on_acc, on_correct, on_total, on_cats, on_results = eval_mmlu_generate(
        model, tokenizer, fraction=fraction, temperature=args.temp,
        enable_thinking=True, max_tokens=args.max_tokens,
        on_progress=_progress,
    )
    on_time = time.perf_counter() - start
    print()
    print(f"Accuracy: {on_acc*100:.1f}% ({on_correct}/{on_total}) in {on_time:.1f}s")
    _print_metrics(on_results)
    _print_categories(on_cats)

    # --- Comparison ---
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Accuracy':>10} {'Time':>10} {'Avg tok/s':>10} {'Avg TTFT':>10} {'Avg tokens':>10}")
    print("-" * 80)
    print(f"{'Logprob':<25} {lp_acc*100:>9.1f}% {lp_time:>9.1f}s {'—':>10} {'—':>10} {'—':>10}")

    off_tps = statistics.median([r.decode_tps for r in off_results if r.decode_tps > 0]) if off_results else 0
    off_ttft = statistics.median([r.ttft_ms for r in off_results]) if off_results else 0
    off_toks = statistics.median([r.tokens_generated for r in off_results]) if off_results else 0
    print(f"{'Generate (no thinking)':<25} {off_acc*100:>9.1f}% {off_time:>9.1f}s {off_tps:>9.1f} {off_ttft:>8.0f}ms {off_toks:>10.0f}")

    on_tps = statistics.median([r.decode_tps for r in on_results if r.decode_tps > 0]) if on_results else 0
    on_ttft = statistics.median([r.ttft_ms for r in on_results]) if on_results else 0
    on_toks = statistics.median([r.tokens_generated for r in on_results]) if on_results else 0
    print(f"{'Generate (thinking)':<25} {on_acc*100:>9.1f}% {on_time:>9.1f}s {on_tps:>9.1f} {on_ttft:>8.0f}ms {on_toks:>10.0f}")

    delta = on_acc - off_acc
    print(f"\nThinking {'improves' if delta > 0 else 'hurts'} accuracy by {abs(delta)*100:.1f}pp")
    print(f"Thinking generates {on_toks/off_toks:.1f}x more tokens per question" if off_toks > 0 else "")

    # Memory
    off_mem = max(r.peak_memory_mb for r in off_results) if off_results else 0
    on_mem = max(r.peak_memory_mb for r in on_results) if on_results else 0
    print(f"Peak memory: {off_mem:.0f} MB (no thinking) vs {on_mem:.0f} MB (thinking)")

    # Per-category delta
    print(f"\n{'Category':<20} {'No Think':>10} {'Think':>10} {'Delta':>10}")
    print("-" * 55)
    all_cats = sorted(set(list(off_cats.keys()) + list(on_cats.keys())))
    for cat in all_cats:
        off_v = off_cats.get(cat, 0)
        on_v = on_cats.get(cat, 0)
        d = on_v - off_v
        print(f"{cat:<20} {off_v*100:>9.1f}% {on_v*100:>9.1f}% {d*100:>+9.1f}pp")


def _progress(current: int, total: int) -> None:
    if current % 50 == 0 or current == total:
        print(f"  {current}/{total}", end="\r", flush=True)


def _print_metrics(results: list) -> None:
    if not results:
        return
    ttfts = [r.ttft_ms for r in results]
    tps_list = [r.decode_tps for r in results if r.decode_tps > 0]
    toks = [r.tokens_generated for r in results]
    mem = max(r.peak_memory_mb for r in results)
    print(f"  TTFT:    median={statistics.median(ttfts):.0f}ms")
    print(f"  Decode:  median={statistics.median(tps_list):.1f} t/s" if tps_list else "  Decode: —")
    print(f"  Tokens:  median={statistics.median(toks):.0f} (total={sum(toks)})")
    print(f"  Memory:  {mem:.0f} MB peak")


def _print_categories(cats: dict[str, float]) -> None:
    print(f"  Per-category:")
    for cat in sorted(cats, key=lambda c: cats[c]):
        print(f"    {cat:<20s} {cats[cat]*100:.1f}%")


if __name__ == "__main__":
    main()
