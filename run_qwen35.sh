#!/usr/bin/env bash
#
# Run all Qwen3.5 benchmarks in batches (largest first).
# Each batch saves its own results — safe to Ctrl+C between batches.
#
# Usage:
#   ./run_qwen35.sh          # uses bench command (pip install -e .)
#   ./run_qwen35.sh uv       # uses uv run bench
#

set -e

BENCH="bench"
if [ "$1" = "uv" ]; then
    BENCH="uv run bench"
fi

echo "========================================"
echo "  Qwen3.5 Full Benchmark Suite"
echo "  $(date)"
echo "  Runner: $BENCH"
echo "========================================"
echo ""
echo "  Batch 1: 122B-A10B, 35B-A3B, 27B, 27B-Claude, 27B-Claude-Reasoning-v2  (12 variants)"
echo "  Batch 2: 9B, 9B-Claude-Reasoning, 9B-HighIQ, 9B-Abliterated            (7 variants)"
echo "  Batch 3: 4B, 4B-Claude-Reasoning-v2                                     (6 variants, batch=[1,16])"
echo "  Batch 4: 2B, 0.8B                                                       (6 variants)"
echo ""

# --- Batch 1: Large models ---
echo "▸ Batch 1/4: Large models (122B-A10B, 35B-A3B, 27B + distilled)"
echo "  Config: configs/qwen35-large.toml"
echo "  Batch sizes: [1]"
echo "  Started: $(date '+%H:%M:%S')"
echo ""
$BENCH --config configs/qwen35-large.toml --no-tui -v
echo ""
echo "  ✓ Batch 1 complete at $(date '+%H:%M:%S')"
echo ""

# --- Batch 2: 9B variants ---
echo "▸ Batch 2/4: 9B models (base + Claude-Reasoning + HighIQ + Abliterated)"
echo "  Config: configs/qwen35-medium.toml"
echo "  Batch sizes: [1]"
echo "  Started: $(date '+%H:%M:%S')"
echo ""
$BENCH --config configs/qwen35-medium.toml --no-tui -v
echo ""
echo "  ✓ Batch 2 complete at $(date '+%H:%M:%S')"
echo ""

# --- Batch 3: 4B variants (with batch=16 test) ---
echo "▸ Batch 3/4: 4B models (base + Claude-Reasoning-v2, with batch throughput)"
echo "  Config: configs/qwen35-4b.toml"
echo "  Batch sizes: [1, 16]"
echo "  Started: $(date '+%H:%M:%S')"
echo ""
$BENCH --config configs/qwen35-4b.toml --no-tui -v
echo ""
echo "  ✓ Batch 3 complete at $(date '+%H:%M:%S')"
echo ""

# --- Batch 4: Small models ---
echo "▸ Batch 4/4: Small models (2B, 0.8B)"
echo "  Config: configs/qwen35-tiny.toml"
echo "  Batch sizes: [1]"
echo "  Started: $(date '+%H:%M:%S')"
echo ""
$BENCH --config configs/qwen35-tiny.toml --no-tui -v
echo ""
echo "  ✓ Batch 4 complete at $(date '+%H:%M:%S')"
echo ""

echo "========================================"
echo "  All batches complete!"
echo "  Finished: $(date)"
echo "  Results in: results/"
echo "========================================"
echo ""
echo "Reports generated:"
ls -t results/bench_*.md 2>/dev/null | head -4
