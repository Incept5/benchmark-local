# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
uv sync                                          # Install dependencies
uv run bench                                     # TUI mode (default)
uv run bench --no-tui                            # CLI mode (stdout)
uv run bench --config configs/qwen35.toml --no-tui -v  # Custom config, verbose
uv run bench --quick --no-tui                    # Quick mode (1 warmup, 1 run, 2 prompts)
uv run bench --discover "Qwen3.5:9B"             # Auto-discover LM Studio models

# Without uv:
pip install -e .
bench --no-tui
bench --discover "Qwen3.5:4B"
python -m bench.mmlu_compare /path/to/model      # MMLU thinking comparison
```

Entry point: `bench = "bench.cli:main"` in pyproject.toml → `src/bench/cli.py:main()`.

## CLI Options

| Flag | Description |
|------|-------------|
| `--config PATH` | Config TOML to use (default: `configs/default.toml`) |
| `--no-tui` | Run in CLI mode without the terminal UI |
| `-v, --verbose` | `-v` for INFO, `-vv` for DEBUG |
| `--quick` | 1 warmup, 1 run, 2 prompts, 8192 fixed context, tiny MMLU-Pro |
| `--mmlu-size {tiny,small,medium,full}` | MMLU-Pro subset (default: small; quick overrides to tiny) |
| `--discover [FILTER]` | Scan ~/.lmstudio/models/, generate TOML config. Filter: `"Qwen3.5:9B"`, `"Qwen3.5"`, or `"all"` |

## Architecture

### Data Flow

```
cli.py → runner.run_benchmark(config, on_progress)
           ├── Quick mode? Filter prompts to QUICK_PROMPT_IDS, set 8192 context
           ├── For each ModelFamily:
           │     ├── Randomize variant order (thermal bias reduction)
           │     ├── Pad prompts to context_tokens (once per family, first variant's tokenizer)
           │     └── For each ModelVariant:
           │           ├── models.load_model() → (model, tokenizer)
           │           ├── power.begin_window()
           │           ├── For each Prompt:
           │           │     ├── Warmup runs (config.effective_warmup_runs())
           │           │     └── Measured runs (config.effective_measured_runs(family))
           │           │           → RunResult (TTFT, prefill t/s, decode t/s, memory)
           │           ├── power.end_window() → PowerReading
           │           ├── batch.measure_batch() for each batch_size (if configured)
           │           ├── quality.compute_perplexity() — sliding window on WikiText-2
           │           ├── quality.eval_mmlu() — logprob scoring on MMLU-Pro
           │           ├── _save_incremental() → JSON + HTML + results-tmp.md
           │           └── gc.collect() + unload model
           ├── Compute output_similarity (quantized vs reference)
           ├── stats.aggregate() all metrics
           ├── store.save_session() → bench-{timestamp}-{config_name}.json
           ├── report.generate_report() → HTML
           └── report_md.generate_markdown_report() → MD (with derivative comparison)
```

### Key Abstractions

- **ModelFamily**: Groups variants of the same base model at different quantization levels (bf16/8bit/4bit). The `reference` variant is the quality baseline. Has `context_tokens` (auto from size) and `quality_temperature` (auto from model name).
- **Prompt**: Has optional `max_tokens` override and `image` field. Text models skip vision prompts; vision models run all. Prompts are padded to `context_tokens` with filler text.
- **ProgressEvent**: Callback pattern (`on_progress`) drives both TUI and CLI output. Stages: loading → warmup → measuring → quality → done.
- **AggregatedMetric**: Median, mean, std, 95% CI (hardcoded t-table, no scipy), CV%. Flagged unreliable if CV% > 10%.
- **BatchResult**: Result of `mlx_lm.batch_generate` at a given batch size — tracks aggregate gen/prefill tok/s.

### Text vs Vision Model Handling

Text models use `mlx_lm.stream_generate()` which yields one token per step. Vision models use `mlx_vlm.generate()` which returns the full output as one chunk. For vision, token count is estimated by re-tokenizing the output post-hoc (`measure.py` lines 63-71). Batch benchmarking is text-only (mlx_vlm has no batch API).

### Qwen3.5 Handling

Qwen3.5 uses a hybrid DeltaNet + Attention architecture (not a standard transformer). Special handling:
- **Thinking mode**: Disabled for all prompts except `reasoning` (which keeps it enabled). Some fine-tuned models hardcode `<think>` in their chat template even when `enable_thinking=False` — this is stripped in both `models.py` and `quality.py`.
- **Quality temperature**: Auto-set to 0.7 (Qwen's recommendation) instead of 0.0
- **Tool calling**: Uses XML format (`<tool_call><function=...>`) — parsed in tool_calling.py
- **Architecture notes**: Both HTML and MD reports flag hybrid architecture

### Context Size Auto-Padding

`config.py:_default_context_tokens(size)` sets context by model size:
- < 2.5B → 4096 tokens
- 2.5B–5B → 8192 tokens
- ≥ 5B → 16384 tokens

In `--quick` mode, all families are overridden to `QUICK_CONTEXT_TOKENS = 8192` for fair cross-size comparison.

Prompts padded in `prompts.py:pad_prompt_to_context()` using a filler passage, once per family (not per variant).

### Large Model Optimization

`config.effective_measured_runs(family)` reduces iterations for large models:
- Models ≥ `large_model_threshold_b` (default 8B) use `large_model_measured_runs` (default 3) instead of 10
- `--quick` mode overrides to 1 measured run and 1 warmup for all models

### MMLU-Pro (quality.py)

Replaced the original 100-question 4-choice MMLU with **MMLU-Pro** — 12,032 questions with 10 choices (A-J) across 14 categories. Uses **logprob scoring**: computes next-token logits and picks the answer with highest probability. Robust to thinking models and broken chat templates.

Subset sizes (`--mmlu-size` or `mmlu_size` in TOML):
- `tiny` (5%, ~600q) — default in `--quick` mode
- `small` (10%, ~1200q) — default in full mode
- `medium` (25%, ~3000q)
- `full` (100%, 12032q)

Dataset auto-downloads from HuggingFace `TIGER-Lab/MMLU-Pro` on first run.

### MMLU-Pro Generation Mode (mmlu_compare.py)

Standalone comparison tool: `python -m bench.mmlu_compare /path/to/model`. Runs three passes:
1. Logprob baseline (fast, no generation)
2. Generation with thinking OFF
3. Generation with thinking ON

Captures per-question TTFT, decode tok/s, tokens generated, and peak memory. Outputs comparison table with per-category accuracy deltas.

### Perplexity (quality.py)

Sliding-window perplexity on WikiText-2 full test set (~297K tokens, capped at 8K for eval). Window=2048, stride=1024. Each scored token gets up to 1024 tokens of prior context. Auto-downloads WikiText-2 on first run, falls back to bundled sample.

### Batch Throughput

`batch.py:measure_batch()` uses `mlx_lm.batch_generate` for concurrent request throughput. Configured via `batch_sizes = [1, 4, 8, 16]` in TOML. Results show aggregate gen tok/s and speedup vs single-request decode.

### Model Discovery (discover.py)

`bench --discover "Qwen3.5:9B"` scans `~/.lmstudio/models/` for MLX models:
- Detects base family, size, quantization from directory names
- Classifies standard vs derivatives (Claude Distilled, Abliterated, HighIQ, etc.)
- Groups by family+size, deduplicates (prefers mlx-community)
- Generates ready-to-run TOML config at `configs/discovered-{name}.toml`
- Strips `discovered-` prefix from output filenames

### Derivative Comparison (report_md.py)

When results contain both standard and derivative models at the same size, the markdown report auto-generates a **Derivative Comparison** section. Shows deltas for decode speed, MMLU-Pro accuracy, and perplexity at matched quantization levels.

### Output Naming

Result files include the config name: `bench-{timestamp}-{config_name}.json/html/md`. The `config_name` is the TOML filename stem with `discovered-`/`bench-`/`config-` prefixes stripped.

## Video Understanding (`video_query.py`)

Standalone script for querying video content using mlx-vlm vision-language models. Three modes:

### Modes

- **`--multi-image`** (recommended): Extracts frames, deduplicates visually similar ones via histogram correlation, sends all as separate `<image>` tokens in a single query. Fast and accurate — one prefill pass, no repetition issues.
- **`--per-frame`**: Queries each frame individually, deduplicates answers. Most thorough but slow (prefill per frame).
- **default (video)**: Sends video natively via `mlx_vlm.video_generate` pipeline. Fast but prone to repetition on list-extraction tasks.

### Key Implementation Details

- **Thinking disabled by default**: Qwen3.5 chat template supports `enable_thinking=False` which injects `<think>\n\n</think>` to skip reasoning. Without this, the model burns the entire token budget on chain-of-thought before answering. Re-enable with `--thinking`.
- **System prompt**: Included by default to constrain output (no scene descriptions, exact text reproduction). Override with `--system`.
- **Frame deduplication**: `deduplicate_frames()` compares consecutive frames using 32-bin RGB histogram correlation. Default threshold 0.95 — typically reduces frames 4-5x (e.g., 53 → 12). Control with `--dedup-threshold`.
- **Defaults**: `--temperature 0.5`, `--max-pixels 1024 1024`, `--fps 2.0`.

### Qwen3.5 as Vision Model

Qwen3.5 is a **native multimodal model** (not text-only). Config includes `video_token_id: 248057`, a vision encoder with 3D Conv patches (`temporal_patch_size: 2`), and `Qwen3VLProcessor`. It passes `is_video_model()` check in mlx-vlm. Works across all sizes (2B, 4B, 9B, 27B). Even the 2B 4-bit model produces accurate results for text extraction tasks.

### mlx-vlm Video Pipeline Constraints

- `VIDEO_MIN_PIXELS = 128 * 28 * 28 = 100,352` — frames below this are too small to read text.
- `VIDEO_MAX_PIXELS = 768 * 28 * 28 = 602,112` — per-frame pixel cap in video mode.
- Frame dimensions must be multiples of 28 (`IMAGE_FACTOR`) for the vision encoder's patch embedding.
- Frame counts rounded to multiples of 2 (`FRAME_FACTOR`).
- `smart_nframes()` computes sample count: `total_frames / video_fps * fps`, clamped to [4, 768].

## Incremental Saving

Results saved after each variant completes. Ctrl+C saves partial results. During a run, `results/results-tmp.md` is updated continuously. Final run produces timestamped `.json`, `.html`, `.md` files.

## Library Workarounds

### mlx_lm 0.31+ API

`stream_generate()` no longer accepts `temp=`. Must use `sampler=make_sampler(temp=...)` from `mlx_lm.generate`. Both `models.py` and `quality.py` use this pattern.

### zeus-ml 0.15.0 Bugs (power.py)

Two issues, both worked around:

1. **Abstract method naming**: `AppleSiliconMeasurement` has `zero_all_fields()` but `DeprecatedAliasABCMeta` registers `zeroAllFields` as abstract → class is uninstantiable. Fix: remove `zeroAllFields` from `__abstractmethods__` before use.

2. **ZeusMonitor doesn't aggregate SoC energy**: `end_window().total_energy` returns 0. Fix: bypass `ZeusMonitor` entirely, use `AppleSilicon` SoC interface directly and read millijoule fields (cpu_total_mj, gpu_mj, dram_mj, ane_mj).

## Memory Management

- `mx.reset_peak_memory()` before each measured run
- Model + tokenizer deleted after each variant, followed by `gc.collect()` + `mx.reset_peak_memory()`
- OOM on model load is caught gracefully — variant is skipped, benchmark continues

## Adding a New Metric

1. Add field to `RunResult` in `measure.py`
2. Compute it in `measure_one()`
3. Aggregate in `runner.py` → `vr.aggregated["new_metric"] = aggregate([...])`
4. Add column to `report.py` tables (use `_th("Label", "num")` for tooltip)
5. Add tooltip text to `_HELP` dict in `report.py`
6. Add column to `report_md.py` tables
