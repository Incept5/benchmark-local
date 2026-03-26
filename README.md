# MacOS-MLX-Benchmark

Scientific benchmarking tool for measuring local LLM inference performance on Apple Silicon. Measures both **performance** (TTFT, prefill/decode tok/s, tokens/watt) and **quality** (perplexity, MMLU-Pro accuracy, quantization loss) across models and quantization levels using MLX.

Features auto-discovery of LM Studio models, derivative model comparison (standard vs fine-tuned/distilled/abliterated), quick mode for fast iteration, and generation-based MMLU-Pro evaluation with thinking mode comparison.

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4)
- **macOS** 13+
- **Python** 3.11+

## Installation

### Option A: Using uv (recommended)

[uv](https://docs.astral.sh/uv/getting-started/installation/) is a fast Python package manager that handles everything automatically.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or: brew install uv

# Clone and install
git clone https://github.com/Incept5/MacOS-MLX-Benchmark.git
cd MacOS-MLX-Benchmark
uv sync
```

### Option B: Using pip

```bash
git clone https://github.com/Incept5/MacOS-MLX-Benchmark.git
cd MacOS-MLX-Benchmark
pip install -e .
```

This installs all dependencies (mlx-lm, mlx-vlm, pandas, pyarrow, textual, zeus-ml) and creates the `bench` command. Models are downloaded automatically on first run.

## Quick Start

**First time? Start here** — this runs two small models (~2GB download) and takes about 5-10 minutes:

```bash
bench --config configs/quick.toml --no-tui
```

### Auto-Discover LM Studio Models

If you have models in LM Studio, discover and benchmark them automatically:

```bash
# Discover all Qwen3.5 9B models (standard + derivatives)
bench --discover "Qwen3.5:9B"

# Run a quick benchmark on the discovered models
bench --config configs/discovered-qwen3.5-9b.toml --no-tui --quick

# Full benchmark (more runs, more prompts, larger MMLU-Pro)
bench --config configs/discovered-qwen3.5-9b.toml --no-tui
```

### Quick Mode

For fast iteration, `--quick` reduces runtime by ~95%:

```bash
bench --config configs/discovered-qwen3.5-4b.toml --no-tui --quick
```

Quick mode uses: 1 warmup, 1 measured run, 2 prompts (short-qa + code-gen), 8192 fixed context, tiny MMLU-Pro (~600 questions).

## Usage

### CLI Options

| Flag | Description |
|------|-------------|
| `--config PATH` | Config TOML to use (default: `configs/default.toml`) |
| `--no-tui` | Run in CLI mode without the terminal UI |
| `-v, --verbose` | Increase logging (`-v` for INFO, `-vv` for DEBUG) |
| `--quick` | Quick mode: 1 warmup, 1 run, 2 prompts, 8K context, tiny MMLU-Pro |
| `--mmlu-size SIZE` | MMLU-Pro subset: `tiny` (5%), `small` (10%), `medium` (25%), `full` (100%) |
| `--discover [FILTER]` | Auto-discover LM Studio models and generate config |

> **Note:** If you installed with `uv`, prefix commands with `uv run` (e.g., `uv run bench --no-tui`).

### Discovery Mode

Scans `~/.lmstudio/models/` for MLX models, classifies them as standard or derivative, and generates a ready-to-run TOML config:

```bash
bench --discover "Qwen3.5:9B"    # All Qwen3.5 9B models
bench --discover "Qwen3.5:4B"    # All Qwen3.5 4B models
bench --discover "Qwen3.5"       # All Qwen3.5 sizes
bench --discover                  # Everything
```

Derivatives are automatically classified:
- **Standard**: `mlx-community/Qwen3.5-9B-MLX-*`
- **Claude Reasoning Distilled**: `Jackrong/MLX-Qwen3.5-*-Claude-*-Reasoning-*`
- **Claude HighIQ**: `TheCluster/*-HighIQ-*`
- **Abliterated**: `huihui-ai/*-abliterated-*`

### TUI Mode

```bash
bench
```

Opens a terminal UI with three screens:

1. **Config** — toggle model families on/off, adjust warmup runs, measured runs, max tokens, and temperature
2. **Run** — live progress bar, streaming per-run metrics (TTFT, tok/s, memory), and a scrollable log
3. **Results** — sortable tables (summary, by-family quantization comparison, per-prompt breakdown) with JSON export

### MMLU-Pro Thinking Comparison

Compare model accuracy with thinking mode on vs off:

```bash
python -m bench.mmlu_compare /path/to/model --size small --temp 0.7
```

Runs three passes (logprob baseline, generation without thinking, generation with thinking) and shows accuracy, timing, memory, and per-category deltas.

### Stopping Early

Press **Ctrl+C** at any time. Results for all completed models are saved automatically.

## Example Results

### Same-Size Comparison: 0.8B bf16 vs 2B 8bit vs 4B 4bit

All roughly the same memory footprint (~3-5GB), but very different quality:

| Model | Quant | Decode | Prefill | TTFT | Memory | PPL | MMLU-Pro |
|-------|-------|-------:|--------:|-----:|-------:|----:|---------:|
| Qwen3.5 0.8B | bf16 | 308.5 t/s | 21,592 t/s | 490ms | 3,125 MB | 18.81 | 17.0% |
| Qwen3.5 2B | 8bit | 213.9 t/s | 11,973 t/s | 791ms | 3,570 MB | 13.31 | 29.0% |
| Qwen3.5 4B | 4bit | 161.0 t/s | 4,630 t/s | 1,901ms | 4,664 MB | 11.05 | 43.1% |

More parameters at lower precision beats fewer parameters at full precision — the 4B 4-bit model scores 2.5x higher on MMLU-Pro than the 0.8B bf16, despite similar memory.

### Derivative Comparison: Qwen3.5 4B Standard vs Claude Reasoning Distilled

| Variant | Quant | Decode | Memory | PPL | MMLU-Pro |
|---------|-------|-------:|-------:|----:|---------:|
| Qwen3.5 4B | bf16 | 55.3 t/s | 10,147 MB | 10.63 | 43.8% |
| Qwen3.5 4B | 8bit | 101.0 t/s | 6,554 MB | 10.67 | 43.6% |
| Qwen3.5 4B | 4bit | 161.4 t/s | 4,664 MB | 11.05 | 43.1% |
| Claude Distilled 4B | bf16 | 53.6 t/s | 10,147 MB | 10.92 | 36.6% |
| Claude Distilled 4B | 8bit | 96.2 t/s | 6,554 MB | 11.01 | 36.6% |
| Claude Distilled 4B | 4bit | 159.7 t/s | 4,664 MB | 11.27 | 36.4% |

The Claude Reasoning Distilled variant trades ~7pp of MMLU-Pro accuracy for (presumably) improved reasoning capabilities not captured by multiple-choice benchmarks. Speed and memory are identical — the architecture is unchanged.

## Configuration

Configs are TOML files in `configs/`.

### Provided Configs

| Config | Description |
|--------|-------------|
| `configs/default.toml` | Full suite — 8 model families from 0.5B to 70B |
| `configs/quick.toml` | Quick test — 2 small models, ~5-10 minutes |
| `configs/qwen35-similar-size.toml` | Same-size comparison: 0.8B bf16 vs 2B 8bit vs 4B 4bit |
| `configs/qwen35-4b.toml` | Qwen3.5 4B + Claude Reasoning Distilled derivative |
| `configs/qwen35-medium.toml` | Qwen3.5 9B + all 9B derivatives |
| `configs/qwen35-large.toml` | Qwen3.5 27B + derivatives |
| `configs/qwen35-tiny.toml` | Qwen3.5 0.8B |
| `configs/qwen35-small.toml` | Qwen3.5 2B |
| `configs/local_example.toml` | Example using local model paths |

Auto-generated configs from `--discover` are saved as `configs/discovered-*.toml`.

### Config Structure

```toml
[benchmark]
warmup_runs = 2              # discarded before measurement
measured_runs = 10            # used for statistics
large_model_measured_runs = 3 # reduced runs for models >= threshold
large_model_threshold_b = 8.0 # size threshold in billions
max_tokens = 2048             # default max tokens (prompts can override)
temperature = 0.0             # 0.0 = deterministic
randomize_order = true        # shuffle variants to reduce thermal bias
prompt_suite = "prompts/suite.toml"
output_dir = "results"
batch_sizes = [1, 4, 8, 16]  # batch throughput test (empty = skip)
batch_runs = 3                # iterations per batch size
mmlu_size = "small"           # MMLU-Pro: tiny=5%, small=10%, medium=25%, full=100%

[[model_family]]
name = "Qwen3.5 9B"
kind = "text"                 # "text" or "vision"
size = "9B"                   # display/grouping + auto context sizing
variants = [
    { repo = "mlx-community/Qwen3.5-9B-MLX-bf16", quant = "bf16" },
    { repo = "mlx-community/Qwen3.5-9B-MLX-8bit", quant = "8bit" },
    { repo = "mlx-community/Qwen3.5-9B-MLX-4bit", quant = "4bit" },
]
reference = "bf16"            # quality baseline for this family
# context_tokens = 8192      # auto: <2.5B=4K, <5B=8K, >=5B=16K
# quality_temperature = 0.7  # auto: 0.7 for Qwen3.5, 0.0 for others
```

### Using Local Models

If you have MLX models on disk (e.g., from LM Studio), point `repo` at the directory:

```toml
{ repo = "/Users/you/.lmstudio/models/mlx-community/Qwen3.5-9B-MLX-8bit", quant = "8bit" }
```

Or use `--discover` to generate configs automatically from your LM Studio library.

## What It Measures

### Performance

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to first token (ms) |
| **Prefill tok/s** | Prompt processing throughput |
| **Decode tok/s** | Token generation throughput (the speed you feel during streaming) |
| **Tokens/watt** | Energy efficiency via [zeus-ml](https://github.com/ml-energy/zeus) |
| **Peak memory** | GPU memory high-water mark (MB) |

### Quality

| Metric | Description |
|--------|-------------|
| **Perplexity** | Sliding-window perplexity on WikiText-2 test set (8K tokens, window=2048, stride=1024). Lower is better. |
| **MMLU-Pro** | Logprob accuracy on MMLU-Pro — 10-choice questions across 14 categories (harder than standard MMLU). Higher is better. |
| **Output similarity** | Token-level F1 between quantized and reference variant output. 1.0 = identical. |
| **Per-category accuracy** | MMLU-Pro breakdown across STEM, law, business, health, etc. Saved in results JSON. |

### Batch Throughput

When `batch_sizes` is configured, the benchmark runs `mlx_lm.batch_generate` at each batch size to measure aggregate throughput with concurrent requests.

### Smart Defaults

| Parameter | Auto Rule | Override |
|-----------|-----------|---------|
| **Context tokens** | <2.5B→4K, <5B→8K, ≥5B→16K | `context_tokens` per family |
| **Quality temperature** | Qwen3.5→0.7, others→0.0 | `quality_temperature` per family |
| **Measured runs** | ≥8B models→3 runs, others→10 | `large_model_measured_runs` |
| **MMLU-Pro size** | Full mode→small, quick→tiny | `--mmlu-size` or `mmlu_size` in TOML |
| **Thinking mode** | Disabled except for reasoning prompts | Per-prompt in models.py |

## Results

Each run produces three output files in `results/`:

```
bench-20260326_142019-qwen3.5-4b.json   # Raw data
bench-20260326_142019-qwen3.5-4b.html   # HTML report (dark theme)
bench-20260326_142019-qwen3.5-4b.md     # Markdown report
```

The filename includes the config name for easy identification.

### HTML Report

- Dark theme with system info dashboard
- Architecture notes for non-standard models (e.g., Qwen3.5 hybrid DeltaNet)
- Sortable summary table with tooltips
- Per-family quantization comparison with bar charts
- Batch throughput, power, and tool calling tables
- Collapsible raw JSON

### Markdown Report

- Summary and per-family quantization tables
- **Derivative comparison** — when standard and derivative models are benchmarked together, shows accuracy/speed/quality deltas at matched quantization levels
- Per-family commentary and quantization analysis
- Architecture notes and metric definitions

During a run, `results/results-tmp.md` is continuously updated after each model variant completes.

## Methodology

- 2 warmup runs discarded per variant/prompt pair (1 in quick mode)
- 10 measured runs by default (3 for large models, 1 in quick mode)
- Temperature 0.0 for deterministic output
- Variant order randomized within families to reduce thermal bias
- 95% CI and CV% reported per-prompt; CV% > 10% flagged as unreliable
- Prompts padded to target context size for fair prefill comparison
- MMLU-Pro uses logprob scoring (immune to thinking models and verbose outputs)
- Perplexity uses sliding-window method with proper context (not independent chunks)

### Running a Fair Benchmark

For reliable, reproducible results:

1. **Close all other applications** — browsers, IDEs, chat clients
2. **Disable background processes** — cloud sync, Time Machine, Spotlight indexing
3. **Plug in your Mac** — battery mode throttles performance
4. **Let the machine cool down** — wait 5-10 minutes after heavy workloads
5. **Don't touch the machine during the run**
6. **Check Activity Monitor** — CPU idle should be >95%

## Prompt Suite

14 prompts covering realistic workloads with varied context and output lengths:

| Prompt | Category | Input | Max Tokens |
|--------|----------|-------|------------|
| short-qa | Short QA | 10 words | 64 |
| medium-explain | Explanation | 49 words | 1,024 |
| long-essay | Essay | 146 words | 4,096 |
| code-gen | Code generation | 121 words | 2,048 |
| reasoning | Math/logic | 152 words | 2,048 |
| summarize-long | Summarization | ~900 words | 512 |
| multi-step | Architecture design | 213 words | 3,072 |
| vision-chart | Chart analysis | image + prompt | 1,024 |
| vision-photo | Photo description | image + prompt | 1,024 |
| vision-diagram | Diagram explanation | image + prompt | 1,024 |
| vision-document | Document reading | image + prompt | 2,048 |
| vision-screenshot | Code screenshot | image + prompt | 1,024 |
| vision-handwriting | Handwriting OCR | image + prompt | 1,024 |
| vision-infographic | Infographic analysis | image + prompt | 1,024 |

In `--quick` mode, only `short-qa`, `code-gen`, and `vision-chart` are used.

## Project Structure

```
MacOS-MLX-Benchmark/
  configs/
    default.toml               # Full benchmark suite
    quick.toml                 # Quick test (~5-10 min)
    qwen35-similar-size.toml   # Same-size comparison
    qwen35-4b.toml             # 4B + derivatives
    qwen35-medium.toml         # 9B + derivatives
    qwen35-large.toml          # 27B + derivatives
    local_example.toml         # Local model paths example
  prompts/
    suite.toml                 # 14 prompts (7 text + 7 vision)
    images/                    # Test images for vision prompts
  evals/
    mmlu_pro/                  # MMLU-Pro dataset (auto-downloaded)
    video/                     # Video eval files
    wikitext2_test.txt         # WikiText-2 test set (auto-downloaded)
    wikitext_sample.txt        # Bundled WikiText sample (fallback)
    mmlu_subset.toml           # Legacy 100-question MMLU
    tool_calling.toml          # Tool calling eval spec
  src/bench/
    cli.py                     # Entry point + --discover
    config.py                  # Config loading + quick/large model logic
    runner.py                  # Benchmark orchestrator
    discover.py                # LM Studio model discovery
    models.py                  # mlx_lm / mlx_vlm wrappers + Qwen3.5 handling
    measure.py                 # Per-run timing/memory
    quality.py                 # Perplexity, MMLU-Pro, similarity
    mmlu_compare.py            # MMLU thinking on/off comparison
    batch.py                   # Batch throughput measurement
    power.py                   # zeus-ml energy measurement
    stats.py                   # Statistical aggregation
    store.py                   # JSON save/load
    report.py                  # HTML report generation
    report_md.py               # Markdown report + derivative comparison
    prompts.py                 # Prompt suite loader
    tool_calling.py            # Qwen3.5 XML tool calling parser
    tui/                       # Textual TUI
  video_query.py               # Standalone video understanding script
  results/                     # Output (git-ignored)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `bench: command not found` | Run `pip install -e .` or use `python -m bench.cli` |
| `mlx` fails to import | Must be on Apple Silicon (M1+). Intel Macs not supported. |
| Model download is slow | Use `configs/quick.toml` for smaller models, or `--discover` for local models |
| Out of memory | Use 4-bit quantizations or remove large models from config. OOMs are caught gracefully. |
| MMLU-Pro not found | Auto-downloads on first run. Requires `huggingface_hub` (installed with deps). |
| WikiText-2 not found | Auto-downloads on first run. Requires `datasets` library. Falls back to bundled sample. |

## License

Apache 2.0 — see [LICENSE](LICENSE).
