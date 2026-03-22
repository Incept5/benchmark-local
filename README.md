# MacOS-MLX-Benchmark

Scientific benchmarking tool for measuring local LLM inference performance on Apple Silicon. Measures both **performance** (TTFT, tokens/sec, tokens/watt) and **quality** (perplexity, accuracy, quantization loss) across models and quantization levels using MLX.

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4)
- **macOS** 13+
- **Python** 3.11+
- **uv** — fast Python package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))

## Installation

### 1. Install uv (if you don't have it)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via Homebrew:

```bash
brew install uv
```

### 2. Clone and install

```bash
git clone https://github.com/Incept5/MacOS-MLX-Benchmark.git
cd MacOS-MLX-Benchmark
uv sync
```

This installs all core dependencies including power measurement (mlx-lm, textual, zeus-ml). Models are downloaded automatically on first run.

### 3. Optional: Vision model support

```bash
uv sync --extra vision
```

Only needed if you want to benchmark vision models (e.g. Qwen2.5-VL). Text-only benchmarking works without this.

## Quick Start

**First time? Start here** — this runs two small models (~2GB download) and takes about 5-10 minutes:

```bash
uv run bench --config configs/quick.toml --no-tui
```

This will:
1. Download two small models (Qwen2.5 0.5B and Llama 3.2 3B) from HuggingFace
2. Run warmup + measured iterations across 7 text prompts
3. Evaluate perplexity and MMLU accuracy
4. Save a JSON file and HTML report to `results/`

Once that works, run the full benchmark:

```bash
uv run bench --no-tui
```

Or use the interactive TUI:

```bash
uv run bench
```

## Usage

### CLI Mode

```bash
uv run bench --no-tui                              # Full benchmark, console output
uv run bench --config configs/quick.toml --no-tui   # Quick benchmark
uv run bench --no-tui -v                            # Verbose logging
```

### TUI Mode

```bash
uv run bench
```

Opens a terminal UI with three screens:

1. **Config** — toggle model families on/off, adjust warmup runs, measured runs, max tokens, and temperature
2. **Run** — live progress bar, streaming per-run metrics (TTFT, tok/s, memory), and a scrollable log
3. **Results** — sortable tables (summary, by-family quantization comparison, per-prompt breakdown) with JSON export

### Options

| Flag | Description |
|------|-------------|
| `--config PATH` | Config TOML to use (default: `configs/default.toml`) |
| `--no-tui` | Run in CLI mode without the terminal UI |
| `-v, --verbose` | Enable debug logging |

## Configuration

Configs are TOML files in `configs/`. Three are provided:

| Config | Description |
|--------|-------------|
| `configs/default.toml` | Full suite — 8 model families from 0.5B to 70B (HuggingFace repos, auto-downloaded) |
| `configs/quick.toml` | Quick test — 2 small models, fewer runs, ~5-10 minutes |
| `configs/local_example.toml` | Example using models already downloaded to a local directory |

### Config Structure

```toml
[benchmark]
warmup_runs = 3          # discarded before measurement
measured_runs = 10        # used for statistics
max_tokens = 2048         # default max tokens (prompts can override)
temperature = 0.0         # 0.0 = deterministic
randomize_order = true    # shuffle variants to reduce thermal bias
prompt_suite = "prompts/suite.toml"
output_dir = "results"

[[model_family]]
name = "Llama 3.2 3B Instruct"
kind = "text"             # "text" or "vision"
size = "3B"               # display/grouping only
variants = [
    { repo = "mlx-community/Llama-3.2-3B-Instruct-bf16", quant = "bf16" },
    { repo = "mlx-community/Llama-3.2-3B-Instruct-8bit", quant = "8bit" },
    { repo = "mlx-community/Llama-3.2-3B-Instruct-4bit", quant = "4bit" },
]
reference = "bf16"        # quality baseline for this family
```

### Using HuggingFace Models (auto-download)

Set `repo` to any `mlx-community/` HuggingFace repo ID. The model downloads automatically on first run and is cached by `huggingface_hub` in `~/.cache/huggingface/`.

```toml
{ repo = "mlx-community/Mistral-7B-Instruct-v0.3-4bit", quant = "4bit" }
```

### Using Local Models (no download)

If you already have MLX models on disk, point `repo` at the directory containing the model files (`config.json`, `*.safetensors`, `tokenizer.json`, etc.):

```toml
{ repo = "/Volumes/MODELS/AI/MLX/Qwen3.5-9B-MLX-8bit", quant = "8bit" }
```

See `configs/local_example.toml` for a complete example.

### Creating Your Own Config

```bash
cp configs/quick.toml configs/my_bench.toml
# edit to add/remove models
uv run bench --config configs/my_bench.toml --no-tui
```

### Tips

- **Per-prompt max_tokens**: Each prompt in the suite can set its own `max_tokens` to reflect realistic output lengths (64 for short QA, 4096 for essays). The config-level `max_tokens` is the fallback default.
- **Families with one variant** still get perplexity and MMLU but skip output similarity comparison.
- **Vision models** (`kind = "vision"`) run all prompts including image-based ones; text models skip image prompts automatically. Requires `uv sync --extra vision`.
- **Large models** that exceed available memory are caught gracefully — the runner logs the OOM and continues to the next variant.
- The **`reference`** variant is the quality baseline. Results show metrics like "4-bit is 1.6x faster, uses 0.5x memory, with 5% perplexity increase vs 8-bit."

## What It Measures

### Performance

| Metric | Description | Method |
|--------|-------------|--------|
| **TTFT** | Time to first token (ms) | `time.perf_counter()` from prompt submit to first yielded token |
| **Tokens/sec** | Generation throughput | `tokens_generated / generation_time` |
| **Tokens/watt** | Energy efficiency | `tokens_per_sec / avg_combined_watts` via [zeus-ml](https://github.com/ml-energy/zeus) |
| **Peak memory** | GPU memory high-water mark | `mx.get_peak_memory()` |

### Quality

| Metric | Description | Method |
|--------|-------------|--------|
| **Perplexity** | Intrinsic language model quality | Cross-entropy on bundled WikiText-2 sample (~5k tokens). Lower is better. |
| **MMLU accuracy** | Knowledge question accuracy | 100-question multiple-choice subset across 4 categories (STEM, humanities, social science, other) |
| **Output similarity** | Quantization drift | Token-level F1 between quantized and reference variant output. 1.0 = identical. |

## Methodology

- 3 warmup runs discarded per variant/prompt pair
- 10 measured runs by default
- Temperature 0.0 for deterministic output
- Variant order randomized within families to reduce thermal bias
- 95% CI and CV% reported per-prompt; CV% > 10% flagged as unreliable
- Full output text stored for reproducibility verification
- System info, library versions, and config snapshot saved in every result file
- Power measured per-variant (not per-run) to reduce noise

### Running a Fair Benchmark

For reliable, reproducible results:

1. **Close all other applications** — browsers, IDEs, chat clients, music players. Any app consuming CPU/GPU/memory will affect measurements.
2. **Disable background processes** — pause cloud sync (iCloud, Dropbox), Time Machine backups, Spotlight indexing, and software updates.
3. **Plug in your Mac** — battery mode throttles performance. Always benchmark on AC power.
4. **Let the machine cool down** — if you've been running heavy workloads, wait 5-10 minutes before starting. Thermal throttling skews results.
5. **Don't touch the machine during the run** — even moving windows or typing causes CPU spikes.
6. **Check Activity Monitor** before starting — CPU idle should be >95%, memory pressure should be green.
7. **Run the same config twice** — compare results across runs. If metrics differ by more than 5%, something was interfering.
8. **Close the terminal's other tabs/panes** — terminal emulators rendering output consume measurable CPU.

The benchmark randomizes variant order within families to reduce thermal bias, but consistent ambient conditions still matter. If you're comparing results across machines, also ensure similar room temperature.

## Default Models

The default config (`configs/default.toml`) includes:

| Family | Size | Quants |
|--------|------|--------|
| Qwen2.5 0.5B Instruct | 0.5B | bf16, 8bit, 4bit |
| Llama 3.2 1B Instruct | 1B | bf16, 8bit, 4bit |
| Llama 3.2 3B Instruct | 3B | bf16, 8bit, 4bit |
| Mistral 7B Instruct v0.3 | 7B | fp16, 8bit, 4bit |
| Llama 3.1 8B Instruct | 8B | bf16, 8bit, 4bit |
| Qwen2.5 14B Instruct | 14B | bf16, 8bit, 4bit |
| Llama 3.1 70B Instruct | 70B | 4bit |
| Qwen2.5 VL 7B Instruct | 7B | 8bit, 4bit (vision, requires `uv sync --extra vision`) |

> **Note:** The full default config downloads many large models. Start with `configs/quick.toml` to verify your setup.

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

Text models run the 7 text prompts. Vision models run all 14. All vision test images are real photographs and documents sourced from Wikimedia Commons (see `prompts/images/ATTRIBUTIONS.md`).

## Power Measurement

Power metrics (tokens/watt) use [zeus-ml](https://github.com/ml-energy/zeus) which reads Apple Silicon power counters via IOKit — no sudo required. This is included in the base install.

## Results

Each run produces a **JSON file** and a **self-contained HTML report** in `results/`.

The HTML report includes:
- System info dashboard (chip, memory, versions, benchmark parameters)
- Sortable summary table (click column headers; hover for metric descriptions)
- Per-family quantization comparison with delta percentages and bar charts
- Per-prompt breakdown with 95% CI and CV% flags
- Power breakdown table
- Collapsible raw JSON

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `uv: command not found` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `mlx` fails to import | Must be on Apple Silicon (M1+). Intel Macs are not supported. |
| Model download is slow | Models download from HuggingFace. Use `configs/quick.toml` for smaller models, or pre-download with `huggingface-cli download mlx-community/MODEL-NAME`. |
| `zeus` install fails | Try `uv sync` again. If it persists, power metrics will be unavailable but the benchmark still runs. |
| Out of memory | Remove large models from your config, or use 4-bit quantizations. The runner catches OOMs and continues. |
| Vision models fail | Install vision support: `uv sync --extra vision` |

## Project Structure

```
MacOS-MLX-Benchmark/
  configs/
    default.toml           # Full benchmark (HuggingFace repos, auto-download)
    quick.toml             # Quick test (2 small models, ~5-10 min)
    local_example.toml     # Example using local model paths
  prompts/
    suite.toml             # 14 prompts (7 text + 7 vision)
    images/                # Test images for vision prompts
  evals/
    mmlu_subset.toml       # 100 MMLU questions
    wikitext_sample.txt    # WikiText-2 excerpt for perplexity
  src/bench/
    cli.py                 # Entry point
    config.py              # Config loading
    runner.py              # Benchmark orchestrator
    measure.py             # Per-run timing/memory
    quality.py             # Perplexity, MMLU, similarity
    power.py               # zeus-ml energy measurement
    models.py              # mlx_lm / mlx_vlm wrappers
    stats.py               # Statistical aggregation
    store.py               # JSON save/load
    report.py              # HTML report generation
    prompts.py             # Prompt suite loader
    tui/                   # Textual TUI
  results/                 # Output (git-ignored)
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
