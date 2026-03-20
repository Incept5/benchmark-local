# benchmark-local

Scientific benchmarking tool for measuring local LLM inference performance on Apple Silicon. Measures both **performance** (TTFT, tokens/sec, tokens/watt) and **quality** (perplexity, accuracy, quantization loss) across models and quantization levels using MLX.

## Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4)
- **macOS** 13+
- **Python** 3.11+

## Installation

```bash
git clone https://github.com/your-org/benchmark-local.git
cd benchmark-local
uv sync
```

## Quick Start

```bash
# TUI mode — interactive model selection, live progress, results viewer
uv run bench

# CLI mode — logs to stdout, good for scripting or SSH sessions
uv run bench --no-tui

# Custom config
uv run bench --config configs/my_config.toml

# Verbose logging
uv run bench --no-tui -v
```

## Usage

### TUI Mode (default)

```bash
uv run bench
```

Opens a terminal UI with three screens:

1. **Config** — toggle model families on/off, adjust warmup runs, measured runs, max tokens, and temperature
2. **Run** — live progress bar, streaming per-run metrics (TTFT, tok/s, memory), and a scrollable log
3. **Results** — sortable tables (summary, by-family quantization comparison, per-prompt breakdown) with JSON export

### CLI Mode

```bash
uv run bench --no-tui
```

Prints progress and a summary table to stdout. Useful for headless runs, piping output, or CI.

### Options

| Flag | Description |
|------|-------------|
| `--config PATH` | Config TOML to use (default: `configs/default.toml`) |
| `--no-tui` | Run in CLI mode without the terminal UI |
| `-v, --verbose` | Enable debug logging |

## Configuration

Configs are TOML files in `configs/`. The default ships with a spread of popular models from 0.5B to 70B.

### Structure

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

### Using HuggingFace Repos

Set `repo` to any `mlx-community/` HuggingFace repo ID. The model is downloaded automatically on first run and cached by `huggingface_hub`.

```toml
{ repo = "mlx-community/Mistral-7B-Instruct-v0.3-4bit", quant = "4bit" }
```

### Using Local Models

Set `repo` to an absolute path containing the MLX model files (`config.json`, `*.safetensors`, `tokenizer.json`, etc.). No download occurs.

```toml
{ repo = "/Volumes/MODELS/AI/MLX/Qwen3.5-9B-MLX-8bit", quant = "8bit" }
```

See `configs/local_example.toml` for a complete example using local paths.

### Creating Your Own Config

```bash
cp configs/default.toml configs/my_bench.toml
# edit to taste
uv run bench --config configs/my_bench.toml
```

### Tips

- **Per-prompt max_tokens**: Each prompt in the suite can set its own `max_tokens` to reflect realistic output lengths (64 for short QA, 4096 for essays). The config-level `max_tokens` is the fallback default.
- **Families with one variant** still get perplexity and MMLU but skip output similarity comparison.
- **Vision models** (`kind = "vision"`) run all prompts including image-based ones; text models skip image prompts automatically.
- **Large models** that exceed available memory are caught gracefully — the runner logs the OOM and continues to the next variant.
- The **`reference`** variant is the quality baseline. Results show metrics like "4-bit is 2.3x faster, uses 0.4x memory, with 3% perplexity increase vs 8-bit."

## What It Measures

### Performance

| Metric | Description | Method |
|--------|-------------|--------|
| **TTFT** | Time to first token (ms) | `time.perf_counter()` from prompt submit to first yielded token |
| **Tokens/sec** | Generation throughput | `tokens_generated / generation_time` |
| **Tokens/watt** | Energy efficiency | `tokens_per_sec / avg_combined_watts` via [zeus-ml](https://github.com/ml-energy/zeus) |
| **Peak memory** | GPU memory high-water mark | `mx.metal.get_peak_memory()` |

### Quality

| Metric | Description | Method |
|--------|-------------|--------|
| **Perplexity** | Intrinsic language model quality | Cross-entropy on bundled WikiText-2 sample (~5k tokens) |
| **MMLU accuracy** | Knowledge question accuracy | 100-question multiple-choice subset across 4 categories |
| **Output similarity** | Quantization drift | Token-level F1 between quantized and reference variant output |

## Methodology

- 3 warmup runs discarded per variant/prompt pair
- 10 measured runs by default
- Temperature 0.0 for deterministic output
- Variant order randomized within families to reduce thermal bias
- 95% CI and CV% reported; CV% > 10% flagged as unreliable
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
7. **Run the same config twice** — compare results across runs. If metrics differ by more than 5%, something was interfering. The CV% column in the report flags measurements with >10% variance.
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
| Qwen2.5 VL 7B Instruct | 7B | 8bit, 4bit (vision) |

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

Power metrics (tokens/watt) use [zeus-ml](https://github.com/ml-energy/zeus) which reads Apple Silicon power counters via IOKit. No sudo required. If zeus cannot access power counters on your machine, the benchmark runs normally — power metrics are simply reported as unavailable.

## Results

Each run produces a **JSON file** and a **self-contained HTML report** in `results/`.

The HTML report includes:
- System info dashboard (chip, memory, versions, benchmark parameters)
- Sortable summary table (click column headers)
- Per-family quantization comparison with delta percentages and bar charts
- Per-prompt breakdown with 95% CI and CV% flags
- Power breakdown table
- Collapsible raw JSON

The JSON file contains:
- **Aggregated stats** — median, mean, std, 95% CI, CV% per metric
- **Power** — watts and joules per variant

## Project Structure

```
benchmark-local/
  configs/
    default.toml           # Default config (HuggingFace repos)
    local_example.toml     # Example using local model paths
  prompts/
    suite.toml             # 7 prompts (5 text + 2 vision)
    images/                # Sample images for vision prompts
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
    prompts.py             # Prompt suite loader
    tui/                   # Textual TUI
  results/                 # Output (git-ignored)
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
