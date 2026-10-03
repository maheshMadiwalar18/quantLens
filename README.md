# LLM Inference Optimization Lab

![CI](https://github.com/mahes/quantLens/actions/workflows/ci.yml/badge.svg)

A professional, fully reproducible benchmarking system that evaluates how different LLM inference configurations affect generation throughput, latency, VRAM usage, GPU utilization, and model quality.

## Core Concepts

- **What is LLM Inference?** The process of passing inputs (prompts) through a trained Large Language Model to generate outputs (tokens) autoregressively.
- **Why Quantization Matters:** Storing parameters in FP16 takes massive amounts of VRAM. Quantization (like INT8 or 4-bit via `bitsandbytes`) compresses the model weights. While it drastically reduces the VRAM footprint (allowing larger models or batches), it may introduce dequantization overhead that can *increase* latency, and it can marginally degrade model intelligence.
- **What Batching Does:** Instead of answering one user request at a time, batching processes multiple requests simultaneously. This utilizes the massive parallel architecture of GPUs, radically increasing aggregate throughput (Tokens/sec) at the cost of a slight increase in individual user latency and VRAM.
- **What KV Cache Does:** Autoregressive models must attend to all previous tokens to generate the next one. The KV cache stores the "Key" and "Value" tensors of previous tokens in memory so they don't have to be redundantly recomputed. This makes generation much faster but consumes huge amounts of VRAM as context length grows.

## Features
- **Deterministic Measurement:** Uses precise `torch.cuda.synchronize()` wrapping to record true GPU latency, discarding naive asynchronous `time.time()` measurements.
- **Graceful Fault Tolerance:** Safely catches CUDA Out-Of-Memory (OOM) exceptions without crashing the matrix runner.
- **Quality Evaluation:** Calculates true Cross-Entropy Perplexity on the Wikitext-2 dataset with strict `-100` context masking to prove if quantization degrades predictive intelligence.

## Installation

1. Create a Python virtual environment:
```bash
python -m venv venv
```
2. Activate the virtual environment:
- Windows: `.\venv\Scripts\Activate.ps1`
- Linux/macOS: `source venv/bin/activate`
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Reproducing Experiments

Define your benchmarking matrix in `configs/experiments.yaml`. Then, kick off the autonomous suite:

```bash
python -m benchmark.run_experiment --config configs/experiments.yaml
```
Results will be dumped into `results/matrix_dataset_<timestamp>.csv` and raw JSONs will be saved in `results/raw/`.

## Key Findings
*(Note: These will populate with real data based on the specific hardware executing the matrix. Theoretical expectations below are derived from standard PyTorch behavior.)*

- **Quantization:** Dropping to 4-bit massively reduces peak VRAM but does not automatically yield higher tokens/sec due to computational dequantization overhead.
- **Batching:** Moving from Batch Size 1 to 8 generally yields over a 3x increase in aggregate throughput, heavily amortizing memory bandwidth bottlenecks.
- **KV Cache:** Disabling the cache causes latency to scale quadratically with sequence length, severely degrading tokens/sec.

## Documentation
See the `docs/` folder for deeper architectural dives:
- [Architecture](docs/architecture.md)
- [Methodology](docs/methodology.md)
- [Limitations](docs/limitations.md)

## Troubleshooting
- **`ModuleNotFoundError: No module named 'torch'`**: Ensure you activated the virtual environment and ran `pip install -r requirements.txt`.
- **`bitsandbytes` warnings**: Ensure you are running on a CUDA-capable GPU. bitsandbytes 8-bit/4-bit quantization will fail and fallback to FP32/FP16 on CPU-only machines.
