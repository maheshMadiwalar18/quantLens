# Methodology

## Latency vs Throughput
We strictly differentiate between:
- **Batch Latency**: Total wall-clock time to execute a forward pass for all sequences.
- **Tokens/sec (Throughput)**: Total generated tokens across all sequences divided by batch latency.

## CUDA Synchronization
Because PyTorch schedules GPU operations asynchronously, naive `time.time()` wrappers will record artificially low execution times. Our engine forces CPU-GPU synchronization via `torch.cuda.synchronize()` before and after the inference block to measure true latency.

## Warm-up Iterations
GPUs enter idle states, and PyTorch often performs lazy initialization or memory reallocation on the first pass. We execute untimed warm-up passes to bring the GPU to steady-state performance before recording benchmarks.

## Perplexity Evaluation
To ensure quantization doesn't silently break the model, we calculate the Negative Log-Likelihood over a fixed slice of the Wikitext-2 dataset. The context window is masked with `-100` so the CrossEntropy loss only penalizes the model for the newly predicted stride tokens.
