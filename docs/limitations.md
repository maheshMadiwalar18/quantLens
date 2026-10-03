# Limitations

While this benchmarking engine provides highly accurate localized metrics, consider the following limitations:

1. **Static Prompting**: Benchmarks use highly uniform repeated prompts to standardize calculations. Real-world prompt distributions are highly variable.
2. **Lack of Continuous Batching**: This engine measures static batching (where all sequences start and finish simultaneously). Production engines like vLLM or TGI use continuous batching, which yields significantly higher throughput in a real server environment.
3. **bitsandbytes Overhead**: The INT8 and 4-bit quantizations used here (via `bitsandbytes`) dequantize weights back to FP16 for the forward pass computation. While this drastically reduces VRAM, it introduces compute overhead, often resulting in lower tokens/sec than native FP16. This is a limitation of the specific quantization backend, not an inherent property of all integer math.
4. **Time To First Token (TTFT)**: Currently, the native HF `generate` call is timed as a whole. Measuring TTFT requires a custom `TextStreamer` implementation which is outside the scope of the current engine iteration.
