import os
import time
from benchmark.config import BenchmarkConfig, Precision
from benchmark.runner import run_benchmark

def main():
    config = BenchmarkConfig(
        experiment_id="real_bench_1",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        precision=Precision.FP16,
        batch_size=1,
        input_length=512,
        max_new_tokens=128,
        use_cache=True,
        num_warmup_runs=3,
        num_benchmark_runs=10,
        run_perplexity=False
    )
    
    print("Starting real benchmark. Please wait...")
    os.makedirs("results", exist_ok=True)
    
    agg = run_benchmark(config)
    
    out_file = f"results/real_benchmark_final.json"
    with open(out_file, "w") as f:
        f.write(agg.model_dump_json(indent=2))
        
    print("\n=== FINAL RESULTS ===")
    print(f"Mean Latency: {agg.latency_mean:.2f}s")
    print(f"Median Latency: {agg.latency_median:.2f}s")
    print(f"Std Dev Latency: {agg.latency_std:.2f}s")
    print(f"Mean TPS: {agg.aggregate_tps_mean:.2f} tokens/sec")
    print(f"Peak VRAM: {agg.peak_vram_mb:.2f} MB")
    print(f"Saved to: {out_file}")

if __name__ == "__main__":
    main()
