import yaml
import itertools
import os
import time
import pandas as pd
from benchmark.config import BenchmarkConfig, Precision
from benchmark.runner import run_benchmark

def main():
    print("=== Stage 8: Automated Experiment Matrix ===\n")
    
    yaml_path = "configs/experiments.yaml"
    if not os.path.exists(yaml_path):
        print(f"Error: Could not find {yaml_path}")
        return
        
    with open(yaml_path, "r") as f:
        matrix_cfg = yaml.safe_load(f)
        
    model_name = matrix_cfg["model"]["name"]
    precisions = [Precision(p) for p in matrix_cfg["precisions"]]
    batch_sizes = matrix_cfg["batch_sizes"]
    use_caches = matrix_cfg["use_cache"]
    context_lengths = matrix_cfg["context_lengths"]
    
    # Generate Cartesian product of all configurations
    combinations = list(itertools.product(precisions, batch_sizes, use_caches, context_lengths))
    print(f"Discovered {len(combinations)} unique configurations to benchmark.")
    
    all_results = []
    
    for idx, (prec, bz, uc, cl) in enumerate(combinations):
        exp_id = f"{prec.value}_b{bz}_c{int(uc)}_l{cl}"
        print(f"\n--- Running Experiment {idx+1}/{len(combinations)}: {exp_id} ---")
        
        cfg = BenchmarkConfig(
            experiment_id=exp_id,
            model_name=model_name,
            precision=prec,
            batch_size=bz,
            use_cache=uc,
            input_length=cl,
            num_warmup_runs=1,     # Reduced for matrix speed
            num_benchmark_runs=3,  # Reduced for matrix speed
            run_perplexity=False   # Disabled for massive matrices by default
        )
        
        try:
            agg_result = run_benchmark(cfg)
            all_results.append(agg_result)
            
            if agg_result.failed_oom:
                print(f"-> Experiment {exp_id} Failed (CUDA OOM/Unsupported). Continuing to next...")
            else:
                print(f"-> Experiment {exp_id} Success: {agg_result.aggregate_tps_mean:.2f} tok/s")
                
            # Save raw JSON immediately to prevent data loss on crash
            os.makedirs("results/raw", exist_ok=True)
            with open(f"results/raw/{exp_id}.json", "w") as f:
                f.write(agg_result.model_dump_json(indent=2))
                
        except Exception as e:
            print(f"-> Experiment {exp_id} crashed fatally: {e}. Recovering...")
            
    print("\n--- Compiling Final Dataset ---")
    
    # Save final flattened CSV dataset
    df_data = []
    for r in all_results:
        df_data.append({
            "experiment_id": r.experiment_id,
            "model": r.model,
            "precision": r.precision,
            "batch_size": r.batch_size,
            "use_cache": r.kv_cache,
            "input_length": r.input_tokens_per_request,
            "failed_oom": r.failed_oom,
            "mean_tps": r.aggregate_tps_mean,
            "mean_latency_s": r.latency_mean,
            "peak_vram_mb": r.peak_vram_mb
        })
    
    df = pd.DataFrame(df_data)
    out_csv = f"results/matrix_dataset_{int(time.time())}.csv"
    df.to_csv(out_csv, index=False)
    
    print(f"\nMatrix completed! Full dataset saved to {out_csv}")

if __name__ == "__main__":
    main()
