import os
import argparse
import time
import statistics
import torch
from models.loader import load_model_and_tokenizer
from benchmark.inference import run_single_inference
from benchmark.metrics import BenchmarkAggregateResult
from benchmark.config import BenchmarkConfig, Precision
from evaluation.perplexity import evaluate_perplexity

def get_hardware_context():
    gpu_name = None
    cuda_version = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
    return gpu_name, cuda_version, torch.__version__

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def run_benchmark(config: BenchmarkConfig) -> BenchmarkAggregateResult:
    """Core function to execute a benchmarking configuration."""
    model, tokenizer = load_model_and_tokenizer(config.model_name, precision=config.precision)
    
    loss_val = None
    ppl_val = None
    if config.run_perplexity:
        loss_val, ppl_val = evaluate_perplexity(model, tokenizer)
        
    prompt = "Explain the history of artificial intelligence in immense detail, focusing on deep learning and transformers. " * 100
    
    for i in range(config.num_warmup_runs):
        res = run_single_inference(
            model, tokenizer, prompt, 
            batch_size=config.batch_size, 
            input_length=config.input_length,
            max_new_tokens=config.max_new_tokens,
            use_cache=config.use_cache,
            iteration_id=-1
        )
        if res.failed_oom or (res.total_batch_latency_seconds == 0.0 and res.total_generated_tokens == 0):
            break
            
    results = []
    failed_oom = False
    
    if not (config.num_warmup_runs > 0 and 'res' in locals() and (res.failed_oom or res.total_batch_latency_seconds == 0.0)):
        for i in range(config.num_benchmark_runs):
            res = run_single_inference(
                model, tokenizer, prompt, 
                batch_size=config.batch_size, 
                input_length=config.input_length,
                max_new_tokens=config.max_new_tokens,
                use_cache=config.use_cache,
                iteration_id=i+1
            )
            
            if res.failed_oom:
                failed_oom = True
                break
            if res.total_batch_latency_seconds == 0.0:
                break
                
            results.append(res)
            
    gpu_name, cuda_version, pt_version = get_hardware_context()
    param_count = sum(p.numel() for p in model.parameters())
    
    if failed_oom or not results:
        agg = BenchmarkAggregateResult(
            experiment_id=config.experiment_id, model=config.model_name, parameter_count=param_count, precision=config.precision.value,
            batch_size=config.batch_size, kv_cache=config.use_cache, iterations=0,
            failed_oom=True, gpu_name=gpu_name, cuda_version=cuda_version, pytorch_version=pt_version,
            input_tokens_per_request=config.input_length, output_tokens_per_request=0,
            latency_mean=0.0, latency_median=0.0, latency_std=0.0,
            aggregate_tps_mean=0.0, aggregate_tps_median=0.0, aggregate_tps_std=0.0,
            peak_vram_mb=0.0, perplexity=ppl_val, loss=loss_val, raw_iterations=[]
        )
    else:
        latencies = [r.total_batch_latency_seconds for r in results]
        tps = [r.aggregate_tokens_per_second for r in results]
        peak_vram = max([r.peak_vram_mb for r in results])
        
        agg = BenchmarkAggregateResult(
            experiment_id=config.experiment_id, model=config.model_name, parameter_count=param_count, precision=config.precision.value,
            batch_size=config.batch_size, kv_cache=config.use_cache, iterations=len(results),
            failed_oom=False, gpu_name=gpu_name, cuda_version=cuda_version, pytorch_version=pt_version,
            input_tokens_per_request=results[0].input_tokens_per_request,
            output_tokens_per_request=results[0].output_tokens_per_request,
            latency_mean=statistics.mean(latencies), latency_median=statistics.median(latencies),
            latency_std=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            aggregate_tps_mean=statistics.mean(tps), aggregate_tps_median=statistics.median(tps),
            aggregate_tps_std=statistics.stdev(tps) if len(tps) > 1 else 0.0,
            peak_vram_mb=peak_vram, perplexity=ppl_val, loss=loss_val, raw_iterations=results
        )
    return agg

def main():
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Runner")
    parser.add_argument("--precision", type=str, choices=["fp16", "int8", "4bit"], default="fp16")
    parser.add_argument("--batch-size", type=int, choices=[1, 4, 8, 16], default=1)
    parser.add_argument("--input-length", type=int, choices=[256, 512, 1024, 2048], default=512)
    parser.add_argument("--use-cache", type=str2bool, default=True)
    parser.add_argument("--run-perplexity", type=str2bool, default=False)
    
    args = parser.parse_args()
    config = BenchmarkConfig(
        experiment_id=f"cli_{int(time.time())}",
        precision=Precision(args.precision), 
        batch_size=args.batch_size,
        input_length=args.input_length,
        use_cache=args.use_cache,
        run_perplexity=args.run_perplexity
    )
    
    print(f"=== Stage 7: CLI Benchmarking ===")
    os.makedirs("results", exist_ok=True)
    
    agg = run_benchmark(config)
    
    out_file = f"results/benchmark_{config.precision.value}_b{config.batch_size}_{int(time.time())}.json"
    with open(out_file, "w") as f:
        f.write(agg.model_dump_json(indent=2))
        
    print(f"Mean Aggregate TPS: {agg.aggregate_tps_mean:.2f} tokens/sec")
    print(f"Detailed JSON report saved to: {out_file}")

if __name__ == "__main__":
    main()
