import torch
import time
import warnings
from transformers import PreTrainedModel, PreTrainedTokenizer
from benchmark.metrics import BenchmarkIterationResult

def run_single_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    batch_size: int = 1,
    input_length: int = 512,
    max_new_tokens: int = 128,
    use_cache: bool = True,
    iteration_id: int = 0
) -> BenchmarkIterationResult:
    
    # Generate identical prompts
    prompts = [prompt] * batch_size
    
    # Enforce exact input length via tokenizer truncation/padding
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        max_length=input_length, 
        truncation=True, 
        padding="max_length"
    ).to(model.device)
    
    actual_input_len = inputs.input_ids.shape[1]
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.perf_counter()
    
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                pad_token_id=tokenizer.pad_token_id
            )
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return BenchmarkIterationResult(
                iteration_id=iteration_id,
                failed_oom=True,
                batch_size=batch_size,
                input_tokens_per_request=actual_input_len,
                output_tokens_per_request=max_new_tokens,
                total_generated_tokens=0,
                total_batch_latency_seconds=0.0,
                average_latency_per_request=0.0,
                aggregate_tokens_per_second=0.0,
                tokens_per_second_per_request=0.0,
                vram_allocated_mb=0.0,
                peak_vram_mb=0.0
            )
        else:
            raise e
    except Exception as e:
        # Catch other errors, e.g., if use_cache=False is unsupported by the model
        warnings.warn(f"Generation failed: {e}")
        return BenchmarkIterationResult(
            iteration_id=iteration_id,
            failed_oom=False,
            batch_size=batch_size,
            input_tokens_per_request=actual_input_len,
            output_tokens_per_request=max_new_tokens,
            total_generated_tokens=0,
            total_batch_latency_seconds=0.0,
            average_latency_per_request=0.0,
            aggregate_tokens_per_second=0.0,
            tokens_per_second_per_request=0.0,
            vram_allocated_mb=0.0,
            peak_vram_mb=0.0
        )
            
    latency = end_time - start_time
    output_len = outputs.shape[1] - actual_input_len
    total_generated_tokens = output_len * batch_size
    
    aggregate_tps = total_generated_tokens / latency if latency > 0 else 0.0
    tps_per_request = aggregate_tps / batch_size
    average_latency_per_request = latency / batch_size
    
    vram_allocated = 0.0
    peak_vram = 0.0
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
    return BenchmarkIterationResult(
        iteration_id=iteration_id,
        failed_oom=False,
        batch_size=batch_size,
        input_tokens_per_request=actual_input_len,
        output_tokens_per_request=output_len,
        total_generated_tokens=total_generated_tokens,
        total_batch_latency_seconds=latency,
        average_latency_per_request=average_latency_per_request,
        aggregate_tokens_per_second=aggregate_tps,
        tokens_per_second_per_request=tps_per_request,
        vram_allocated_mb=vram_allocated,
        peak_vram_mb=peak_vram
    )
