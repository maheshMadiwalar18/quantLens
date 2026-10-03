from pydantic import BaseModel
from typing import Optional, List

class BenchmarkIterationResult(BaseModel):
    iteration_id: int
    failed_oom: bool = False
    
    batch_size: int
    input_tokens_per_request: int
    output_tokens_per_request: int
    total_generated_tokens: int
    
    total_batch_latency_seconds: float
    average_latency_per_request: float
    
    aggregate_tokens_per_second: float
    tokens_per_second_per_request: float
    
    vram_allocated_mb: float
    peak_vram_mb: float

class BenchmarkAggregateResult(BaseModel):
    experiment_id: str
    model: str
    parameter_count: int
    precision: str
    batch_size: int
    kv_cache: bool
    iterations: int
    
    failed_oom: bool
    
    gpu_name: Optional[str]
    cuda_version: Optional[str]
    pytorch_version: str
    
    input_tokens_per_request: int
    output_tokens_per_request: int
    
    latency_mean: float
    latency_median: float
    latency_std: float
    
    aggregate_tps_mean: float
    aggregate_tps_median: float
    aggregate_tps_std: float
    
    peak_vram_mb: float
    
    perplexity: Optional[float] = None
    loss: Optional[float] = None
    
    raw_iterations: List[BenchmarkIterationResult]
