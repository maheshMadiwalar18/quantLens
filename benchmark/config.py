from enum import Enum
from pydantic import BaseModel, Field, model_validator

class Precision(str, Enum):
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "4bit"

class BenchmarkConfig(BaseModel):
    experiment_id: str = Field(default="")
    model_name: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct")
    precision: Precision = Field(default=Precision.FP16)
    batch_size: int = Field(default=1, ge=1)
    input_length: int = Field(default=512)
    max_new_tokens: int = Field(default=128, ge=1)
    use_cache: bool = Field(default=True)
    num_warmup_runs: int = Field(default=3, ge=0)
    num_benchmark_runs: int = Field(default=5, ge=1)
    run_perplexity: bool = Field(default=False)
    
    @model_validator(mode='after')
    def validate_input_length(self):
        valid_lengths = [256, 512, 1024, 2048]
        if self.input_length not in valid_lengths:
            raise ValueError(f"input_length must be one of {valid_lengths}")
        return self
