import torch
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple, Any
from benchmark.config import Precision

def load_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", precision: Precision = Precision.FP16) -> Tuple[Any, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        warnings.warn(
            "CUDA is not available. Falling back to CPU. "
            "bitsandbytes quantization requires CUDA. Precision will fallback to FP32.", 
            UserWarning
        )
        precision_dtype = torch.float32
        quantization_config = None
    else:
        precision_dtype = torch.float16
        
        try:
            if precision == Precision.INT8:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif precision == Precision.INT4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            else:
                quantization_config = None
        except Exception as e:
            warnings.warn(f"Failed to configure bitsandbytes quantization: {e}")
            quantization_config = None

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure padding for batched inference
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Required for batched generation in decoder-only autoregressive models
    tokenizer.padding_side = "left"
    
    print(f"Loading model {model_name} on {device} (requested precision: {precision.value})...")
    
    kwargs = {
        "torch_dtype": precision_dtype,
        "device_map": device,
    }
    
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    
    print("\n--- Model Info ---")
    print(f"Model Name:      {model_name}")
    print(f"Parameter Count: {param_count:,}")
    print(f"Device:          {model.device}")
    
    if device == "cuda":
        allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"VRAM Allocated:  {allocated_mb:.2f} MB")
    print("------------------\n")
    
    return model, tokenizer
