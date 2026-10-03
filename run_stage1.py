import json
from benchmark.config import ExperimentConfig, QuantizationType
from benchmark.hardware import get_hardware_info

def main():
    print("=== Stage 1: Hardware & Configuration Verification ===\n")
    
    # 1. Test Hardware Profiling
    print("Gathering Hardware Information...")
    hw_info = get_hardware_info()
    print(json.dumps(hw_info, indent=2))
    
    if not hw_info["cuda_available"]:
        print("\nWARNING: CUDA is not available. PyTorch is running on CPU.")
        print("Benchmarking requires a CUDA-capable GPU for accurate results.")
    else:
        print(f"\nDetected {hw_info['gpu_count']} GPU(s). Everything looks good for benchmarking!")

    # 2. Test Configuration Instantiation
    print("\nCreating Sample Experiment Configuration...")
    config = ExperimentConfig(
        batch_size=4,
        quantization=QuantizationType.INT8,
        kv_cache_enabled=True
    )
    
    print(f"Experiment ID: {config.get_experiment_id()}")
    print("Configuration Dump:")
    print(config.model_dump_json(indent=2))
    
    print("\nStage 1 verification completed successfully.")

if __name__ == "__main__":
    main()
