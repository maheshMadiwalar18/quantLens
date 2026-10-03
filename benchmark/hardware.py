import platform
import psutil
import torch
import json

def get_hardware_info() -> dict:
    """
    Retrieves information about the system's hardware and PyTorch environment.
    This metadata is critical for contextualizing benchmark results.
    """
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu": platform.processor(),
        "system_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = []
        for i in range(info["gpu_count"]):
            device_props = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "id": i,
                "name": device_props.name,
                "total_memory_gb": round(device_props.total_memory / (1024**3), 2),
                "compute_capability": f"{device_props.major}.{device_props.minor}"
            })
    
    return info

def get_current_vram_usage(device_id: int = 0) -> float:
    """
    Returns the currently allocated GPU VRAM in MB.
    """
    if not torch.cuda.is_available():
        return 0.0
    
    allocated = torch.cuda.memory_allocated(device_id)
    return allocated / (1024 * 1024)

if __name__ == "__main__":
    print("=== Hardware Detection ===")
    print(json.dumps(get_hardware_info(), indent=2))
