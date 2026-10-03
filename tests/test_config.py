import pytest
from pydantic import ValidationError
from benchmark.config import BenchmarkConfig, Precision

def test_default_config():
    config = BenchmarkConfig()
    assert config.precision == Precision.FP16
    assert config.batch_size == 1
    assert config.use_cache is True
    assert config.input_length == 512

def test_valid_config_overrides():
    config = BenchmarkConfig(
        precision=Precision.INT8,
        batch_size=4,
        input_length=1024,
        use_cache=False,
        num_warmup_runs=1,
        num_benchmark_runs=10
    )
    assert config.precision == Precision.INT8
    assert config.batch_size == 4
    assert config.input_length == 1024
    assert config.use_cache is False

def test_invalid_precision():
    with pytest.raises(ValueError):
        BenchmarkConfig(precision="fp32")

def test_invalid_batch_size():
    with pytest.raises(ValidationError):
        BenchmarkConfig(batch_size=0)
        
def test_valid_input_lengths():
    for length in [256, 512, 1024, 2048]:
        config = BenchmarkConfig(input_length=length)
        assert config.input_length == length

def test_invalid_input_length():
    with pytest.raises(ValidationError):
        BenchmarkConfig(input_length=300)
