# Experiments Configuration

Experiments are defined declaratively in `configs/experiments.yaml`. The matrix runner computes the Cartesian product of all provided arrays to test every possible combination.

Example structure:
```yaml
model:
  name: Qwen/Qwen2.5-1.5B-Instruct

precisions:
  - fp16
  - int8

batch_sizes:
  - 1
  - 8

use_cache:
  - true

context_lengths:
  - 512
```
This generates $2 \times 2 \times 1 \times 1 = 4$ unique experiments.

To run:
```bash
python -m benchmark.run_experiment --config configs/experiments.yaml
```
