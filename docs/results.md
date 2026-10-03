# Results Data Schema

The `run_experiment.py` script outputs a consolidated CSV dataset and individual JSON files for each run.

The JSON schema corresponds to `BenchmarkAggregateResult` in `benchmark/metrics.py`:
- `experiment_id`: Unique identifier combining the hyperparameters.
- `model`: Hugging Face model ID.
- `precision`: One of fp16, int8, 4bit.
- `batch_size`: Number of concurrent sequences.
- `kv_cache`: Boolean.
- `input_tokens_per_request`: Tokens in the prompt.
- `aggregate_tps_mean`: The main throughput metric (total tokens generated across the batch per second).
- `peak_vram_mb`: Maximum GPU memory allocated during the iteration.
- `perplexity`: Quality metric (lower is better). Computed if `--run-perplexity` is active.
