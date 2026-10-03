# Architecture

The LLM Inference Optimization Lab is designed as a modular, stateless benchmarking engine.

## Core Components
1. **Configuration (`benchmark/config.py`)**: Uses Pydantic to enforce strict benchmarking schemas.
2. **Model Loader (`models/loader.py`)**: Centralizes Hugging Face model instantiation, automatically handling FP16/INT8/4-bit quantization and device allocation.
3. **Inference Engine (`benchmark/inference.py`)**: Manages the exact execution of the model, tracking precise CUDA execution times and handling Out-Of-Memory exceptions safely.
4. **Evaluator (`evaluation/perplexity.py`)**: Measures model predictive quality using the Wikitext-2 dataset.
5. **Matrix Runner (`benchmark/run_experiment.py`)**: Reads a YAML configuration, computes the Cartesian product of parameters, and executes the suite autonomously.

## Data Flow
YAML Config -> Matrix Runner -> Model Loader -> Inference Engine -> Metric Aggregation -> JSON/CSV Dataset -> (Dashboard API).
