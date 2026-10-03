# QuantLens Complete Reverse Audit

## CRITICAL ISSUES
1. **Perplexity Masking Defect (`evaluation/perplexity.py`)**
   - *What is wrong*: The code clones `input_ids` to `target_ids` but never actually masks the context tokens with `-100`. 
   - *Why it matters*: CrossEntropyLoss calculates loss over the entire sequence instead of just the autoregressively generated stride.
   - *Benchmark effect*: Perplexity scores are artificially deflated (appearing better) because the model is graded on the known context tokens.
   - *Fix*: Implement `target_ids[:, :-trg_len] = -100` before the forward pass.

2. **Terminology & Metric Definitions (`benchmark/metrics.py`)**
   - *What is wrong*: `latency / batch_size` is labeled as "average latency per request".
   - *Why it matters*: In static batching, all requests complete at the same time (the total batch latency). Users do not experience `total / batch_size` latency.
   - *Benchmark effect*: Misleads engineers into thinking batch size 16 yields incredibly fast individual request times, ignoring TTFT and continuous batching realities.
   - *Fix*: Rename to `batch_amortized_latency` and decouple it from user-experienced latency.

3. **Silent VRAM Omissions (`benchmark/inference.py`)**
   - *What is wrong*: We only track `torch.cuda.memory_allocated()`.
   - *Why it matters*: The PyTorch caching allocator reserves memory (fragmentation) that isn't actively allocated to tensors but is blocked from other processes.
   - *Benchmark effect*: VRAM consumption appears lower than it practically is.
   - *Fix*: Record both `memory_allocated` and `memory_reserved`.

## HIGH ISSUES
1. **Inefficient Experiment Matrix (`configs/experiments.yaml` & `run_experiment.py`)**
   - *What is wrong*: The matrix runner executes 72 permutations and reloads the model from disk every time.
   - *Why it matters*: Model instantiation to GPU can take 10-30 seconds. 72 reloads adds ~30 minutes of dead time to the suite.
   - *Benchmark effect*: Frustrates reproducibility; tests run too slowly.
   - *Fix*: Refactor runner to sort configurations by `Precision` and cache the loaded `AutoModel` inside the loop.

2. **Uncontrolled Prompt Generation**
   - *What is wrong*: Prompts are created by string multiplication and relying on the tokenizer to truncate/pad.
   - *Why it matters*: Depending on the tokenizer, this might not yield exact token lengths, and padding tokens (`<pad>`) are attended to differently depending on the attention mask.
   - *Benchmark effect*: Token generation speeds might be skewed by massive padding blocks.
   - *Fix*: Construct deterministic integer sequences of exact lengths directly via PyTorch tensors.

## MEDIUM ISSUES
1. **Dashboard Empty State (`dashboard/`)**
   - *What is wrong*: The React app is broken / uninitialized.
   - *Why it matters*: Phase 15 requires a true quantitative dashboard.
   - *Fix*: Re-scaffold with Vite safely, handle empty states gracefully.

2. **README Theoretical Claims**
   - *What is wrong*: The README claims batching yields 3x improvements before measurements run.
   - *Fix*: Separate into "Theoretical Expectations" and "Measured Findings".

## LOW ISSUES
1. **Duplicate Runners**
   - *What is wrong*: `benchmark/matrix.py` and `benchmark/run_experiment.py` are identical.
   - *Fix*: Delete `matrix.py` and unify under `run_experiment.py`.
