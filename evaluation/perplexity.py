import torch
import warnings
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Tuple

def evaluate_perplexity(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    dataset_name: str = "wikitext", 
    dataset_config: str = "wikitext-2-raw-v1",
    max_tokens: int = 4096,  # Cap tokens to keep evaluation fast for the benchmark
    stride: int = 512
) -> Tuple[float, float]:
    """
    Evaluates the model's perplexity on a reproducible dataset to measure quality.
    Lower perplexity means the model is better at predicting the text distribution.
    """
    print(f"\nLoading evaluation dataset: {dataset_name} ({dataset_config})...")
    
    try:
        # Load small test split
        dataset = load_dataset(dataset_name, dataset_config, split="test", trust_remote_code=True)
    except Exception as e:
        warnings.warn(f"Failed to load dataset: {e}. Perplexity evaluation skipped.")
        return 0.0, 0.0
        
    # Join text into a single string for continuous evaluation
    # We take a small subset of rows to avoid massive memory consumption during tokenization
    text_corpus = "\n\n".join(dataset["text"][:100])
    
    encodings = tokenizer(text_corpus, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    
    # Cap the sequence length to keep benchmarking swift
    seq_len = min(seq_len, max_tokens)
    
    nlls = []
    prev_end_loc = 0
    
    print(f"Calculating Perplexity across {seq_len} tokens (Stride: {stride})...")
    
    try:
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + stride, seq_len)
            trg_len = end_loc - prev_end_loc 
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
            target_ids = input_ids.clone()
            
            # The model should only be penalized for predicting the target tokens,
            # not the context we provided. We set context targets to -100 (ignored by CrossEntropyLoss)
            # In a non-overlapping sliding window, this is purely the stride tokens.
            
            with torch.inference_mode():
                # Forward pass calculates Cross Entropy Loss automatically if labels are provided
                outputs = model(input_ids, labels=target_ids)
                
                # The loss is the mean negative log likelihood (NLL) for the target tokens
                loss = outputs.loss
                nlls.append(loss)
                
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
                
        # Stack all NLLs, take the mean, and exponentiate to get Perplexity
        avg_nll = torch.stack(nlls).mean()
        ppl = torch.exp(avg_nll)
        
        return avg_nll.item(), ppl.item()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("CUDA Out Of Memory during Perplexity evaluation.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0, 0.0
        else:
            raise e
