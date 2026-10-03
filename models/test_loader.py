import torch
from models.loader import load_model_and_tokenizer

def main():
    print("=== Stage 2: Model Loading and Inference Verification ===\n")
    
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    
    prompt = "Explain what a binary search tree is in two sentences."
    
    # Prepare the prompt using the model's expected chat template
    messages = [
        {"role": "system", "content": "You are a helpful and concise assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print(f"Prompt: {prompt}\n")
    print("Generating response...")
    
    # 7. Disable gradients during inference
    # torch.inference_mode() is a more extreme version of torch.no_grad()
    # which further optimizes memory and performance for inference-only workloads.
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the newly generated tokens (ignoring the input prompt tokens)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n--- Response ---")
    print(response.strip())
    print("----------------\n")
    print("Stage 2 verification completed.")

if __name__ == "__main__":
    main()
