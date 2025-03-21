import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    generated_text = generate_text(prompt)
    print("\nGenerated Text:\n")
    print(generated_text)
