from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_response(prompt, max_length=50):

    inputs = tokenizer.encode(prompt, return_tensors="pt")  
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1) 
    response = tokenizer.decode(outputs[0], skip_special_tokens=True) 
    return response

def main():
    print("Welcome to the Chat Service")
    while True:
        user_input = input("You:    ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = generate_response(user_input)
        print(f"Bot     : {response}")

if __name__ == "__main__":
    main()
