import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Загрузка модели и токенизатора GPT-3
model_name = 'ai-forever/ruGPT-3.5-13B'
cache_dir = "D:/models/huggingface"
model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)


# Перевод модели в режим оценки
model.eval()

def generate_text(prompt, max_length=100, temperature=1.0, top_k=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)
    
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            # max_length=max_length, 
            temperature=temperature,
            max_new_tokens=max_length,
            num_beams=4, 
            top_k=top_k,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def main():
    print("ruGPT3.5 Text Generation. Type 'exit' to quit.")
    
    while True:
        prompt = input("Enter your prompt: ")
        
        if prompt.lower() == 'exit':
            print("Exiting...")
            break
        
        generated_text = generate_text(prompt)
        print("\nGenerated Text:\n" + generated_text + "\n")

if __name__ == "__main__":
    main()
