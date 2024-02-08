from transformers import GPT2LMHeadModel, GPT2Tokenizer
from generator import generate_text

# Load pretrained GPT-2 model and tokenizer
model_path = "./tuned_model"
token_path = "./tokenizer"

tokenizer = GPT2Tokenizer.from_pretrained(token_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

min_length = 400
max_length = 600

generated_text = generate_text(model, tokenizer, min_length, max_length)

with open('data/generated.txt', 'w', encoding='utf-8') as file:
	file.write(generated_text)
