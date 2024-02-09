import re
from torch.utils.data import Dataset
import pandas as pd


class ShakespeareDataset(Dataset):
    def __init__(self, path, col, tokenizer):
        self._data = pd.read_csv(path)
        
        self.texts = self._data[col].tolist()[:1000]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):        
        tokens = self.tokenizer(self.texts, return_tensors='pt', padding=True, truncation=True)
        ids = tokens['input_ids'][idx]
        attention = tokens['attention_mask'][idx]     
        return {'input_ids' : ids,
               'attention_mask': attention}
    



def generate_text(model, tokenizer, min_length, max_length, prompt_text=None):
    if prompt_text is None:
        prompt_text = "<BOS>"
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    else:
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    
    # Generate text using the model
    output = model.generate(input_ids, min_length=min_length, max_length=max_length, do_sample=True)
    
    start = len(prompt_text)+1

    # Decode the generated text and return
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)   
    
    matches = re.finditer(r'\.', generated_text)
    finish = len(generated_text)
    for match in matches:
        if match.end() < max_length:
            finish = match.end()

    generated_text = generated_text[start:finish] .strip()

    return generated_text