import pandas as pd
from sklearn.model_selection import train_test_split

csv_data = pd.read_csv('data/Shakespeare_data.csv')
data_shuffled = csv_data.sample(frac=1, random_state=42)
train_df, val_df = train_test_split(data_shuffled, test_size=0.2, random_state=42)

sents_train = "\n".join(train_df['PlayerLine'].tolist())
sents_val = "\n".join(val_df['PlayerLine'].tolist())


with open('data/train.txt', 'w', encoding='utf-8') as file_train:
	file_train.write(sents_train)

with open('data/val.txt', 'w', encoding='utf-8') as file_val:
	file_val.write(sents_val)