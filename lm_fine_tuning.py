from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, TextDataset

# Define model and training parameters

model_name = "gpt2"
train_path = 'data/train.txt'
validation_path = 'data/val.txt'
output_dir="./shakespeare_lm1"
model_path = './tuned_model'
tokenizer_path = './tokenizer'
block_size = 256
epochs = 10
batch_size = 16

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Load training and validation datasets
train_dataset = TextDataset(
                    file_path=train_path,
                    tokenizer=tokenizer,
                    block_size=block_size            
            )

validation_dataset = TextDataset(
                    file_path=validation_path,
                    tokenizer=tokenizer,
                    block_size=block_size            
            )

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    save_steps=10_000,
    save_total_limit=2,
)

# Define Trainer object with both training and validation datasets
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset, 
)

# Start training
trainer.train()

trainer.save_model(model_path)
tokenizer.save_pretrained(tokenizer_path)