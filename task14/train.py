import os
import tensorflow as tf
from datasets import load_dataset
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq

dataset = load_dataset("amazon_polarity")

train_data = dataset["train"].shuffle(seed=42).select(range(10000))
valid_data = dataset["test"].shuffle(seed=42).select(range(2000))

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["content"], padding="max_length", truncation=True, max_length=128)

train_data = train_data.map(tokenize_function, batched=True)
valid_data = valid_data.map(tokenize_function, batched=True)

columns_to_keep = ["input_ids", "attention_mask", "labels"]
train_data = train_data.remove_columns([col for col in train_data.column_names if col not in columns_to_keep])
valid_data = valid_data.remove_columns([col for col in valid_data.column_names if col not in columns_to_keep])

train_data.set_format(type="tensorflow")
valid_data.set_format(type="tensorflow")

training_args = TrainingArguments(
    output_dir="./review_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_total_limit=2,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer),
)

trainer.train()

output_dir = "./fine_tuned_review_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

def generate_review(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="tf")
    outputs = model.generate(inputs["input_ids"], max_length=max_length, num_return_sequences=1, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "This product is amazing because"
print(generate_review(prompt))
