import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, BertConfig, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

print(torch.cuda.is_available())

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name, num_labels=2, hidden_dropout_prob=0.5)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))
test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(500))

train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=lambda x: {
    'input_ids': torch.tensor([f['input_ids'] for f in x]),
    'attention_mask': torch.tensor([f['attention_mask'] for f in x]),
    'labels': torch.tensor([f['label'] for f in x])
})
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=lambda x: {
    'input_ids': torch.tensor([f['input_ids'] for f in x]),
    'attention_mask': torch.tensor([f['attention_mask'] for f in x]),
    'labels': torch.tensor([f['label'] for f in x])
})

optimizer = AdamW(model.parameters(), lr=0.0001)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

epochs = 5

num_training_steps = len(train_loader) * epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

model.train()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    loop = tqdm(train_loader, leave=True)
    epoch_train_loss = 0
    all_train_predictions = []
    all_train_labels = []

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        epoch_train_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        all_train_predictions.extend(predictions.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())

        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item())

    epoch_train_loss /= len(train_loader)
    train_accuracy = accuracy_score(all_train_labels, all_train_predictions)

    train_losses.append(epoch_train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()
    epoch_test_loss = 0
    all_test_predictions = []
    all_test_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            epoch_test_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            all_test_predictions.extend(predictions.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    epoch_test_loss /= len(test_loader)
    test_accuracy = accuracy_score(all_test_labels, all_test_predictions)

    test_losses.append(epoch_test_loss)
    test_accuracies.append(test_accuracy)

    model.train()

    print(f"Epoch {epoch + 1} Summary:")
    print(f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

model.save_pretrained("./bert-imdb-pytorch")
tokenizer.save_pretrained("./bert-imdb-pytorch")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(len(train_losses)), train_losses, label='Train Loss optimized')
plt.plot(range(len(test_losses)), test_losses, label='Validation optimized')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(train_accuracies)), train_accuracies, label='Train Accuracy optimized')
plt.plot(range(len(test_accuracies)), test_accuracies, label='Validation Accuracy optimized')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
