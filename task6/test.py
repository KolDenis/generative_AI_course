import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "./bert-imdb-pytorch"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

def predict_text(texts, model, tokenizer, device):
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    return predictions.cpu().numpy()

texts = [
    "The movie was absolutely amazing! I loved it.",
    "The plot was terrible and the acting was even worse."
]

predictions = predict_text(texts, model, tokenizer, device)
for text, pred in zip(texts, predictions):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")
