import kagglehub
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def preprocess_text(text, max_length=128):
    encoding = tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors="tf")
    return encoding

path = kagglehub.dataset_download("subhajournal/phishingemails")
df = pd.read_csv(path + '/phishing_email.csv')

df = df.drop(columns=['Unnamed: 0'])
df.columns = ['Email Text', 'Email Type']
df['Email Type'] = df['Email Type'].map({'Phishing Email': 0, 'Safe Email': 1})

train_texts = df['Email Text'].tolist()
train_texts = [str(text) if text is not None else '' for text in train_texts]

train_labels = df['Email Type'].tolist()

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="tf")

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings), train_labels
)).batch(32)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model.fit(train_dataset, epochs=1)

test_texts = [
    "Congratulations, you won a prize! Click here to claim your reward.",
    "Your account has been compromised. Please reset your password immediately.",
    "I would like to offer you an exclusive business opportunity.",
    "This is a reminder that your invoice is due tomorrow.",
    "We have detected suspicious activity in your account. Please verify your identity."
]

inputs = preprocess_text(test_texts)

predictions = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])[0]
predicted_labels = tf.argmax(predictions, axis=1).numpy()

labels = ["Safe", "Phishing"]
for text, label in zip(data, predicted_labels):
    print(f"Text: {text}\nPrediction: {labels[label]}\n")