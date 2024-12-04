import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, EarlyStopping

class Logger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs['loss'])
        self.train_accuracy.append(logs['accuracy'])
        self.val_loss.append(logs['val_loss'])
        self.val_accuracy.append(logs['val_accuracy'])

    def on_train_begin(self, logs=None):
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
logger = Logger()

model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=100,
        validation_data=(X_test, y_test),
        callbacks=[logger, early_stopping]
    )
model.save(os.getcwd()+"/model.keras")

train_loss2 = np.array(logger.train_loss)
train_accuracy2 = np.array(logger.train_accuracy)
val_loss2 = np.array(logger.val_loss)
val_accuracy2 = np.array(logger.val_accuracy)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(len(train_loss2)), train_loss2, label='Train Loss optimized')
plt.plot(range(len(val_loss2)), val_loss2, label='Validation optimized')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(len(train_accuracy2)), train_accuracy2, label='Train Accuracy optimized')
plt.plot(range(len(val_accuracy2)), val_accuracy2, label='Validation Accuracy optimized')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()