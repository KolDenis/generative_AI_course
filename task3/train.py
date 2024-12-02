import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, EarlyStopping
from keras.regularizers import l2
from keras.layers import Dropout, BatchNormalization

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

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

model2 = models.Sequential([
    layers.Input(shape=(28*28,)),
    layers.Dense(64, activation='relu'),
    Dropout(0.1),
    layers.Dense(32, activation='relu'),
    Dropout(0.1),
    layers.Dense(10, activation='softmax', kernel_regularizer=l2(0.001))
])
model2.compile(
    optimizer='adam',
    loss='crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
logger2 = Logger()
model2.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[logger2, early_stopping]
    )
model2.save(os.getcwd()+"/model.keras")

train_loss2 = np.array(logger2.train_loss)
train_accuracy2 = np.array(logger2.train_accuracy)
val_loss2 = np.array(logger2.val_loss)
val_accuracy2 = np.array(logger2.val_accuracy)

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

