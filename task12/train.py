import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, datasets
import matplotlib.pyplot as plt
import os

batch_size = 32
epochs = 100
learning_rate = 0.001

output_dir = "res"
os.makedirs(output_dir, exist_ok=True)

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

def create_ebm_model():
    model = models.Sequential([
        layers.Conv2D(64, (9, 9), padding='valid', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool2D(),
        layers.Conv2D(128, (5, 5), padding='valid'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(256, (3, 3), padding='valid'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(512),
        layers.ReLU(),
        layers.Dense(10)
    ])
    return model

def energy_loss(y_true, energy):
    y_true_one_hot = tf.one_hot(y_true, depth=10)
    pos_energy = tf.reduce_sum(y_true_one_hot * energy, axis=1)
    neg_energy = tf.reduce_logsumexp(energy, axis=1)
    margin = 10.0
    loss = tf.reduce_mean(margin - pos_energy + neg_energy)
    return loss

model = create_ebm_model()
model.summary()
optimizer = optimizers.Adam(learning_rate=learning_rate)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        energy = model(images, training=True)
        loss = energy_loss(labels, energy)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def test_step(images, labels):
    energy = model(images, training=False)
    predictions = tf.argmax(energy, axis=1, output_type=tf.int32)
    accuracy = tf.reduce_mean(tf.cast(predictions == tf.cast(labels, tf.int32), tf.float32))
    return accuracy

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

def save_epoch_examples(model, dataset, epoch, output_dir, num_examples=10):
    images, labels = next(iter(dataset))
    images = images[:num_examples]
    labels = labels[:num_examples]

    predictions = tf.argmax(model(images, training=False), axis=1, output_type=tf.int32)

    num_rows = 2
    num_cols = num_examples // num_rows

    plt.figure(figsize=(15, 6))
    for i in range(num_examples):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i].numpy())
        plt.title(f"True: {labels[i]}, Pred: {predictions[i]}")
        plt.axis("off")

    save_path = os.path.join(output_dir, f"epoch_{epoch + 1}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Примеры сохранены: {save_path}")

for epoch in range(epochs):
    train_loss = 0.0
    for images, labels in train_dataset:
        loss = train_step(images, labels)
        train_loss += loss.numpy()

    test_accuracy = 0.0
    for images, labels in test_dataset:
        accuracy = test_step(images, labels)
        test_accuracy += accuracy.numpy()

    train_loss /= len(train_dataset)
    test_accuracy /= len(test_dataset)

    save_epoch_examples(model, test_dataset, epoch, output_dir)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {test_accuracy:.4f}")
