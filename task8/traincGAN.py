import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, BatchNormalization, Conv2DTranspose, Conv2D, Flatten, ReLU
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train.reshape(-1, 28 * 28)

latent_dim = 100
num_classes = 10

def build_generator(latent_dim, num_classes):
    generator = Sequential([
        Dense(64, input_dim=latent_dim + num_classes),
        BatchNormalization(),
        ReLU(),
        Dense(128),
        BatchNormalization(),
        ReLU(),
        Dense(28 * 28, activation='tanh')
    ])
    return generator

def build_discriminator(img_dim, num_classes):
    discriminator = Sequential([
        Dense(128, input_dim=img_dim + num_classes),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dense(64),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    return discriminator

generator = build_generator(latent_dim, num_classes)
discriminator = build_discriminator(28 * 28, num_classes)

g_optimizer = Adam(learning_rate=0.0005, beta_1=0.5)
d_optimizer = Adam(learning_rate=0.0005, beta_1=0.5)
loss_fn = tf.keras.losses.BinaryCrossentropy()

def train_cgan(generator, discriminator, epochs=10000, batch_size=64):
    for epoch in range(epochs):
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]
        real_images = real_images.reshape((batch_size, -1))
        real_labels = y_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        labels = [np.zeros(10) for label in real_labels]
        for i, label in enumerate(real_labels):
            labels[i][label] = 1

        noise_inp = np.hstack((noise, labels))
        fake_images = generator.predict(noise_inp)

        real_inp = np.hstack((real_images, labels))
        fake_inp = np.hstack((fake_images, labels))
        with tf.GradientTape() as d_tape:
            real_output = discriminator(real_inp)
            fake_output = discriminator(fake_inp)
            d_loss_real = loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = d_loss_real + d_loss_fake

        d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            fake_images = generator(noise_inp)
            fake_inp = tf.concat([fake_images, labels], axis=-1)
            fake_output = discriminator(fake_inp)
            g_loss = loss_fn(tf.ones_like(fake_output), fake_output)

        g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")
            save_generated_images(epoch, generator)

def save_generated_images(epoch, generator, examples=16, dim=(4, 4), figsize=(6, 6)):
    noise = np.random.normal(0, 1, (examples, 100))

    labels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6])
    labels_arrs = [np.zeros(10) for label in labels]
    for i, label in enumerate(labels):
        labels_arrs[i][label] = 1
    inp = tf.concat([noise, labels_arrs], axis=-1)
    generated_images = generator.predict(inp)
    generated_images = 0.5 * generated_images + 0.5

    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(generated_images[cnt].reshape(28, 28), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.savefig(f"cGAN/gan_generated_image_epoch_{epoch}.png")
    plt.close()

train_cgan(generator, discriminator, epochs=50000, batch_size=64)
