import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train.reshape(-1, 28 * 28)

def build_generator(latent_dim):
    generator = Sequential([
        Dense(64, activation='relu', input_dim=latent_dim),
        Dense(128, activation='relu'),
        Dense(28 * 28, activation='tanh')
    ])
    return generator

def build_discriminator(img_dim):
    discriminator = Sequential([
        Dense(128, input_dim=img_dim),
        LeakyReLU(alpha=0.2),
        Dense(64),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    return discriminator

latent_dim = 100
generator = build_generator(latent_dim)
discriminator = build_discriminator(28 * 28)

g_optimizer = Adam(learning_rate=0.0005, beta_1=0.5)
d_optimizer = Adam(learning_rate=0.0005, beta_1=0.5)
loss_fn = tf.keras.losses.BinaryCrossentropy()

def train_gan(generator, discriminator, epochs=10000, batch_size=64):
    for epoch in range(epochs):
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)

        with tf.GradientTape() as d_tape:
            real_output = discriminator(real_images)
            fake_output = discriminator(fake_images)
            d_loss_real = loss_fn(tf.ones_like(real_output), real_output)
            d_loss_fake = loss_fn(tf.zeros_like(fake_output), fake_output)
            d_loss = d_loss_real + d_loss_fake

        d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            fake_images = generator(noise)
            fake_output = discriminator(fake_images)
            g_loss = loss_fn(tf.ones_like(fake_output), fake_output)

        g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")
            save_generated_images(epoch, generator)

# Функция для сохранения изображений, сгенерированных генератором
def save_generated_images(epoch, generator, examples=16, dim=(4, 4), figsize=(6, 6)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Преобразование из [-1, 1] в [0, 1]

    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(generated_images[cnt].reshape(28, 28), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.close()

# Тренировка GAN
train_gan(generator, discriminator, epochs=50000, batch_size=64)
