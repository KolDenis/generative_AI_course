import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train[..., np.newaxis]

def build_generator(latent_dim):
    model = Sequential([
        Dense(7 * 7 * 64, input_dim=latent_dim),
        Reshape((7, 7, 64)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(1, kernel_size=7, activation='tanh', padding='same')
    ])
    return model

def build_critic(img_shape):
    model = Sequential([
        Conv2D(128, kernel_size=4, strides=2, padding='same', input_shape=img_shape),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Conv2D(256, kernel_size=4, strides=2, padding='same'),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Flatten(),
        Dense(1)
    ])
    return model

latent_dim = 100
img_shape = (28, 28, 1)

generator = build_generator(latent_dim)
critic = build_critic(img_shape)

g_optimizer = RMSprop(learning_rate=0.0002)
c_optimizer = RMSprop(learning_rate=0.0002)

def clip_weights(model, clip_value=0.01):
    for layer in model.layers:
        if isinstance(layer, Dense) or isinstance(layer, Conv2D):
            weights = layer.get_weights()
            clipped_weights = [np.clip(w, -clip_value, clip_value) for w in weights]
            layer.set_weights(clipped_weights)

def train_wgan(generator, critic, epochs=10000, batch_size=64, n_critic=5):
    for epoch in range(epochs):
        for _ in range(n_critic):
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_images = x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_images = generator.predict(noise)

            with tf.GradientTape() as c_tape:
                real_output = critic(real_images)
                fake_output = critic(fake_images)
                c_loss = -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)

            c_gradients = c_tape.gradient(c_loss, critic.trainable_variables)
            c_optimizer.apply_gradients(zip(c_gradients, critic.trainable_variables))

            clip_weights(critic)

        with tf.GradientTape() as g_tape:
            fake_images = generator(noise)
            fake_output = critic(fake_images)
            g_loss = -tf.reduce_mean(fake_output)

        g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Critic Loss: {c_loss}, Generator Loss: {g_loss}")
            save_generated_images(epoch, generator)

def save_generated_images(epoch, generator, examples=16, dim=(4, 4), figsize=(6, 6)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(generated_images[cnt].reshape(28, 28), cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    plt.savefig(f"WGAN/wgan_generated_image_epoch_{epoch}.png")
    plt.close()

train_wgan(generator, critic, epochs=50000, batch_size=64)
