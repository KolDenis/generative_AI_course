from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image

# Параметры
img_height = 16
img_width = 16
batch_size = 64


# Функция для загрузки изображений по мере надобности
def load_image_batch(damaged_folder, clean_folder, batch_size, img_height, img_width):
    # Получаем список файлов в папках
    damaged_files = os.listdir(damaged_folder)
    clean_files = os.listdir(clean_folder)

    while True:
        idx = np.random.randint(0, 10000, batch_size)

        damaged_images = []
        clean_images = []

        for i in idx:
            # Загрузка поврежденного изображения
            damaged_img = image.load_img(os.path.join(damaged_folder, damaged_files[i]),
                                         target_size=(img_height, img_width))
            damaged_img = image.img_to_array(damaged_img)
            damaged_images.append(damaged_img)

            # Загрузка чистого (неповрежденного) изображения
            clean_img = image.load_img(os.path.join(clean_folder, clean_files[i]), target_size=(img_height*4, img_width*4))
            clean_img = image.img_to_array(clean_img)
            clean_images.append(clean_img)

        # Нормализуем изображения
        damaged_images = np.array(damaged_images).astype(np.float32)
        clean_images = np.array(clean_images).astype(np.float32)

        damaged_images = (damaged_images - 127.5) / 127.5
        clean_images = (clean_images - 127.5) / 127.5

        yield damaged_images, clean_images


# Генератор для загрузки данных
damaged_folder = 'celeba_2_16'
clean_folder = 'celeba_2'

train_gen = load_image_batch(damaged_folder, clean_folder, batch_size, img_height, img_width)

#res
def build_generator():
    model = Sequential([
        Input((img_height, img_width, 3)),

        Conv2DTranspose(256, kernel_size=6, strides=2, padding='same'), #6
        BatchNormalization(),
        LeakyReLU(negative_slope=0.2),

        Conv2DTranspose(192, kernel_size=6, strides=1, padding='same'), #6
        BatchNormalization(),
        LeakyReLU(negative_slope=0.2),

        Conv2DTranspose(128, kernel_size=6, strides=2, padding='same'), #6
        BatchNormalization(),
        LeakyReLU(negative_slope=0.2),

        Conv2D(3, kernel_size=7, activation='tanh', padding='same')
    ])
    return model

def build_discriminator():
    model = Sequential([
        Input((img_height * 4, img_width * 4, 3)),

        Conv2D(128, kernel_size=5, strides=2, padding='same'), #5
        BatchNormalization(),
        LeakyReLU(negative_slope=0.2),

        Conv2D(192, kernel_size=5, strides=2, padding='same'), #5
        BatchNormalization(),
        LeakyReLU(negative_slope=0.2),

        Conv2D(256, kernel_size=5, strides=2, padding='same'), #5
        BatchNormalization(),
        LeakyReLU(negative_slope=0.2),

        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

init_gen_lr = 0.0002
init_disc_lr = 0.0001

# Определение оптимизаторов и потерь
g_optimizer = Adam(learning_rate=init_gen_lr, beta_1=0.5)
d_optimizer = Adam(learning_rate=init_disc_lr, beta_1=0.5)
loss_bc = tf.keras.losses.BinaryCrossentropy()
loss_mae = tf.keras.losses.MeanAbsoluteError()

# Построение модели
generator = build_generator()
discriminator = build_discriminator()

def lr_schedule(epoch, md):
    if md == "g":
        return init_gen_lr * tf.math.exp(-init_gen_lr * epoch)
    elif md == "d":
        return init_disc_lr * tf.math.exp(-init_disc_lr * epoch)

def train_gan(generator, discriminator, train_gen, epochs=10000, steps_per_epoch=100):
    for epoch in range(epochs):
        g_optimizer.learning_rate.assign(lr_schedule(epoch, "g"))
        d_optimizer.learning_rate.assign(lr_schedule(epoch, "d"))

        for step in range(steps_per_epoch):
            # Получаем следующий batch данных из генератора
            damaged_images, clean_images = next(train_gen)
            fake_images = generator(damaged_images, training=True)

            with tf.GradientTape() as d_tape:
                real_output = discriminator(clean_images)
                fake_output = discriminator(fake_images)
                d_loss_real = loss_bc(tf.ones_like(real_output), real_output)
                d_loss_fake = loss_bc(tf.zeros_like(fake_output), fake_output)
                d_loss = (d_loss_real + d_loss_fake) / 2

            d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

            with tf.GradientTape() as g_tape:
                fake_images = generator(damaged_images)
                fake_output = discriminator(fake_images)
                g_loss = loss_bc(tf.ones_like(fake_output), fake_output)
                content = loss_mae(clean_images, fake_images)
                g_loss = g_loss + 0.01 * content

            g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

        print(f"Epoch {epoch}, Step {step}, D Loss: {d_loss}, G Loss: {g_loss}")
        save_generated_images(epoch, generator, damaged_images, clean_images)

def save_generated_images(epoch, generator, damaged_images, clean_images, dim=(3, 4), figsize=(6, 3)):
    generated_images = generator.predict(damaged_images)

    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    for i in range(dim[0]):
        cnt = 0
        for j in range(dim[1]):
            if i == 0:
                axs[i, j].imshow(clean_images[cnt]*0.5 + 0.5)
                axs[i, j].axis('off')
                cnt += 1
            elif i == 1:
                axs[i, j].imshow(damaged_images[cnt]*0.5 + 0.5)
                axs[i, j].axis('off')
                cnt += 1
            else:
                axs[i, j].imshow(generated_images[cnt]*0.5 + 0.5)
                axs[i, j].axis('off')
                cnt += 1
    plt.savefig(f"res2/epoch_{epoch}.png")
    plt.close()


train_gan(generator, discriminator, train_gen, epochs=10000, steps_per_epoch=10)


