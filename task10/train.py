import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
from tensorflow.keras.losses import MeanSquaredError, KLDivergence, binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

latent_dim = 1024
image_shape = x_train[0].shape

mse_loss = MeanSquaredError()
kl_loss = KLDivergence()

class VAE(keras.Model):
    def __init__(self, latent_dim, image_shape, beta):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.beta = beta
        # encoder
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=image_shape),

            layers.Conv2D(64, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            layers.Conv2D(128, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            layers.Conv2D(256, 3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            layers.Conv2D(512, 3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            layers.Flatten(),
        ])

        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(8 * 8 * 256, activation='relu'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            layers.Reshape((8, 8, 256)),
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),

            layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='sigmoid'),
        ])

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def encode(self, data):
        x = self.encoder(data)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        return z_mean, z_log_var

    def reparameterization(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_recon = self.decode(z)
        return x_recon, mean, logvar

    def decode(self, data):
        return self.decoder(data)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def recon_loss(self, data, reconstruction):
        return tf.reduce_mean(binary_crossentropy(data, reconstruction))

    def kl_divergence(self, Z_logvar, Z_mu):
        kl = -0.5 * tf.reduce_mean(1 + Z_logvar - Z_mu ** 2 - tf.math.exp(Z_logvar))
        return self.beta * kl

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(data)
            z = self.reparameterization(z_mean, z_log_var)
            reconstruction = self.decode(z)
            reconstruction_loss = self.recon_loss(data, reconstruction)
            kl_loss = self.kl_divergence(z_log_var, z_mean)
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

vae = VAE(latent_dim,  image_shape, 0.3)
vae.decoder.summary()

vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
history = vae.fit(x_train, epochs=50, batch_size=128) ##callbacks=[early_stopping]

num_images = 5

def generate_images_from_latent_vectors(vae, num_images_to_generate):
    random_latent_vectors = np.random.normal(size=(num_images_to_generate, latent_dim))
    generated_images = vae.decoder.predict(random_latent_vectors)
    return generated_images

generated_images = generate_images_from_latent_vectors(vae, num_images)

def plot_real_and_generated_images(real_images, generated_images, num_images_to_generate):
    plt.figure(figsize=(20, 10))

    for i in range(num_images_to_generate):
        plt.subplot(2, num_images_to_generate, i + 1)
        plt.imshow(real_images[i])
        plt.title('Real Image')
        plt.axis('off')

    for i in range(num_images_to_generate):
        plt.subplot(2, num_images_to_generate, num_images_to_generate + i + 1)
        plt.imshow(generated_images[i])
        plt.title('Generated Image')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"res/epoch_{"last5"}.png")
    plt.close()

plot_real_and_generated_images(x_test[:num_images], generated_images, num_images)