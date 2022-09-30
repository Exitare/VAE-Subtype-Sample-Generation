import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import os
from pathlib import Path
import json
from tensorflow.keras.models import Model

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        
        self.encoder: Model = encoder
        self.decoder: Model = decoder
        
        self.total_loss_tracker = keras.metrics.Mean(name = "total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name = "reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name = "kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss_fn = keras.losses.MeanSquaredError()
            reconstruction_loss = reconstruction_loss_fn(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
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

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 10
base_path = 'rvrs_out'

X, y = make_classification(n_samples=1000,
                           n_features=100,
                           n_informative=10,
                           n_redundant=90,
                           random_state=1,
                           shift = 30)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(X)
X = scaler.transform(X)
data = X

data = np.trunc(1000 * data) / 1000

X_train, X_val = train_test_split(data, test_size=0.2)

train_data = X_train
val_data = X_val

input_dimensions = train_data.shape[1]

encoder_inputs = keras.Input(shape=(input_dimensions,))
x = layers.Dense(units=input_dimensions / 2, activation="relu")(encoder_inputs)
x = layers.Dense(units=input_dimensions / 3, activation="relu")(x)
x = layers.Dense(units=input_dimensions / 4, activation="relu")(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(units=input_dimensions / 4, activation="relu")(latent_inputs)
x = layers.Dense(units=input_dimensions / 3, activation="relu")(x)
x = layers.Dense(units=input_dimensions / 2, activation="relu")(x)

decoder_outputs = layers.Dense(units=input_dimensions, activation="relu")(x)

decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

vae: VAE = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

callbacks = []
early_stop = EarlyStopping(monitor="reconstruction_loss",
                           mode="min", patience=5,
                           restore_best_weights=True)

callbacks.append(early_stop)
csv_logger = CSVLogger(os.path.join(base_path, 'training.log'), # Is this writing a file?
                       separator='\t')
callbacks.append(csv_logger)
history = vae.fit(train_data, # Fit the VAE, work backward to carve out minimum viable code path to run this function
                  callbacks=callbacks,
                  validation_data=(val_data, val_data))
z_mean, z_var, embedding = vae.encoder.predict(test_data)
embedding = pd.DataFrame(embedding)