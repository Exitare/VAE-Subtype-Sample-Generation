import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import argparse
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import os
from pathlib import Path
import json
from tensorflow.keras.models import Model
# Own created methods
from library.preprocessing.scaling import Scaling
from library.preprocessing.splits import SplitHandler
from library.data.folder_management import FolderManagement

base_path = Path("results")

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder: Model = encoder
        self.decoder: Model = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

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


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data", action="store", required=False,
                        help="The file to use for coding genes")
    parser.add_argument("--train", action="store", required=False,
                        help="The normalized train data")
    parser.add_argument("--val", action="store", required=False,
                        help="The normalized val data")
    parser.add_argument("--test", action="store", required=False,
                        help="The normalized test data")
    parser.add_argument("-lt", "--latent_space", type=int, action="store", required=True,
                        help="Defines the latent space dimensions")
    parser.add_argument("-s", "--scaling", action="store", required=False,
                        help="Which type of scaling should be used", choices=["min", "s"], default="s")
    parser.add_argument("-p", "--prefix", action="store", required=True, type=str,
                        help="The prefix for creating the results folder")
    return parser.parse_args()


# Load args
args = get_args()
base_path = f"{args.prefix}_{base_path}"
FolderManagement.create_directory(path=Path(base_path))

latent_dim = args.latent_space

if args.data is not None:
    data = pd.read_csv(args.data, sep='\t', index_col=0)
    # truncate everything after the 4th decimal
    data = np.trunc(1000 * data) / 1000

    train_data, val_data, test_data = SplitHandler.create_splits(input_data=data, without_val=False)

    train_data, scaler = Scaling.normalize(train_data, features=train_data.columns)
    val_data, _ = Scaling.normalize(val_data, features=val_data.columns, scaler=scaler)
    test_data, _ = Scaling.normalize(data=test_data, features=test_data.columns, scaler=scaler)
else:
    train_data = pd.read_csv(args.train, sep='\t', index_col=0)
    val_data = pd.read_csv(args.val, sep='\t', index_col=0)
    test_data = pd.read_csv(args.test, sep='\t', index_col=0)

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

# vae.summary()

callbacks = []

early_stop = EarlyStopping(monitor="reconstruction_loss",
                           mode="min", patience=5,
                           restore_best_weights=True)
callbacks.append(early_stop)

csv_logger = CSVLogger(os.path.join(base_path, 'training.log'),
                       separator='\t')
callbacks.append(csv_logger)

history = vae.fit(train_data,
                  callbacks=callbacks,
                  validation_data=(val_data, val_data),
                  epochs=500, batch_size=128)

# Save it under the form of a json file
json.dump(history.history, open(Path(base_path, "history.json"), 'w'))
vae.save(Path(base_path, f'{args.prefix}_model'))

z_mean, z_var, embedding = vae.encoder.predict(test_data)
embedding = pd.DataFrame(embedding)
embedding.to_csv(Path(base_path, f"{args.prefix}_embeddings.csv"), index=False)