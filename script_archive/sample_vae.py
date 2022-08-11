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
from typing import Dict, List
from sklearn.metrics import r2_score

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
            total_loss = reconstruction_loss + 0.0001 * kl_loss
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

    parser.add_argument("-d", "--data", action="store", nargs='+', required=True,
                        help="Files used to train the VAE")
    parser.add_argument("-lt", "--latent_space", type=int, action="store", required=True,
                        help="Defines the latent space dimensions")
    parser.add_argument("-s", "--scaling", action="store", required=False,
                        help="Which type of scaling should be used", choices=["n", "s"], default="s")
    parser.add_argument("-p", "--prefix", action="store", required=True, type=str,
                        help="The prefix for creating the results folder")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    latent_dim = int(args.latent_space)
    data_paths = args.data

    base_path = f"{args.prefix}_{base_path}"
    FolderManagement.create_directory(path=Path(base_path))

    raw_data: Dict = {}
    for file in data_paths:
        file = Path(file)
        raw_data[file.stem] = pd.read_csv(file, sep='\t', index_col=0)

    train_data: Dict = {}
    test_data: Dict = {}

    # Create test and train data for each file
    for file, loaded_data in raw_data.items():
        train_data[file] = loaded_data.sample(frac=0.8, random_state=1).copy()
        test_data[file] = loaded_data.loc[~loaded_data.index.isin(train_data[file].index)].copy()

        pd.Series(test_data[file].index).to_csv(f'{base_path}/{file}_test_indices.csv')
        pd.Series(train_data[file].index).to_csv(f'{base_path}/{file}_train_indices.csv')

    train_set: pd.DataFrame = pd.concat([train for train in train_data.values()])
    test_set: pd.DataFrame = pd.concat([test for test in test_data.values()])

    # Reshuffle rows
    train_set, val_set = SplitHandler.create_train_val_split(input_data=train_set)
    test_set = test_set.sample(frac=1)

    # print(test_set.shape)
    # print(train_set.shape)
    # print(val_set.shape)

    if args.scaling == "s":
        train_set, scaler = Scaling.standardize(data=train_set, features=list(train_set.columns))
        test_set, _ = Scaling.standardize(data=test_set, features=list(test_set.columns), scaler=scaler)

    else:
        train_set, scaler = Scaling.normalize(data=train_set, features=list(train_set.columns))
        test_set, _ = Scaling.normalize(data=test_set, features=list(test_set.columns), scaler=scaler)

    # print(train_set.isna().sum())
    assert not np.any(np.isnan(train_set))
    assert not np.any(np.isnan(val_set))
    assert not np.any(np.isnan(test_set))

    # VAE

    input_dimensions = train_set.shape[1]

    initializer = tf.keras.initializers.Zeros()

    encoder_inputs = keras.Input(shape=(input_dimensions,))
    x = layers.Dense(units=input_dimensions / 2, activation="relu")(encoder_inputs)
    x = layers.Dense(units=input_dimensions / 3, activation="relu")(x)
    # x = layers.Dense(units=input_dimensions / 4, activation="relu")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    latent_inputs = keras.Input(shape=(latent_dim,))
    # x = layers.Dense(units=input_dimensions / 4, activation="relu")(latent_inputs)
    x = layers.Dense(units=input_dimensions / 3, activation="relu")(latent_inputs)
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

    history = vae.fit(train_set,
                      callbacks=callbacks,
                      validation_data=(val_set, val_set),
                      epochs=500, batch_size=32)

    # Save it under the form of a json file
    json.dump(history.history, open(Path(base_path, "history.json"), 'w'))
    vae.save(Path(base_path, f'{args.prefix}_model'))

    z_mean, z_var, embedding = vae.encoder.predict(train_set)
    reconstructed = pd.DataFrame(vae.decoder.predict(embedding), columns=train_set.columns)

    r2_scores: List = []
    for column in test_set.columns:
        r2_scores.append(r2_score(train_set[column], reconstructed[column]))

    scores = pd.DataFrame(r2_scores)
    scores.rename(columns={0: "Score"}, inplace=True)
    scores["Feature"] = test_set.columns
    scores.to_csv(f"{base_path}/reconstruction_r2_scores.csv", index=False)
