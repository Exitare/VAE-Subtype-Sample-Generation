import argparse
import tensorflow as tf
from library.preprocessing.scaling import Scaling
import pandas as pd


def get_args():
    """
       Load all provided cli args
       """
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", action="store", required=True,
                        help="The VAE model to load")
    parser.add_argument("-d", "--data", action="store", required=True,
                        help="The data to create the embeddings for data generation")
    parser.add_argument("-p", "--prefix", action="store", required=True, type=str,
                        help="The prefix for creating the results folder")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    model_path = args.model
    data = pd.read_csv(args.data)

    data, scaler = Scaling.normalize(data=data, features=list(data.columns))

    vae = tf.keras.models.load_model(model_path)

    embeddings, z_mean, z_var = vae.encoder.predict(data)


