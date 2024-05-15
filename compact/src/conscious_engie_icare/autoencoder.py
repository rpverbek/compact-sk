import pandas as pd 
import numpy as np
import tensorflow as tf


# data dimensions // hyperparameters 
input_dim = X_train_transformed.shape[1]
BATCH_SIZE = 256
EPOCHS = 100


class Autoencoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder = tf.keras.models.Sequential([
            # deconstruct / encode
            tf.keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim,)), 
            tf.keras.layers.Dense(16, activation='elu'),
            tf.keras.layers.Dense(8, activation='elu'),
            tf.keras.layers.Dense(4, activation='elu'),
            tf.keras.layers.Dense(2, activation='elu'),
            # reconstruction / decode
            tf.keras.layers.Dense(4, activation='elu'),
            tf.keras.layers.Dense(8, activation='elu'),
            tf.keras.layers.Dense(16, activation='elu'),
            tf.keras.layers.Dense(input_dim, activation='elu')
        ])

    def compile(self):
        # https://keras.io/api/models/model_training_apis/
        self.autoencoder.compile(optimizer="adam",
                                 loss="mse",
                                 metrics=["acc"])
