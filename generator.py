import tensorflow as tf
from tensorflow import keras
import numpy as np


class Generator(keras.Model):

    def __init__(self, codings_size=128):
        super().__init__(name='generator')

        self.input_layer = keras.layers.Dense(codings_size, activation='relu',
                                              kernel_initializer=keras.initializers.he_normal())
        self.dense_1 = keras.layers.Dense(codings_size * 5, activation='relu',
                                          kernel_initializer=keras.initializers.he_normal())
        self.dense_2 = keras.layers.Dense(28 * 28 * 5, activation='softmax')

    @tf.function
    def call(self, input_tensor):
        x = self.input_layer(input_tensor)
        x = self.dense_1(x)
        x = self.dense_2(x)

        return x
