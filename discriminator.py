from tensorflow import keras
import tensorflow as tf


class Discriminator(keras.Model):

    def __init__(self):
        super().__init__(name='discriminator')
        self.flatten = keras.layers.Flatten()
        self.input_layer = keras.layers.Dense(28*28, activation='relu',
                                              kernel_initializer=keras.initializers.he_normal())
        self.dense_1 = keras.layers.Dense(128, activation='relu',
                                          kernel_initializer=keras.initializers.he_normal())
        self.dense_2 = keras.layers.Dense(1)

    @tf.function
    def call(self, input_tensor):
        x = self.flatten(input_tensor)
        x = self.input_layer(x)
        x = self.dense_1(x)
        x = self.dense_2(x)

        return x
