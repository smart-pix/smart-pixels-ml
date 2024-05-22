import tensorflow as tf
import keras
import numpy as np

def data_prep_quantizer(data, bits=3, int_bits=0): # remember there's a secret sign bit
    frac_bits = bits - int_bits
    return np.round(data * 2**frac_bits) * 2**-frac_bits

def diffable_quantizer(data, bits=7, int_bits=0): # remember there's a secret sign bit
    frac_bits = bits - int_bits
    return tf.math.round(data * 2**frac_bits) * 2**-frac_bits

class LearnedScale(keras.layers.Layer):
    def __init__(self, input_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.scale = self.add_weight(
            shape=(self.input_dim, ), initializer="glorot_uniform", trainable=True
        )
        #self.shift = self.add_weight(shape=(input_dim, ), initializer="zeros", trainable=True)

    def call(self, inputs):
        return inputs * tf.math.softplus(self.scale) # + self.shift
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim
        })
        return config