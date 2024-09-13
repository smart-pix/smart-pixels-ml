import shutil
from pathlib import Path
import tensorflow as tf
import keras
import numpy as np

def safe_remove_directory(directory_path):
    if Path(directory_path).exists():
        print(f"Directory {directory_path} is removed...")
        shutil.rmtree(directory_path)
    else:
        print(f"Directory {directory_path} does not exist and cannot be removed.")

def check_GPU():
    # set gpu growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    else:
        print("No GPU(s)")

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
