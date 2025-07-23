import keras
from keras.layers import *
from keras.models import Sequential, Model
from keras.utils import Sequence
from qkeras import *

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def var_network(var, hidden=10, output=2):
    var = Flatten(name="flatten")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="dense_1"
    )(var)
    #var = keras.activations.tanh(var)
    var = QActivation("quantized_tanh(8, 0, 1)", name="activation_tanh_2")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="dense_2"
    )(var)
    #var = keras.activations.tanh(var)
    var = QActivation("quantized_tanh(8, 0, 1)", name="activation_tanh_3")(var)
    return QDense(
        output,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        name="dense_3"
    )(var)

def mlp_encoder_network(var, hidden=16, hidden_dimx=16, hidden_dimy=16):
    proj_x = AveragePooling2D(
        pool_size=(1, 21), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_x = Flatten()(proj_x)

    proj_y = AveragePooling2D(
        pool_size=(13, 1), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_y = Flatten()(proj_y)

    proj_x = QDense(
        hidden_dimx,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_x)
    proj_x = QActivation("quantized_relu(bits=13, integer=5)(x)")(var)

    proj_y = QDense(
        hidden_dimy,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_y)
    proj_y = QActivation("quantized_relu(bits=13, integer=5)(x)")(var)

    var = Concatenate(axis=1)([proj_x, proj_y])

    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)

    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    return var

def CreateModel(shape):
    x_base = x_in = Input(shape, name="input_pxls/")
    stack = mlp_encoder_network(x_base)
    stack = var_network(stack, hidden=16, output=3) # this network should only be used with 'slim' (3) or 'diagonal' (8) regression targets
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model
