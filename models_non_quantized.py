import tensorflow as tf
from tensorflow.keras.layers import (Input, Flatten, Dense, Activation,
                                     Conv2D, SeparableConv2D,
                                     AveragePooling2D)
from tensorflow.keras.models import Model

def var_network(var, hidden=10, output=2):
    var = Flatten()(var)

    # First Dense layer
    var = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh")(var)

    # Second Dense layer
    var = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh")(var)

    # Output layer
    return Dense(
        output,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    )(var)

def conv_network(var, n_filters=5, kernel_size=3):
    var = SeparableConv2D(
        n_filters, kernel_size,
        depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh")(var)

    var = Conv2D(
        n_filters, 1,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh")(var)
    return var

def CreateModel(shape, n_filters, pool_size):
    x_base = x_in = Input(shape)
    stack = conv_network(x_base)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size),
        strides=None,
        padding="valid",
        data_format=None,
    )(stack)
    stack = var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack)
    return model