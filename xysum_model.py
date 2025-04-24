import keras
from keras.layers import *
from keras.models import Sequential, Model
from keras.utils import Sequence
from qkeras import *

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def var_network(var, hidden=10, output=2):
    var = Flatten()(var)
    var = Dense(
        hidden,
        #kernel_quantizer=quantized_bits(8, 0, alpha=1),
        #bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = keras.activations.tanh(var) # QActivation("quantized_tanh(8, 0, 1)")(var)
    var = Dense(
        hidden,
        #kernel_quantizer=quantized_bits(8, 0, alpha=1),
        #bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = keras.activations.tanh(var) # QActivation("quantized_tanh(8, 0, 1)")(var)
    return Dense(
        output,
        #kernel_quantizer=quantized_bits(8, 0, alpha=1),
        #bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    )(var)

def conv_network(var, kernel_size=3):
    proj_x = AveragePooling2D(
        pool_size=(1, 21), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_x = Reshape((13, 20))(proj_x)
    proj_y = AveragePooling2D(
        pool_size=(13, 1), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_y = Reshape((21, 20))(proj_y)
    
    proj_x = Conv1D(
        5,kernel_size,
        #depthwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        #pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        #bias_quantizer=quantized_bits(4, 0, alpha=1),
        #depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        #pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_x)

    proj_y = Conv1D(
        5,kernel_size,
        #depthwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        #pointwise_quantizer=quantized_bits(4, 0, 1, alpha=1),
        #bias_quantizer=quantized_bits(4, 0, alpha=1),
        #depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        #pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_y)

    var = Concatenate(axis=1)([proj_x, proj_y])
    
    var = keras.activations.tanh(var) #QActivation("quantized_tanh(4, 0, 1)")(var)
 
    return var

def CreateModel(shape):
    x_base = x_in = Input(shape)
    stack = conv_network(x_base)
    #stack = AveragePooling2D(
    #    pool_size=(2, 2), 
    #    strides=None, 
    #    padding="valid", 
    #    data_format=None,        
    #)(stack)
    #stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack)
    return model
