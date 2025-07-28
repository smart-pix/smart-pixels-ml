import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from qkeras import *
import numpy as np

def CreateSlim1DModel(input_shape=(16, 16, 2), n_filters=5, model_name="smrtpxl_slim"):
    """
    Create slim 1D model with Conv1D projections and only 3 outputs: x, y, cotBeta
    
    Args:
        input_shape: Input shape (height, width, channels), default (16, 16, 2)
        n_filters: Number of filters for Conv1D layers, default 5
        model_name: Name for the model
    
    Returns:
        Keras Model with 3 outputs
    """
    
    # Input layer
    input_layer = Input(shape=input_shape, name="input_pxls/")
    
    # Projection layers - average pooling to reduce 2D to 1D
    avg_pooling_2d_proj_x = AveragePooling2D(
        pool_size=(1, input_shape[1]), 
        name="avg_pooling_2d_proj_x"
    )(input_layer)
    
    avg_pooling_2d_proj_y = AveragePooling2D(
        pool_size=(input_shape[0], 1), 
        name="avg_pooling_2d_proj_y"
    )(input_layer)
    
    # Reshape to 1D for Conv1D processing
    reshape_proj_x = Reshape(
        (input_shape[0], input_shape[2]), 
        name="reshape_proj_x"
    )(avg_pooling_2d_proj_x)
    
    reshape_proj_y = Reshape(
        (input_shape[1], input_shape[2]), 
        name="reshape_proj_y"
    )(avg_pooling_2d_proj_y)
    
    # Conv1D layers for each projection
    conv1d_proj_x = QConv1D(
        n_filters, 
        3, 
        kernel_quantizer=quantized_bits(4, 0, alpha=1), 
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        name="conv1d_proj_x"
    )(reshape_proj_x)
    
    conv1d_proj_y = QConv1D(
        n_filters, 
        3, 
        kernel_quantizer=quantized_bits(4, 0, alpha=1), 
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        name="conv1d_proj_y"
    )(reshape_proj_y)
    
    # Concatenate the two Conv1D outputs
    concatenate_layer = Concatenate(axis=1, name="concatenate")([conv1d_proj_x, conv1d_proj_y])
    
    # First activation
    activation_tanh_1 = QActivation("tanh", name="activation_tanh_1")(concatenate_layer)
    
    # Flatten for dense layers
    flatten_layer = Flatten(name="flatten")(activation_tanh_1)
    
    # First dense layer
    dense_1 = QDense(
        16, 
        kernel_quantizer=quantized_bits(4, 0, alpha=1), 
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        name="dense_1"
    )(flatten_layer)
    activation_tanh_2 = QActivation("tanh", name="activation_tanh_2")(dense_1)
    
    # Second dense layer
    dense_2 = QDense(
        16, 
        kernel_quantizer=quantized_bits(4, 0, alpha=1), 
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        name="dense_2"
    )(activation_tanh_2)
    activation_tanh_3 = QActivation("tanh", name="activation_tanh_3")(dense_2)
    
    # Output layer - 3 outputs for slim model: x, y, cotBeta
    dense_3 = QDense(
        3,  # Changed from 14 to 3 for slim model
        kernel_quantizer=quantized_bits(4, 0, alpha=1), 
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        name="dense_3"
    )(activation_tanh_3)
    
    # Create model
    model = Model(inputs=input_layer, outputs=dense_3, name=model_name)
    
    return model

def CreateSlim1DModelWithDropout(input_shape=(16, 16, 2), n_filters=5, dropout_rate=0.1, model_name="smrtpxl_slim_dropout"):
    """
    Create slim 1D model with dropout for regularization
    
    Args:
        input_shape: Input shape (height, width, channels), default (16, 16, 2)
        n_filters: Number of filters for Conv1D layers, default 5
        dropout_rate: Dropout rate for regularization, default 0.1
        model_name: Name for the model
    
    Returns:
        Keras Model with 3 outputs and dropout layers
    """
    
    # Input layer
    input_layer = Input(shape=input_shape, name="input_pxls/")
    
    # Projection layers
    avg_pooling_2d_proj_x = AveragePooling2D(
        pool_size=(1, input_shape[1]), 
        name="avg_pooling_2d_proj_x"
    )(input_layer)
    
    avg_pooling_2d_proj_y = AveragePooling2D(
        pool_size=(input_shape[0], 1), 
        name="avg_pooling_2d_proj_y"
    )(input_layer)
    
    # Reshape
    reshape_proj_x = Reshape(
        (input_shape[0], input_shape[2]), 
        name="reshape_proj_x"
    )(avg_pooling_2d_proj_x)
    
    reshape_proj_y = Reshape(
        (input_shape[1], input_shape[2]), 
        name="reshape_proj_y"
    )(avg_pooling_2d_proj_y)
    
    # Conv1D layers
    conv1d_proj_x = QConv1D(
        n_filters, 
        3, 
        kernel_quantizer=quantized_bits(4, 0, alpha=1), 
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        name="conv1d_proj_x"
    )(reshape_proj_x)
    
    conv1d_proj_y = QConv1D(
        n_filters, 
        3, 
        kernel_quantizer=quantized_bits(4, 0, alpha=1), 
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        name="conv1d_proj_y"
    )(reshape_proj_y)
    
    # Concatenate and activation
    concatenate_layer = Concatenate(axis=1, name="concatenate")([conv1d_proj_x, conv1d_proj_y])
    activation_tanh_1 = QActivation("tanh", name="activation_tanh_1")(concatenate_layer)
    
    # Flatten
    flatten_layer = Flatten(name="flatten")(activation_tanh_1)
    
    # Dense layers with dropout
    dense_1 = QDense(
        16, 
        kernel_quantizer=quantized_bits(4, 0, alpha=1), 
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        name="dense_1"
    )(flatten_layer)
    activation_tanh_2 = QActivation("tanh", name="activation_tanh_2")(dense_1)
    dropout_1 = Dropout(dropout_rate, name="dropout_1")(activation_tanh_2)
    
    dense_2 = QDense(
        16, 
        kernel_quantizer=quantized_bits(4, 0, alpha=1), 
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        name="dense_2"
    )(dropout_1)
    activation_tanh_3 = QActivation("tanh", name="activation_tanh_3")(dense_2)
    dropout_2 = Dropout(dropout_rate, name="dropout_2")(activation_tanh_3)
    
    # Output layer
    dense_3 = QDense(
        3,  # 3 outputs for slim model
        kernel_quantizer=quantized_bits(4, 0, alpha=1), 
        bias_quantizer=quantized_bits(4, 0, alpha=1),
        name="dense_3"
    )(dropout_2)
    
    # Create model
    model = Model(inputs=input_layer, outputs=dense_3, name=model_name)
    
    return model

def CreateSlim1DModelCustomizable(input_shape=(16, 16, 2), 
                                 n_filters=5, 
                                 dense_units=[16, 16], 
                                 quantization_bits=4,
                                 model_name="smrtpxl_slim_custom"):
    """
    Create customizable slim 1D model
    
    Args:
        input_shape: Input shape (height, width, channels)
        n_filters: Number of filters for Conv1D layers
        dense_units: List of units for dense layers
        quantization_bits: Number of bits for quantization
        model_name: Name for the model
    
    Returns:
        Keras Model with customizable architecture
    """
    
    # Input layer
    input_layer = Input(shape=input_shape, name="input_pxls/")
    
    # Projection layers
    avg_pooling_2d_proj_x = AveragePooling2D(
        pool_size=(1, input_shape[1]), 
        name="avg_pooling_2d_proj_x"
    )(input_layer)
    
    avg_pooling_2d_proj_y = AveragePooling2D(
        pool_size=(input_shape[0], 1), 
        name="avg_pooling_2d_proj_y"
    )(input_layer)
    
    # Reshape
    reshape_proj_x = Reshape(
        (input_shape[0], input_shape[2]), 
        name="reshape_proj_x"
    )(avg_pooling_2d_proj_x)
    
    reshape_proj_y = Reshape(
        (input_shape[1], input_shape[2]), 
        name="reshape_proj_y"
    )(avg_pooling_2d_proj_y)
    
    # Conv1D layers
    conv1d_proj_x = QConv1D(
        n_filters, 
        3, 
        kernel_quantizer=quantized_bits(quantization_bits, 0, alpha=1), 
        bias_quantizer=quantized_bits(quantization_bits, 0, alpha=1),
        name="conv1d_proj_x"
    )(reshape_proj_x)
    
    conv1d_proj_y = QConv1D(
        n_filters, 
        3, 
        kernel_quantizer=quantized_bits(quantization_bits, 0, alpha=1), 
        bias_quantizer=quantized_bits(quantization_bits, 0, alpha=1),
        name="conv1d_proj_y"
    )(reshape_proj_y)
    
    # Concatenate and activation
    concatenate_layer = Concatenate(axis=1, name="concatenate")([conv1d_proj_x, conv1d_proj_y])
    x = QActivation("tanh", name="activation_tanh_1")(concatenate_layer)
    
    # Flatten
    x = Flatten(name="flatten")(x)
    
    # Dense layers
    for i, units in enumerate(dense_units):
        x = QDense(
            units, 
            kernel_quantizer=quantized_bits(quantization_bits, 0, alpha=1), 
            bias_quantizer=quantized_bits(quantization_bits, 0, alpha=1),
            name=f"dense_{i+1}"
        )(x)
        x = QActivation("tanh", name=f"activation_tanh_{i+2}")(x)
    
    # Output layer
    outputs = QDense(
        3,  # 3 outputs for slim model
        kernel_quantizer=quantized_bits(quantization_bits, 0, alpha=1), 
        bias_quantizer=quantized_bits(quantization_bits, 0, alpha=1),
        name="dense_output"
    )(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=outputs, name=model_name)
    
    return model

# Helper function to get model summary
def print_model_info(model):
    """Print detailed model information"""
    print(f"Model name: {model.name}")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    model.summary()