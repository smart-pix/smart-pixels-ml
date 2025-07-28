import tensorflow as tf
import numpy as np

def slim_sse_loss(y_true, y_pred):
    """
    Sum of Squared Errors loss for slim model (no uncertainty quantification)
    
    Args:
        y_true: Ground truth values [batch_size, 3] for [x, y, cotBeta]
        y_pred: Predicted values [batch_size, 3] for [x, y, cotBeta]
    
    Returns:
        SSE loss value
    """
    squared_errors = tf.square(y_true - y_pred)
    sse = tf.reduce_sum(squared_errors)
    return sse

def slim_mse_loss(y_true, y_pred):
    """
    Mean Squared Errors loss for slim model (alternative option)
    
    Args:
        y_true: Ground truth values [batch_size, 3] for [x, y, cotBeta]
        y_pred: Predicted values [batch_size, 3] for [x, y, cotBeta]
    
    Returns:
        MSE loss value
    """
    squared_errors = tf.square(y_true - y_pred)
    mse = tf.reduce_mean(squared_errors)
    return mse

def slim_weighted_sse_loss(weights=None):
    """
    Weighted SSE loss for slim model with different weights for each output
    
    Args:
        weights: List or array of weights for [x, y, cotBeta]. If None, uses equal weights.
    
    Returns:
        Weighted SSE loss function
    """
    if weights is None:
        weights = [1.0, 1.0, 1.0]  # Equal weights
    
    weights_tensor = tf.constant(weights, dtype=tf.float32)
    
    def loss_fn(y_true, y_pred):
        squared_errors = tf.square(y_true - y_pred)
        weighted_errors = squared_errors * weights_tensor
        sse = tf.reduce_sum(weighted_errors)
        return sse
    
    return loss_fn

# Custom metrics for monitoring training
def slim_mae_metric(y_true, y_pred):
    """Mean Absolute Error metric for monitoring"""
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def slim_rmse_metric(y_true, y_pred):
    """Root Mean Squared Error metric for monitoring"""
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Individual parameter metrics
def x_mae_metric(y_true, y_pred):
    """MAE for x parameter only"""
    return tf.reduce_mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))

def y_mae_metric(y_true, y_pred):
    """MAE for y parameter only"""
    return tf.reduce_mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))

def cotBeta_mae_metric(y_true, y_pred):
    """MAE for cotBeta parameter only"""
    return tf.reduce_mean(tf.abs(y_true[:, 2] - y_pred[:, 2]))