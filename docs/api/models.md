# Models Module

The `models.py` file defines **Mixture Density Network (MDN)** network with a **4D Multivariate Normal Distribution** neural network architecture using quantized layers. The implementation uses QKeras to quantize the weights and activations of the network.

## Functions

### `CreateModel(shape, n_filters, pool_size)`
Creates a quantized neural network model for regression task with quantized layers and activations as in [Model](../Images/ML_model_arch.png). The model has `14` output nodes with `4` being the target variables and the rest `10` being the co-variances.

- **Arguments**:
  - `shape` (tuple): Input shape (e.g., `(13, 21, 2)`/ `(13, 21, 20)`).
  - `n_filters` (int): Number of filters for the convolutional layers.
  - `pool_size` (int): Size of the pool for the pooling layer.
- **Returns**:
  - `keras.Model`: A compiled Keras model instance.
- **Example**:
 ```python
  from models import CreateModel

  model = CreateModel((13, 21, 2), n_filters=5, pool_size=3)
  model.summary()

---

### Additional Helper Functions

## `conv_network(var, n_filters=5, kernel_size=3)`
Defines the convolutional network block, with quantized layers and activations.

- **Arguments**:
  - `var (InputLayer: tf.Tensor)`: Input tensor.
  - `n_filters (int)`: Number of filters.
  - `kernel_size (int)`: Kernel size.

- **Returns**:
  - `tf.Tensor`: Output tensor.

## `var_network(var, hidden=10, output=2)`
Defines the dense network block, with quantized layers and activations.

- **Arguments**:
  - `var (InputLayer: tf.Tensor)`: Input tensor.
  - `hidden (int)`: Number of hidden units.
  - `output (int)`: Number of output units.

- **Returns**:
  - `tf.Tensor`: Output tensor.
