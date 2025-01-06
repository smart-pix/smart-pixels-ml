# Loss Module

This module uses a **Mixture Density Network(MDN)** and hence a negative log-likelihood loss function. It uses TensorFlow and TensorFlow Probability for the loss computation.

---

## **Loss Function**

### **`custom_loss(y, p_base, minval=1e-9, maxval=1e9, scale=512)`**
Calculates the **Negative Log-Likelihood (NLL)** for a batch of data using the model's predicted parameters.

---
### **Brief Description**
The model parameters are the mean and the lower triangular part of the covariance matrix.

loss function is vectorized with batches.


### **Arguments**

- `y` (tf.Tensor): The target data, shape (batch_size, 4).
- `p_base` (tf.Tensor): The predicted parameters, shape (batch_size, 16).
- `minval` (float): The minimum value for the likelihood, default 1e-9.
- `maxval` (float): The maximum value for the likelihood, default 1e9.
- `scale` (float): .

### **Returns**

- `tf.Tensor`: Negative Log-Likelihood (NLL) for the given batch., shape (batch_size,).

### **Example Usage**

```python
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from loss import custom_loss
from models import CreateModel

model = CreateModel((13, 21, 2), n_filters=5, pool_size=3)
model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss)
```