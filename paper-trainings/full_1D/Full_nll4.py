'''
Arghya's diagonal loss
'''
import tensorflow as tf
import tensorflow_probability as tfp

# custom loss function foo diag model (with 8 outputs)
def custom_diag_loss(y, p_base, ):
    mu = p_base[:, 0:4]
    
    minval=1e-9
    maxval=1e9
    
    raw_diag = p_base[:, 4:8]
    Mdia  = tf.nn.softplus(raw_diag) + minval
    
    dist = tfp.distributions.MultivariateNormalDiag(
        loc=mu,
        scale_diag=Mdia
    )
    NLL = -dist.log_prob(y)

    return tf.reduce_sum(NLL) 