import tensorflow as tf
import tensorflow_probability as tfp

#------------------------------------------------------------
# I recommend placing the following functions on utils.py 
nll_tracker = tf.keras.metrics.Mean(name="nll")
reg_term_tracker = tf.keras.metrics.Mean(name="reg_term")

def track_loss_values(nll, weighted_reg_term):
    nll_tracker.update_state(nll)
    reg_term_tracker.update_state(weighted_reg_term)

# This function is useful if you are implementing an epoch logger
def reset_loss_trackers():
    nll_tracker.reset_states()
    reg_term_tracker.reset_states()

def get_loss_metrics():
    return {
        'nll': float(nll_tracker.result().numpy()),
        'reg_term': float(reg_term_tracker.result().numpy())
    }
#------------------------------------------------------------

# Custom loss: NLL + sum of standard deviations regularizer
current_reg_weight = tf.Variable(0.5, trainable=False, dtype=tf.float32, name='reg_weight')

def custom_loss(y, p_base, minval=1e-9, maxval=1e9):

    reg_weight = current_reg_weight
    
    p = p_base
    
    mu = p[:, 0:8:2]
    
    # creating each matrix element in 4x4
    Mdia = minval + tf.math.maximum(p[:, 1:8:2], 0.0)
    Mcov = p[:,8:]
    
    # placeholder zero element
    zeros = tf.zeros_like(Mdia[:,0])
    
    # assembles scale_tril matrix
    row1 = tf.stack([Mdia[:,0],zeros,zeros,zeros])
    row2 = tf.stack([Mcov[:,0],Mdia[:,1],zeros,zeros])
    row3 = tf.stack([Mcov[:,1],Mcov[:,2],Mdia[:,2],zeros])
    row4 = tf.stack([Mcov[:,3],Mcov[:,4],Mcov[:,5],Mdia[:,3]])

    scale_tril = tf.transpose(tf.stack([row1,row2,row3,row4]),perm=[2,0,1])

    dist = tfp.distributions.MultivariateNormalTriL(loc = mu, scale_tril = scale_tril) 
    
    likelihood = dist.prob(y)  
    likelihood = tf.clip_by_value(likelihood,minval,maxval)

    NLL = -1*tf.math.log(likelihood)

    cov_matrix = tf.matmul(scale_tril, tf.transpose(scale_tril, [0, 2, 1])) 
    variances = tf.linalg.diag_part(cov_matrix)
    stds = tf.sqrt(variances + minval)

    sigma_regularizer_1 = tf.reduce_sum(stds, axis=1)

    track_loss_values(NLL, reg_weight * sigma_regularizer_1)

    total_loss = NLL + (reg_weight * sigma_regularizer_1)
    
    return tf.keras.backend.sum(total_loss)