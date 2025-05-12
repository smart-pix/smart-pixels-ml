import tensorflow as tf
import tensorflow_probability as tfp

# custom loss function
@tf.function
def custom_loss(y, p_base, minval=1e-9, maxval=1e9):
    
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

    return tf.keras.backend.sum(NLL)

@tf.function
def empirical_kl(p_vals, q_vals, epsilon=1e-10):
    # p_vals: (dims, n_grid)
    # q_vals: (n_grid,)
    # Normalize p_vals per dim
    p_sum = tf.reduce_sum(p_vals, axis=1, keepdims=True)
    p_sum = tf.where(p_sum < epsilon, tf.ones_like(p_sum), p_sum)
    
    p_vals = p_vals / p_sum # (dims, n_grid)
    q_vals = q_vals / tf.reduce_sum(q_vals) # (n_grid,)
    
    # Clip everything to avoid log(0) or log(negative)
    safe_p = tf.clip_by_value(p_vals, epsilon, 1.0)
    safe_q = tf.clip_by_value(q_vals, epsilon, 1.0)

    kl_term = tf.reduce_sum(safe_p * tf.math.log(safe_p / safe_q), axis=1)
    # Replace NaN with 0 just in case (better be safe than be nan XD)
    kl_term = tf.where(tf.math.is_nan(kl_term), tf.zeros_like(kl_term), kl_term)
    return kl_term # (dims,)

@tf.function   
def kde_tf(samples, grid, bandwidth):
    samples_T = tf.transpose(samples) 
    grid_exp = tf.reshape(grid, [1, -1, 1])  
    
    samples_exp = tf.expand_dims(samples_T, 1)  
    diff = grid_exp - samples_exp
    
    kernel_vals = tf.exp(-0.5 * tf.square(diff / bandwidth))
    kernel_vals /= (bandwidth * tf.sqrt(2. * tf.constant(3.14159, dtype=tf.float32)))
    
    # Average over samples (axis=2) to get KDE values per dimension:
    return tf.reduce_mean(kernel_vals, axis=2)  # (dims, n_grid)

@tf.function
def kl_div_term(y, p_base):
    batch_size = tf.cast(tf.shape(p_base)[0], tf.float32)
    mu = p_base[:, 0:8:2]  # shape (batch_size, 4)
    Mdia = 1e-9 + tf.math.maximum(p_base[:, 1:8:2], 0.0)  # shape (batch_size, 4)
    Mcov = p_base[:, 8:]  # shape (batch_size, 6)
    
    # Build the lower triangular matrix for each batch sample.
    zeros = tf.zeros_like(Mdia[:, 0])
    row1 = tf.stack([Mdia[:, 0], zeros,      zeros,      zeros], axis=1)  # (batch_size, 4)
    row2 = tf.stack([Mcov[:, 0], Mdia[:, 1],  zeros,      zeros], axis=1)
    row3 = tf.stack([Mcov[:, 1], Mcov[:, 2],  Mdia[:, 2], zeros], axis=1)
    row4 = tf.stack([Mcov[:, 3], Mcov[:, 4],  Mcov[:, 5], Mdia[:, 3]], axis=1)
    scale_tril = tf.stack([row1, row2, row3, row4], axis=1)  # (batch_size, 4, 4)

    scale_tril += 1e-6 * tf.eye(4, batch_shape=[tf.shape(scale_tril)[0]])
    
    # Solve for the pulls via the lower-triangular system.
    residual = y - mu  # (batch_size, 4)
    residual = tf.expand_dims(residual, -1)  # (batch_size, 4, 1)
    pull = tf.linalg.triangular_solve(scale_tril, residual, lower=True)
    pull = tf.squeeze(pull, axis=-1)  # (batch_size, 4)
    
    # Compute KDE for each of the 4 dimensions simultaneously.
    grid = tf.linspace(-5.0, 5.0, 500) 
    bandwidth = 0.3
    
    kde_vals = kde_tf(pull, grid = grid, bandwidth = bandwidth)  # (4, 500)
    
    # Evaluate the standard normal distribution on the grid (same for all dims).
    q_dist = tfp.distributions.Normal(loc=0., scale=1.)
    q_vals = q_dist.prob(grid)  # (500,)
    
    # Calculate the empirical KL divergence for each dimension.
    kl_values = empirical_kl(kde_vals, q_vals)  # (4,)
    total_kl = tf.reduce_mean(kl_values)
    return total_kl * batch_size


# custom loss function
def custom_loss_with_regularization(y, p_base, minval=1e-9, maxval=1e9, scale=0.01, log_var=False):
    
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

    scale_tril = tf.transpose(tf.stack([row1,row2,row3,row4]),perm=[2,0,1]) # this is the cholesky decomposition
    cov_matrix = tf.matmul(scale_tril, scale_tril, transpose_b=True) # covariance_matrix = cholesky_decomp * cholesky_decomp_transposed
    variances = tf.linalg.diag_part(cov_matrix) # diagonals of covariance matrix
    log_variances = tf.math.log(variances + minval) 

    dist = tfp.distributions.MultivariateNormalTriL(loc = mu, scale_tril = scale_tril) 
    
    likelihood = dist.prob(y)  
    likelihood = tf.clip_by_value(likelihood,minval,maxval)
    
    NLL = -1*tf.math.log(likelihood)

    # Regularization term (Mdia, the uncertainty predictions)
    if log_var:
        reg_term = tf.abs(scale * tf.reduce_mean(tf.reduce_sum(log_variances, axis=1)))
    else:
        reg_term = scale * tf.reduce_sum(tf.square(variances))

    return tf.keras.backend.sum(NLL) + reg_term

# regularization loss
def regularization_loss_variance(y, p_base, minval=1e-9, maxval=1e9, scale=0.01, log_var=False):
    p = p_base
    mu = p[:, 0:8:2]
    Mdia = minval + tf.math.maximum(p[:, 1:8:2], 0.0)
    Mcov = p[:,8:]
    zeros = tf.zeros_like(Mdia[:,0])
    
    row1 = tf.stack([Mdia[:,0],zeros,zeros,zeros])
    row2 = tf.stack([Mcov[:,0],Mdia[:,1],zeros,zeros])
    row3 = tf.stack([Mcov[:,1],Mcov[:,2],Mdia[:,2],zeros])
    row4 = tf.stack([Mcov[:,3],Mcov[:,4],Mcov[:,5],Mdia[:,3]])

    scale_tril = tf.transpose(tf.stack([row1,row2,row3,row4]),perm=[2,0,1]) # this is the cholesky decomposition
    cov_matrix = tf.matmul(scale_tril, scale_tril, transpose_b=True) # covariance_matrix = cholesky_decomp * cholesky_decomp_transposed
    variances = tf.linalg.diag_part(cov_matrix) # diagonals of covariance matrix
    log_variances = tf.math.log(variances + minval) 

    # Regularization term (Mdia, the uncertainty predictions)
    if log_var:
        reg_term = tf.abs(scale * tf.reduce_mean(tf.reduce_sum(log_variances, axis=1)))
    else:
        reg_term = scale * tf.reduce_sum(tf.square(variances))

    return reg_term

# custom kl divergence loss function
def custom_kld_loss(y, p_base, minval=1e-9, maxval=1e9):

    # truth values
    xtrue = y[:,0]
    ytrue = y[:,1]
    cotAtrue = y[:,2]
    cotBtrue = y[:,3]

    # predictions
    p = p_base
    x = p[:,0]
    y = p[:,2]
    cotA = p[:,4]
    cotB = p[:,6]
    sigmax = tf.abs(p[:,1])+minval
    sigmay = tf.sqrt(p[:,3]**2 + p[:,8]**2 )+minval
    sigmacotA = tf.sqrt(p[:,5]**2 + p[:,9]**2 + p[:,10]**2)+minval
    sigmacotB = tf.sqrt(p[:,6]**2 + p[:,7]**2 + p[:,11]**2 + p[:,12]**2+ p[:,13]**2)+minval
    
    # pulls
    pullx = (xtrue - x) / sigmax
    pully = (ytrue - y) / sigmay
    pullcotA = (cotAtrue - cotA) / sigmacotA
    pullcotB = (cotBtrue - cotB) / sigmacotB

    stacked_pulls = tf.stack([pullx, pully, pullcotA, pullcotB], axis=0) 

    # construct multivariate Gaussian
    mean = p[:, 0:8:2]
    
    Mdia = minval + tf.math.maximum(p[:, 1:8:2], 0.0)
    Mcov = p[:,8:]
    zeros = tf.zeros_like(Mdia[:,0])
    row1 = tf.stack([Mdia[:,0],zeros,zeros,zeros])
    row2 = tf.stack([Mcov[:,0],Mdia[:,1],zeros,zeros])
    row3 = tf.stack([Mcov[:,1],Mcov[:,2],Mdia[:,2],zeros])
    row4 = tf.stack([Mcov[:,3],Mcov[:,4],Mcov[:,5],Mdia[:,3]])
    scale_tril = tf.transpose(tf.stack([row1,row2,row3,row4]),perm=[2,0,1])

    mv_gaussian = tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril = scale_tril) 

    # construct multivariate Normal 
    standard_normal = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(4), scale_diag=tf.ones(4))

    # construct the probability distributions (pseudo-code)
    #KDE_fit_mv_gaussian_probs     --> mv_gaussian.prob(stacked_pulls)
    #KDE_fit_standard_normal_probs --> standard_normal.prob(stacked_pulls)
    
    # compute KL divergence
    kl_divergence = tfp.distributions.kl_divergence(KDE_fit_mv_gaussian_probs, KDE_fit_standard_normal_probs)

    return kl_divergence

# custom sse loss function (only predict values, no uncertainties)
def custom_sse_loss(y, p_base, minval=1e-9, maxval=1e9):

    # truth values
    x_true = y[:,0]
    y_true = y[:,1]
    cotA_true = y[:,2]
    cotB_true = y[:,3]

    # predictions
    p = p_base
    x_pred = p[:,0]
    y_pred = p[:,1]
    cotA_pred = p[:,2]
    cotB_pred = p[:,3]

    sse_x = tf.reduce_sum(tf.square(x_true - x_pred))
    sse_y = tf.reduce_sum(tf.square(y_true - y_pred))
    sse_cotA = tf.reduce_sum(tf.square(cotA_true - cotA_pred))
    sse_cotB = tf.reduce_sum(tf.square(cotB_true - cotB_pred))

    return (sse_x + sse_y + sse_cotA + sse_cotB) / 4.0

def custom_mle_loss(y, p_base, minval=1e-9, maxval=1e9):
    # truth values
    x_true = y[:,0]
    y_true = y[:,1]
    cotA_true = y[:,2]
    cotB_true = y[:,3]

    # predictions
    p = p_base
    x_pred = p[:,0]
    log_sigma_x_pred = p[:,1]
    y_pred = p[:,2]
    log_sigma_y_pred = p[:,3]
    cotA_pred = p[:,4]
    log_sigma_cotA_pred = p[:,5]
    cotB_pred = p[:,6]
    log_sigma_cotB_pred = p[:,7]

    # loss terms
    x_loss = tf.reduce_sum(2.0*log_sigma_x_pred + tf.square((x_true - x_pred)/tf.exp(log_sigma_x_pred)))
    y_loss = tf.reduce_sum(2.0*log_sigma_y_pred + tf.square((y_true - y_pred)/tf.exp(log_sigma_y_pred)))
    cotA_loss = tf.reduce_sum(2.0*log_sigma_cotA_pred + tf.square((cotA_true - cotA_pred)/tf.exp(log_sigma_cotA_pred)))
    cotB_loss = tf.reduce_sum(2.0*log_sigma_cotB_pred + tf.square((cotB_true - cotB_pred)/tf.exp(log_sigma_cotB_pred)))

    return (x_loss + y_loss + cotA_loss + cotB_loss) / 4.0

    
    
    




    