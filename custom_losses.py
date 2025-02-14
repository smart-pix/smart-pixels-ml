import tensorflow as tf
import tensorflow_probability as tfp

# custom loss function for model_1 (predicting values-only)
def custom_loss_sse(y, pred):
    p = pred

    # truth values
    x_true = y[:, 0]
    y_true = y[:, 1]
    cotA_true = y[:, 2]
    cotB_true = y[:, 3]

    # predicted values
    x_pred = p[:, 0]
    y_pred = p[:, 1]
    cotA_pred = p[:, 2]
    cotB_pred = p[:, 3]

    # sse values
    sse_x = tf.reduce_sum(tf.square(x_true - x_pred))
    sse_y = tf.reduce_sum(tf.square(y_true - y_pred))
    sse_cotA = tf.reduce_sum(tf.square(cotA_true - cotA_pred))
    sse_cotB = tf.reduce_sum(tf.square(cotB_true - cotB_pred))

    sse_final = sse_x + sse_y + sse_cotA + sse_cotB

    return sse_final

# custom loss function for model_1 (predicting values-only)
def custom_loss_sse_weighted(weight=5.0, edge_value=1.0):
    def loss_function(y, pred):
        # truth values
        x_true = y[:, 0]
        y_true = y[:, 1]
        cotA_true = y[:, 2]
        cotB_true = y[:, 3]

        # predicted values
        x_pred = pred[:, 0]
        y_pred = pred[:, 1]
        cotA_pred = pred[:, 2]
        cotB_pred = pred[:, 3]

        # weighting loss values (higher for values further from 0)
        x_distances = tf.abs(x_true)
        y_distances = tf.abs(y_true)
        cotA_distances = tf.abs(cotA_true)
        cotB_distances = tf.abs(cotB_true)

        x_weights = 1.0 + weight * tf.exp(-(edge_value - x_distances) / edge_value)
        y_weights = 1.0 + weight * tf.exp(-(edge_value - y_distances) / edge_value)
        cotA_weights = 1.0 + weight * tf.exp(-(edge_value - cotA_distances) / edge_value)
        cotB_weights = 1.0 + weight * tf.exp(-(edge_value - cotB_distances) / edge_value)

        # sse values
        sse_x = tf.reduce_sum(tf.square(x_true - x_pred) * x_weights)
        sse_y = tf.reduce_sum(tf.square(y_true - y_pred) * y_weights)
        sse_cotA = tf.reduce_sum(tf.square(cotA_true - cotA_pred) * cotA_weights)
        sse_cotB = tf.reduce_sum(tf.square(cotB_true - cotB_pred) * cotB_weights)

        sse_final = sse_x + sse_y + sse_cotA + sse_cotB

        return sse_final
    
    return loss_function

    
# custom loss function for default model 
def custom_loss_kld(y, p_base, minval=1e-9, maxval=1e9):

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

    # gaussian approximation
    mu_x, stdev_x = tf.reduce_mean(pullx), tf.math.reduce_std(pullx)
    mu_y, stdev_y = tf.reduce_mean(pully), tf.math.reduce_std(pully)
    mu_cotA, stdev_cotA = tf.reduce_mean(pullcotA), tf.math.reduce_std(pullcotA)
    mu_cotB, stdev_cotB = tf.reduce_mean(pullcotB), tf.math.reduce_std(pullcotB)
    
    # Clip values to avoid numerical instability
    stdev_x = tf.clip_by_value(stdev_x, minval, maxval)
    stdev_y = tf.clip_by_value(stdev_y, minval, maxval)
    stdev_cotA = tf.clip_by_value(stdev_cotA, minval, maxval)
    stdev_cotB = tf.clip_by_value(stdev_cotB, minval, maxval)

    # Construct multivariate Gaussian distributions
    mean = tf.stack([mu_x, mu_y, mu_cotA, mu_cotB], axis=0)
    covariance = tf.linalg.diag(tf.stack([stdev_x**2, stdev_y**2, stdev_cotA**2, stdev_cotB**2], axis=0))
    tril_covariance = tf.linalg.cholesky(covariance)
    mv_gaussian = tfp.distributions.MultivariateNormalTriL(loc=mean, scale_tril=tril_covariance)

    # Standard 4D normal distribution (mean=0, identity covariance)
    standard_normal = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(4), scale_diag=tf.ones(4))

    # Compute KL divergence
    kl_divergence = tfp.distributions.kl_divergence(mv_gaussian, standard_normal)

    return kl_divergence