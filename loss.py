import tensorflow as tf
import tensorflow_probability as tfp

# custom loss function
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