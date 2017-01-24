import tensorflow as tf
import numpy as np

"""
Utilities, distributions, etc
"""

_epsilon = 1.0e-8

def log_sum_exp(x):
    """
    Calculate the logarithm of the sum of the exponential of values x,
    as the operation can have potential issues with under/overflow depending 
    on the size of x values. Instead subtract the maximum value of x_n from
    x_n, ensuring the largest value is 0. Note that underflow may still
    occur if there is a large difference between values
    
    Function works on the basis of identity:
    
    \log\sum_{n=1}^N\exp{x_n} = a + \log\sum_{n=1}^N\exp{x_n - a},
    
    where a = \max_n{x_n}
    
    Parameters
    ==========
    
    x : N dim array of values
    
    Returns
    =======
    
    N dim array of the sum of exponentiated values of x, as calculated by 
    the procedure outlined above
    
    """
    # Ensure it's a float
    x = tf.to_float(x)
    
    maxes = tf.reduce_max(x, keep_dims=True)
    x -= maxes
    
    return tf.squeeze(maxes, [-1]) + tf.log(tf.reduce_sum(tf.exp(x), -1))

def dirichlet_expected_log_likelihood(alpha):
    """
    Calculate the expected log likelihood of a Dirichlet distribution
    
    Parameters
    ==========
        
    alpha : D dim array of parameters of Dirichlet distribution
    
    Returns
    =======
    
    ln_lik : D dim array of expected log likelihood for each d in D
    """
    
    with tf.name_scope('Dirichlet_Expected_Log_Likelihood'):
    
        ln_lik = tf.digamma(alpha + _epsilon) - \
            tf.digamma(tf.reduce_sum(alpha, axis=-1, keep_dims=True) + _epsilon)

    return ln_lik

def dirichlet_lower_bound(alpha_0, alpha):
    """
    Calculate the variational lower bound of a Dirichlet distribution
    """
    
    with tf.name_scope('Dirichlet_Variational_Lower_Bound'):
    
        sum_a_0 = tf.reduce_sum(alpha_0, keep_dims=True, axis=-1)
        sum_a = tf.reduce_sum(alpha, keep_dims=True, axis=-1)
    
        energy = tf.lgamma(sum_a_0) - tf.reduce_sum(tf.lgamma(alpha_0), 
                                                    keep_dims=True, axis=-1) + \
            tf.reduce_sum((alpha_0 - 1) * (tf.digamma(alpha + _epsilon) + \
            tf.digamma(sum_a + _epsilon)), keep_dims=True, axis=-1)
    
        entropy = -(tf.lgamma(sum_a) - tf.reduce_sum(tf.lgamma(alpha), 
                                                    keep_dims=True, axis=-1) + \
            tf.reduce_sum((alpha - 1) * (tf.digamma(alpha + _epsilon) + \
            tf.digamma(sum_a + _epsilon)), keep_dims=True, axis=-1))
        
    return energy + entropy
    

def niw_expected_log_likelihood(x, mu, sigma, kappa, nu):
    """
    Calculate the expected log likelihood of a Normal Inverse-
    Wishart distribution. See equations 10.71, 10.32 and 10.65
    in PRML, Bishop
    
    Parameters
    ==========
        
    x : T x D dim array of T observations of dimensionality D
        
    mu : D dim array
    
    sigma : D x D symmetric definite matrix
    
    kappa : Float > 0
    
    nu : Float > D + 2
    
    Returns
    =======
    
    ln_lik : T dim array of expected log likelihood for each t \in T
    """
    
    with tf.name_scope('NIW_Expected_Log_Likelihood'):
        
        D = tf.cast(tf.shape(mu)[0], tf.float32)
    
        ln_lambda_tilde = niw_log_lambda_tilde(nu, sigma)

        x_minus_mu = (x-tf.expand_dims(mu,0)) # Can use broadcasting here
        
        # There is no broadcasting with matmul, so have to perform
        # the second matrix multiplication manually
        xT_sig_x = tf.reduce_sum(tf.matmul(x_minus_mu, sigma) * x_minus_mu, axis=1)
        
        ln_lik = ln_lambda_tilde / 2 - D / (2 * kappa) - nu / 2 * \
            xT_sig_x - D / 2 * tf.log(2*np.pi)
    
    return ln_lik

def niw_lower_bound(mu_0, sigma_0, kappa_0, nu_0, mu, sigma, kappa, nu):
    """
    Calculate the variational lower bound of a Normal Inverse-
    Wishart distribution. See equations 10.74, 10.77 in PRML, Bishop
    (though performed for a single distribution, therefore no summation
    over K)
    
    Parameters
    ==========
    
    Prior hyperparams:
    
    mu_0 : D dim array
    
    sigma_0 : D x D symmetric definite matrix
    
    kappa_0 : Float > 0
    
    nu_0 : Float > D + 2
        
    variational hyperparams:
    
    mu : D dim array
    
    sigma : D x D symmetric definite matrix
    
    kappa : Float > 0
    
    nu : Float > D + 2
    
    Returns
    =======
    
    Float, variational lower bound
    """
    
    with tf.name_scope('NIW_Variational_Lower_Bound'):
        D = tf.cast(tf.shape(mu)[0], tf.float32)
    
        # Entropy of NIW - 10.77
        ln_lambda_tilde = niw_log_lambda_tilde(nu, sigma)
        
        # Log entropy of the Wishart distribution parameterised by Sigma, nu
        # given by B.82, Bishop
        ln_B = iw_log_partition_function(nu, sigma)
    
        entropy_wishart = -ln_B - 0.5 * (nu - D - 1) * ln_lambda_tilde + \
            0.5 * nu * D
        
        entropy = 0.5 * ln_lambda_tilde + 0.5 * D * tf.log(kappa / (2*np.pi)) - \
            0.5 * D - entropy_wishart
            
        # Energy of NIW prior
        ln_B_0 = iw_log_partition_function(nu_0, sigma_0)
        
        mu_minus = tf.expand_dims(mu - mu_0, -1)
        
        energy = 0.5 * (D * tf.log(kappa_0 / (2 * np.pi)) + ln_lambda_tilde - \
            D * tf.div(kappa_0, kappa) - kappa_0 * nu * tf.matmul(tf.matmul(
            tf.transpose(mu_minus), sigma), mu_minus)) + ln_B_0 + \
            0.5 * (nu_0 - D - 1) * ln_lambda_tilde - 0.5 * nu * tf.trace(
            tf.matmul(tf.matrix_inverse(sigma_0), sigma))
        
    return entropy + tf.squeeze(energy)
        
        
def niw_log_lambda_tilde(nu, sigma):
    """
    Given by 10.65 in PRML, Bishop
    
    Parameters
    ==========
    
    nu : Float > D + 2
    
    sigma : D x D symmetric definite matrix
    """
    
    D = tf.cast(tf.shape(sigma)[0], tf.float32)
    
    return tf.reduce_sum(tf.digamma(0.5 * (nu - tf.range(0,D)) + _epsilon)) + \
        D * np.log(2) + tf.log(tf.matrix_determinant(sigma) + _epsilon)


def iw_log_partition_function(nu, sigma):
    """
    Given by B.79 in PRML, Bishop
    
    Parameters
    ==========
    
    nu : Float > D + 2
    
    sigma : D x D symmetric definite matrix
    """
    
    D = tf.cast(tf.shape(sigma)[0], tf.float32)
    
    return -0.5 * nu * tf.matrix_determinant(sigma) - 0.5 * nu * D * np.log(2) - \
        0.25 * D * (D-1) - tf.reduce_sum(tf.lgamma(0.5 * (nu - tf.range(0,D))))  

def niw_sufficient_statistics(data, marginal):
    """
    Calculate the expectation of the sufficient statistics of a Normal Inverse 
    Wishart distribution with respect to a variational distribution q(x). 
    Used for update equations.
    """
    
    with tf.name_scope('Gaussian_Expected_Sufficient_Statistics'):
        
        t_1 = tf.reduce_sum(data * marginal, axis=0)
        t_2 = tf.reduce_sum(marginal, axis=0)
        t_3 = tf.reduce_sum(tf.reduce_sum(data * data, axis=0) * marginal, axis=0)
        t_4 = tf.reduce_sum(marginal, axis=0)
        
    return t_1, t_2, t_3, t_4

def niw_natural_to_moment(u_1, u_2, u_3, u_4):
    """
    Convert the natural parameters of a Normal-Inverse Wishart Distribution
    to the moment form of the distribution (e.g. expressed in terms of mu,
    sigma, kappa and nu).
    """
    
    D = tf.cast(tf.shape(u_1)[0], tf.float32)
    
    mu = u_1 / u_2
    kappa = u_2
    sigma = u_3 - mu * tf.transpose(mu) * kappa
    nu = u_4 - 2 - D
    
    return mu, sigma, kappa, nu


def NIW_moment_to_natural(mu, sigma, kappa, nu):
    """
    Convert the natural parameters of a Normal-Inverse Wishart Distribution
    to the moment form of the distribution (e.g. expressed in terms of mu,
    sigma, kappa and nu).
    """
    D = tf.cast(tf.shape(mu)[0], tf.float32)
    
    u_1 = kappa * mu
    u_2 = kappa
    u_3 = sigma + mu * tf.transpose(mu) * kappa
    u_4 = nu + 2 + D
    
    return u_1, u_2, u_3, u_4