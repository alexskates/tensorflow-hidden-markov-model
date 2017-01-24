import numpy as np
from scipy.special import digamma, gammaln

"""
Utilities, distributions, etc
"""

_epsilon = 1.0e-8

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
    
    ln_lik = digamma(alpha + _epsilon) - \
        digamma(np.sum(alpha, axis=-1, keepdims=True) + _epsilon)

    return ln_lik

def dirichlet_lower_bound(alpha_0, alpha):
    """
    Calculate the variational lower bound of a Dirichlet distribution
    """
    
    sum_a_0 = np.sum(alpha_0, axis=-1, keepdims=True)
    sum_a = np.sum(alpha, axis=-1, keepdims=True)
    
    energy = gammaln(sum_a_0) - \
        np.sum(gammaln(alpha_0), axis=-1, keepdims=True) + \
        np.sum((alpha_0 - 1) * (digamma(alpha + _epsilon) + 
        digamma(sum_a + _epsilon)), axis=-1, keepdims=True)
    
    entropy = -(gammaln(sum_a) - \
        np.sum(gammaln(alpha), axis=-1, keepdims=True) + \
        np.sum((alpha - 1) * (digamma(alpha + _epsilon) + \
        digamma(sum_a + _epsilon)), axis=-1, keepdims=True))
        
    return np.sum(energy + entropy)