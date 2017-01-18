import tensorflow as tf
import numpy as np
from src.utilities import *

_epsilon = 1.0e-8

class _HMM():
    
    """
    Hidden Markov model base class using Tensorflow.
    Do not use directly, instead inherit from it.
    """
    
    def __init__(self, obs, prior_pi, prior_A, init_pi=None, init_A=None):
        """
        Initialise the model. N.b. initialisation of prior/posterior
        of emission distribution is not done in base class.
        
        Parameters
        ==========
        
        obs : T x D dim array of T observations of dimensionality D
        
        prior_pi : K dim array of prior params of initial distributions, 
            e.g. hyperparameters of Dirichlet dbn
        
        prior_A : K x K dim array of prior params of transition matrix, 
            e.g. hyperparameters of K Dirichlet dbns
            
        init_pi : K dim  array of initialisation of variational initial 
            distribution. If not specified, take mean of prior_pi
            
        init_A : K x K dim array of initialisation of variational transition 
            matrix. If not specified, take mean of each row of prior_A
        
        """
        
        with tf.name_scope('Scalar_Constants'):
            self._K = len(prior_pi)
            self._T, self._D = np.shape(obs)
        
        with tf.name_scope('Data'):
            self._obs = tf.constant(obs, dtype=tf.float32, name='observations')
            
        with tf.name_scope('Prior_Hyperparams'):
            self._pi_0 = tf.Variable(prior_pi, 
                                    dtype=tf.float32, name='pi_0')
            self._A_0 = tf.Variable(prior_A, 
                                   dtype=tf.float32, name='A_0')
                
        with tf.name_scope('Variational_Hyperparams'):
            if init_pi is None:
                prior_pi = tf.cast(prior_pi, tf.float32)
                self._var_pi = tf.Variable(
                    prior_pi / tf.reduce_sum(prior_pi, keep_dims=True), 
                    dtype=tf.float32, name='pi')
            else:
                self._var_pi = tf.Variable(init_pi, dtype=tf.float32, name='pi')
                
            if init_A is None:
                prior_A = tf.cast(prior_A, tf.float32)
                self._var_A = tf.Variable(
                    prior_A / tf.reduce_sum(prior_A, axis=1, keep_dims=True),
                    dtype=tf.float32, name='A')
            else:
                self._var_A = tf.Variable(init_A, dtype=tf.float32, name='A')
        
    
    def _fwd(self, obs=None):
        """
        Compute the forward parameters as well as scaling parameters.
        
        Parameters
        ==========
        
        obs : S x D dim array of S observations of dimensionality D. Used when
            not performing batched inference, as defaults to self._obs
        """
        
        if obs is None:
            obs = self._obs

        l_A = self._aux_A
        l_l = self._ln_lik
        
        # Belief at t=0
        with tf.name_scope('Forward_First_Step'):
            # Calculate the values
            l_alpha = self._aux_pi + l_l[0,:]
            l_scale = tf.reciprocal(
                tf.log(tf.reduce_sum(tf.exp(l_alpha), keep_dims=True)))
            
            # Update the variables
            tf.scatter_update(
                    self._ln_scale, tf.constant([0]), tf.expand_dims(l_scale, 0))
            tf.scatter_update(
                self._ln_alpha, tf.constant([0]), tf.expand_dims(l_alpha + l_scale, 0))

        # Propagate belief
        for t in range(1, self._T):
            
            with tf.name_scope('Forward_Time_Step_{}'.format(t)):
                # N.B. transpose transition matrix for broadcasting addition
                # across all values of k
                l_alpha = log_sum_exp(
                    self._ln_alpha[t-1, :] + tf.transpose(l_A)) + l_l[t,:]
                l_scale = tf.reciprocal(
                    tf.log(tf.reduce_sum(tf.exp(l_alpha), keep_dims=True)))

                # Update forward matrix
                tf.scatter_update(
                    self._ln_scale, tf.constant([t]), tf.expand_dims(l_scale, 0))
                tf.scatter_update(
                    self._ln_alpha, tf.constant([t]),
                    tf.expand_dims(l_alpha + l_scale, 0))

                
    def _bwd(self, obs=None):
        """
        Compute the backwards parameters.
        
        Parameters
        ==========
        
        obs : S x D dim array of S observations of dimensionality D. Used when
            not performing batched inference, as defaults to self._obs
        """
        
        if obs is None:
            obs = self._obs

        l_A = self._aux_A
        l_l = self._ln_lik
        
        with tf.name_scope('Backwards_First_Step'):
            # At first backwards step, scaled beta = 1 * scale[T-1]
            b_0 = np.array([1] * self._K) * self._ln_scale[self._T-1]
            tf.scatter_update(
                self._ln_alpha, tf.constant([self._T-1]), tf.expand_dims(b_0,0))

        for t in range(self._T-2, -1, -1):
            
            with tf.name_scope('Backwards_Time_Step_{}'.format(t)):
                
                # N.B. transpose transition matrix for broadcasting addition
                # across all values of k
                l_beta = log_sum_exp(l_A +  self._ln_beta[t+1, :] + l_l[t+1,:])

                # Update backwards matrix
                tf.scatter_update(
                    self._ln_beta, tf.constant([t]),
                    tf.expand_dims(l_beta + self._ln_scale[t], 0))
        
        
    def _local_update(self, obs=None):
        """
        Local updates. Creating modified parameters (auxillary parameters)
        for running the forward-backwards algorithm on. Obs parameter
        defaults to the entire set of observations, for in the case of batch
        VB.
        
        Parameters
        ==========
        
        obs : S x D dim array of S observations of dimensionality D. Used when
            not performing batched inference, as defaults to self._obs
        """
        
        if obs is None:
            obs = self._obs
        
        self._aux_pi = tf.assign(self._aux_pi,
            dirichlet_expected_log_likelihood(self._var_pi))
        
        self._aux_A = tf.assign(self._aux_A,
            dirichlet_expected_log_likelihood(self._var_A))
        
        # Compute log-likelihoods
        self._ln_lik = self._emission_log_likelihood(obs)
        
        # Update forward, and backward values
        self._fwd()
        self._bwd()
        
        # Update the marginal belief
        self._marginal.assign(tf.exp(self._ln_alpha + self._ln_beta))
        
        
    def _emission_log_likelihood(self):
        """
        Must override.
        """
        pass
        
        
    def _emission_lower_bound(self):
        """
        Must override.
        """
        pass
    
    
    def _lower_bound(self):
        """
        Compute variational lower bound
        """
        
        elbo = tf.constant(0, dtype=tf.float32)
        
        elbo += dirichlet_lower_bound(self._pi_0, self._var_pi)
        elbo += tf.reduce_sum(
            dirichlet_lower_bound(self._A_0, self._var_A))
        elbo += self._emission_lower_bound()
        # Entropy of states
        elbo += -tf.reduce_sum(self._ln_scale)
        
        return tf.squeeze(elbo)
        
    
    def _summarise(self, name, value):
        #log for tensorboard
        tf.summary.scalar(name, value)