import tensorflow as tf
import numpy as np
from src.hmm_base_class import _HMM
from src.utilities import *

_epsilon = 1.0e-8

class VB_HMM(_HMM):
    
    """
    Hidden Markov model using Tensorflow.
    Trains a HMM with gaussian emissions using batch VB,
    updating parameters using the natural gradient
    """
    
    def __init__(self, obs, prior_pi, prior_A, prior_emit, 
                 init_pi=None, init_A=None, init_emit=None, 
                 conv=1.0e-8, max_iter=2):
        """
        Initialise the model. N.b. initialisation of prior/posterior
        of emission distribution is not done in base class.
        
        Parameters
        ==========
        
        obs : T x D dim array of T observations of dimensionality D
        
        prior_pi : K dim array of params of initial distribution priors
            e.g. hyperparameters of Dirichlet dbn
        
        prior_A : K x K dim array of params of transition  matrix prior
            e.g. hyperparameters of K Dirichlet dbns
            
        prior_emit : 4 arrays of parameters of K emission dbn priors
            i.e. K NIW parameterised by:
                prior_emit[0][k] = mu_0 D dimensional array
                prior_emit[1][k] = Sigma_0 D x D symmetric definite matrix
                prior_emit[2][k] = k_0 > 0
                prior_emit[3][k] = v_0 > D + 2
            
        init_pi : K dim array of initialisation of variational 
            initial distribution. If not specified, take mean of prior_pi
            
        init_A : K x K dim array of initialisation of variational
            transition matrix. If not specified, take mean of each
            row of prior_A
            
        conv : Float, convergence criteria. Defaults to 1.0 x 10^-8
        
        max_iter : Integer specifying the maximum number of iterations 
            performed in the inference, in the case that conv is not met.
            Defaults to 100.
        
        """
        
        # Call _HMM base class to initialise prior/variational approx of
        # initial dbn and transition matrix
        super(VB_HMM, self).__init__(obs, prior_pi, prior_A, init_pi, init_A)
        
        with tf.name_scope('Scalar_Constants'):
            self._epsilon = conv
            self._max_iter = max_iter
        
        with tf.name_scope('Prior_Hyperparams'):# K different D dimensional NIWs
            self._mu_0 = tf.Variable(prior_emit[0], dtype=tf.float32, name='mu_0')
            self._Sigma_0 = tf.Variable(prior_emit[1], dtype=tf.float32, name='Sigma_0')
            self._k_0 = tf.Variable(prior_emit[2], dtype=tf.float32, name='kappa_0')
            self._v_0 = tf.Variable(prior_emit[3], dtype=tf.float32, name='nu_0')
        
        with tf.name_scope('Variational_Hyperparams'):
            self._var_mu = tf.Variable(prior_emit[0], dtype=tf.float32, name='mu')
            self._var_Sigma = tf.Variable(prior_emit[1], dtype=tf.float32, name='Sigma')
            self._var_k = tf.Variable(prior_emit[2], dtype=tf.float32, name='kappa')
            self._var_v = tf.Variable(prior_emit[3], dtype=tf.float32, name='nu')

        self._initialise_fwd_bwd()
        
        with tf.name_scope('Marginal_Belief'):
            #marginal = np.random.rand(self._T, self._K)
            #marginal /= np.sum(marginal, axis=1)[:,np.newaxis]
            marginal = np.zeros([self._T, self._K])
            self._marginal = tf.Variable(marginal, dtype=tf.float32, name='marginal_belief')

        with tf.name_scope('Auxillary_Params'):
            # The auxillary parameters used in the message passing
            self._aux_pi = tf.Variable(
                tf.zeros([self._K], dtype=tf.float32), name='pi_tilde')
            self._aux_A = tf.Variable(
                tf.zeros([self._K, self._K], dtype=tf.float32), name='A_tilde')
        
        with tf.name_scope('Lower_Bound'):
            # Start off with a big number so will not have converged on first iteration
            self._elbo = tf.Variable(1e10, dtype=tf.float32, name='elbo')
            self._elbo = tf.Variable(1e10, dtype=tf.float32, name='elbo')


    def _initialise_fwd_bwd(self):
        """
        Note that this is not defined in the base class as the size of 
        these matrices vary depending on the method
        """
        
        with tf.name_scope('Forward_Backward_Params'):
            self._ln_alpha = tf.Variable(
                tf.zeros([self._T, self._K], dtype=tf.float32), name='ln_alpha')
            self._ln_beta = tf.Variable(
                tf.zeros([self._T, self._K], dtype=tf.float32), name='ln_beta')
            self._ln_scale = tf.Variable(
                tf.zeros([self._T, 1], dtype=tf.float32), name='ln_scaling')
            self._ln_lik = tf.Variable(
                tf.zeros([self._T, self._K], dtype=tf.float32), name='ln_likelihood')
    
    
    def _fit(self):
        """
        Perform VI through coordinate descent, effectively maximising
        the ELBO. In practice, alternate between updating local variables {x_t}
        and global variables theta = {pi, A, {mu_k, Sigma_k}_k=1^K}
        This is effectively the variational equivalent of the EM algorithm
        
        The updates are derived by differentiating the ELBO wrt the respective
        variational parameters.
        """
            
        with tf.name_scope('Baum_Welch_Recursions'):
            counter = tf.constant(0)
            converged = tf.constant(False)
            
            def body_fn(i):
                tf.Print(i, [i], 'Iteration')
                
                with tf.name_scope('Local_Update'):
                    self._local_update()
                
                with tf.name_scope('Global_Update'):
                    self._global_update()
                
                lb = self._lower_bound()
                self._summarise('Lower_Bound', lb)
                
                converged = tf.abs(lb - self._elbo) < self._epsilon
                self._elbo.assign(lb)
                
                return converged
            
            def conditions(iteration, converged):
                return tf.logical_or(
                    tf.less(iteration, self._max_iter), 
                    tf.logical_not(converged))
            
            body = lambda i, c: (i + 1, body_fn(i))
            vars = [counter, converged]
            count, conv = tf.while_loop(conditions, body, vars)
        
        return count, conv
    
    
    def infer(self, summary=True):
        """
        Perform inference. Note that self._fit() does all the work,
        this function is a wrapper in which we initialise the session,
        summaries, etc.
        """
        
        with tf.Session() as sess:
            counter, converged = self._fit()
            print('test1')
            sess.run(tf.global_variables_initializer())
            
            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter('Logs', graph=sess.graph)
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str)
        
            print('test2')
            count, conv = sess.run([counter, converged])
            print('test3')
            self.lower_bound = sess.run([self._elbo])
            
                

        return count, conv
       
        
    def _emission_log_likelihood(self, obs):
        """
        Log likelihood for each NIW emission distribution
        
        Parameters
        ==========
        
        obs : T x D dim array of T observations of dimensionality D
        
        Returns
        =======
        
        ln_lik : T x K dim array of log likelihoods of each emission at each t
        
        """
        
        return tf.transpose(tf.pack(
                [niw_expected_log_likelihood(
                        obs, self._var_mu[i], self._var_Sigma[i], self._var_k[i], 
                        self._var_v[i]) for i in np.arange(self._K)]))
        
            
    def _emission_lower_bound(self):
        """
        Variational lower bound for all emissions
        
        """
        lower_bound = tf.constant([0], dtype=tf.float32)
        
        for k in range(self._K):
            lower_bound += niw_lower_bound(self._mu_0[k], self._Sigma_0[k], self._k_0[k], 
                                           self._v_0[k], self._var_mu[k], 
                                           self._var_Sigma[k], self._var_k[k], 
                                           self._var_v[k])
        
        return lower_bound
    
    def _global_update(self):
        """
        Global update for batch VI. Update the hyperparameters of the 
        variational distributions over the parameters of the HMM.
        """
         # Initial parameter update
        self._var_pi.assign(self._pi_0 + self._marginal[0,:])

        # Transition parameter updates
        for t in range(self._T):
            tf.assign_add(self._var_A, 
                          tf.matmul(tf.expand_dims(self._marginal[t-1,:],-1),
                          tf.expand_dims(self._marginal[t,:],0)))

        # Emission parameter updates
        # Get the expected sufficient statistics
        t_1, t_2, t_3, t_4 = niw_sufficient_statistics(self._obs, self._marginal)
        
        for k in range(self._K):
            
            # Get prior parameters and convert to natural form
            u_1, u_2, u_3, u_4 = NIW_moment_to_natural(self._mu_0[k], self._Sigma_0[k],
                                                       self._k_0[k], self._v_0[k])
            # Update the hyperparameters of the emission distributions.
            w_1 = u_1 + t_1[k]
            w_2 = u_2 + t_2[k]
            w_3 = u_3 + t_3[k]
            w_4 = u_4 + t_4[k]
            
            mu, sigma, kappa, nu = niw_natural_to_moment(w_1, w_2, w_3, w_4)
            
            tf.scatter_update(self._var_mu, tf.constant([k]), tf.expand_dims(mu, 0))
            tf.scatter_update(self._var_Sigma, tf.constant([k]), tf.expand_dims(sigma, 0))
            tf.scatter_update(self._var_k, tf.constant([k]), tf.expand_dims(kappa, 0))
            tf.scatter_update(self._var_v, tf.constant([k]), tf.expand_dims(nu, 0))