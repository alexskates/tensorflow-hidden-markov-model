import numpy as np
from src.hmm_base_class import _HMM
from copy import deepcopy

_epsilon = 1.0e-10

class VB_HMM(_HMM):
    
    """
    Hidden Markov model.
    Trains a HMM with gaussian emissions using batch VB,
    updating parameters using coordinate descent.
    """
    
    def __init__(self, obs, prior_pi, prior_A, prior_emit, 
                 init_pi=None, init_A=None, init_emit=None, 
                 conv=1.0e-8, max_iter=100):
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
            
        prior_emit : K dim array of emission distribution priors
            
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
        
        self._epsilon = conv
        self._max_iter = max_iter
        
        # Prior_Hyperparams: K  D-dimensional NIWs
        self._emit_0 = deepcopy(prior_emit)
        
        # Variational_Hyperparams
        self._var_emit = deepcopy(prior_emit)
        
        self._initialise_fwd_bwd()
        
        # Marginal_Belief
        marginal = np.random.rand(self._T, self._K)
        marginal /= np.sum(marginal, axis=1)[:,np.newaxis]
        self._marginal = marginal

        # Auxillary parameters used in the message passing
        self._aux_pi = np.zeros([self._K])
        self._aux_A = np.zeros([self._K, self._K])
        
        # Start off with a big number so will not have converged on first iteration
        self._elbo = 1e10


    def _initialise_fwd_bwd(self):
        """
        Note that this is not defined in the base class as the size of 
        these matrices vary depending on the method
        """
        
        self._ln_alpha = np.zeros([self._T, self._K])
        self._ln_beta = np.zeros([self._T, self._K])
        self._ln_lik = np.zeros([self._T, self._K])
    
    
    def infer(self):
        """
        Perform inference.
        """
        
        lb_array = np.zeros(self._max_iter)
        
        for i in range(self._max_iter):
            print("Iteration {}".format(i))
            
            self._local_update()
            self._global_update()
            
            lb = self._lower_bound()
            converged = np.absolute(lb - self._elbo) < self._epsilon
            #print("Current lb: {} Previous lb: {}".format(lb, self._elbo))
            self._elbo = lb
            
            lb_array[i] = lb
            
            if converged:
                lb_array = lb_array[:i] # remove trailing zeros
                break
                
        return converged, lb_array
       
        
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
        
        ln_lik = np.zeros([self._T, self._K])
        
        for i in range(self._K):
            ln_lik[:, i] = self._var_emit[i].expected_log_likelihood(obs)
        
        return ln_lik
        
            
    def _emission_lower_bound(self):
        """
        Variational lower bound for all emissions
        
        """
        
        lower_bound = 0
        
        for k in range(self._K):
            lower_bound += self._var_emit[k].get_vlb()
        
        return lower_bound
    
    def _global_update(self):
        """
        Global update for batch VI. Update the hyperparameters of the 
        variational distributions over the parameters of the HMM.
        """
        
        # Initial parameter update
        self._var_pi = self._pi_0 + self._marginal[0]

        # Transition parameter updates
        self._var_A = self._A_0.copy()
        for t in range(1, self._T):
            self._var_A += np.outer(self._marginal[t-1], self._marginal[t])

        # Emission parameter updates
        for k in range(self._K):
            self._var_emit[k].meanfieldupdate(self._obs, self._marginal[:, k])