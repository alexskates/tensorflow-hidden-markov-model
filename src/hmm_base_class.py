import numpy as np
from src.np_utilities import dirichlet_expected_log_likelihood, dirichlet_lower_bound

class _HMM():
    
    """
    Hidden Markov model base class. Do not use directly, instead inherit from it.
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
        
        self._K = len(prior_pi)
        self._T, self._D = np.shape(obs)
        self._obs = obs
        
        #Prior hyperparams
        self._pi_0 = prior_pi
        self._A_0 = prior_A
        
        #Variational hyperparams
        if init_pi is None:
            self._var_pi = prior_pi / np.sum(prior_pi)
        else:
            self._var_pi = init_pi.copy()
            
        if init_A is None:
            self._var_A = prior_A / np.sum(prior_A, axis=1)[:,np.newaxis]
        else:
            self._var_A = init_A.copy()
        
    
    def _fwd(self, obs=None):
        """
        Compute the forward parameters.
        
        Parameters
        ==========
        
        obs : S x D dim array of S observations of dimensionality D. Used when
            not performing batched inference, as defaults to self._obs
        """
        
        if obs is None:
            obs = self._obs
        
        # Belief at t=0
        self._ln_alpha[0] = self._aux_pi + self._ln_lik[0]
            
        # Propagate belief
        for t in range(1, self._T):
            # Log sum exp trick for matrix multiplication in log space
            self._ln_alpha[t] = np.logaddexp.reduce(
                self._ln_alpha[t-1] + self._aux_A.T, axis=1) + self._ln_lik[t]

                
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
        
        # At first backwards step, beta = 1, so log beta = 0
        self._ln_beta[self._T-1] = 0.

        for t in range(self._T-2, -1, -1):
            self._ln_beta[t] = np.logaddexp.reduce(
                self._aux_A + self._ln_beta[t+1] + self._ln_lik[t+1], axis=1)
        
        
    def _local_update(self, obs=None):
        """
        Use the current distributions over model parameters to calculate 
        local updates. Obs parameter defaults to the entire set of 
        observations, for in the case of batch VB.
        
        Parameters
        ==========
        
        obs : S x D dim array of S observations of dimensionality D. Used when
            not performing batched inference, as defaults to self._obs
        """
        
        if obs is None:
            obs = self._obs
        
        # Expectation wrt variational distributions of global params
        self._aux_pi = dirichlet_expected_log_likelihood(self._var_pi)
        self._aux_A = dirichlet_expected_log_likelihood(self._var_A)
        self._ln_lik = self._emission_log_likelihood(obs)
        
        # Update forward, and backward values
        self._fwd()
        self._bwd()
        
        # Update the marginal belief
        self._marginal = self._ln_alpha + self._ln_beta
        self._marginal -= np.max(self._marginal, axis=1)[:, np.newaxis]
        self._marginal = np.exp(self._marginal)
        self._marginal /= np.sum(self._marginal, axis=1)[:, np.newaxis]
        
        
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
        
        elbo = 0
        
        elbo += dirichlet_lower_bound(self._pi_0, self._var_pi)
        elbo += dirichlet_lower_bound(self._A_0, self._var_A)
        elbo += self._emission_lower_bound()
        # Entropy of states
        elbo += np.sum(np.logaddexp.reduce(self._ln_alpha, axis=1))
        
        return elbo