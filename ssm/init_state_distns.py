import autograd.numpy as np
from autograd.scipy.misc import logsumexp

class InitialStateDistribution(object):
    def __init__(self, num_states, observation_dim, input_dim=0):
        self.log_pi0 = -np.log(num_states) * np.ones(num_states)

    @property
    def params(self):
        return (self.log_pi0,)

    @params.setter
    def params(self, value):
        self.log_pi0 = value[0]

    def initialize(self, dataset):
        pass

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_pi0 = self.log_pi0[perm]

    @property
    def init_state_distn(self):
        return np.exp(self.log_pi0 - logsumexp(self.log_pi0))

    def log_prior(self):
        return 0

    def log_initial_state_distn(self, data):
        return self.log_pi0 - logsumexp(self.log_pi0)

    def m_step(self, expectations, dataset, **kwargs):
        pi0 = sum([Ez[0] for Ez, _, _ in expectations]) + 1e-8
        self.log_pi0 = np.log(pi0 / pi0.sum())
