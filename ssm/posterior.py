import autograd.numpy as np
import autograd.numpy.random as npr

import ssm.messages

class Posterior(object):
    """
    Base class for a posterior distribution over latent states given data x
    and parameters theta.

        p(z | x; theta) = p(z, x; theta) / p(x; theta)

    where z is a latent variable and x is the observed data.
    """
    def __init__(self, model, data, input=None, mask=None, tag=None):
        """
        Initialize the posterior with a ref to the model and datas,
        where datas is a list of data arrays.
        """
        self.model = model
        self.data = data
        self.input = input
        self.mask = mask
        self.tag = tag
        self.T = data.shape[0]

    @property
    def expectations(self):
        """
        Return posterior expectations of the latent states given the data
        """
        raise NotImplementedError

    @property
    def mode(self):
        """
        Return posterior mode of the latent states given the data
        """
        raise NotImplementedError

    @property
    def marginal_likelihood(self):
        """
        Compute (or approximate) the marginal likelihood of the data p(x; theta).
        For simple models like HMMs and LDSs, this will be exact.  For more
        complex models, like SLDS and rSLDS, this will be approximate.
        """
        raise NotImplementedError

    def sample(self, num_samples=1):
        """
        Return samples from p(z | x; theta)
        """
        raise NotImplemented

    def update(self):
        """
        Update the posterior distribution given the model parameters.
        """
        raise NotImplementedError


class HMMExactPosterior(Posterior):
    """
    Exact posterior distribution for a hidden Markov model found via message passing.
    """
    def __init__(self, model, data, input=None, mask=None, tag=None):
        super(HMMExactPosterior, self).__init__(model, data, input, mask, tag)

        # Save the log likelihood and expectations
        self._ll = None
        self._expectations = None

    @property
    def _model_params(self):
        model = self.model
        data = self.data
        input = self.input
        mask = self.mask
        tag = self.tag

        pi0 = model.init_state_distn.initial_state_distn(data, input, mask, tag)
        Ps = model.transitions.transition_matrices(data, input, mask, tag)
        log_likes = model.observations.log_likelihoods(data, input, mask, tag)
        return pi0, Ps, log_likes

    @property
    def expectations(self):
        return ssm.messages.hmm_expected_states(*self._model_params)

    @property
    def marginal_likelihood(self):
        return ssm.messages.hmm_normalizer(*self._model_params)

    @property
    def mode(self):
        return ssm.messages.viterbi(*self._model_params)

    def sample(self, num_samples=1):
        if num_samples == 1:
            return ssm.messages.hmm_sample(*self._model_params)
        else:
            params = self._model_params
            return np.array([ssm.messages.hmm_sample(*params) for _ in range(num_samples)])

    def filter(self):
        return ssm.messages.hmm_filter(*self._model_params)

    def denoise(self):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return self.model.observations.smooth(self.expectations[0], self.data, self.input, self.tag)

    def update(self):
        pass
