import copy
import warnings
from functools import partial
from tqdm.auto import trange

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.tracer import getval
from autograd.misc import flatten
from autograd import value_and_grad

from ssm.optimizers import adam_step, rmsprop_step, sgd_step, convex_combination
from ssm.primitives import hmm_normalizer, hmm_expected_states, hmm_filter, hmm_sample, viterbi
from ssm.util import check_dataset, check_data, check_slds_args, check_variational_args, \
    replicate, collapse

class BaseHMM(object):
    """
    Base class for hidden Markov models.
    """
    def __init__(self, num_states, observation_dim,
                 init_state_distn, transitions, observations,
                 input_dim=0):
        """
        Construct a hidden Markov model (HMM).

        Parameters
        ----------
        num_states : int > 0
            The number of discrete states

        observation_dim : int > 0 or tuple of ints > 0

        init_state_distn : InitialStateDistribution object
            Object encapsulating the initial state distribution
            p(z_1)

        transitions : _Transitions object
            Object encapsulating the transition distribution
            p(z_t | z_{t-1}, ...)

        observations : _Observations object
            Object encapsulating the observation distribution
            p(x_t | z_t, ...)

        input_dim

        """
        self.num_states = num_states
        self.observation_dim = observation_dim
        self.input_dim = input_dim
        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.observations = observations

    @property
    def params(self):
        return self.init_state_distn.params, \
               self.transitions.params, \
               self.observations.params

    @params.setter
    def params(self, value):
        self.init_state_distn.params = value[0]
        self.transitions.params = value[1]
        self.observations.params = value[2]

    @check_dataset
    def initialize(self, dataset):
        """
        Initialize parameters given data.

        Parameters
        ----------

        dataset : dictionary or list of dictionaries
            The dataset to be evaluated. Each dictionary must follow
            the dictionary layout described in help(ssm)

        """
        self.init_state_distn.initialize(dataset)
        self.transitions.initialize(dataset)
        self.observations.initialize(dataset)

    def permute(self, perm):
        """
        Permute the discrete latent states.

        Parameters
        ----------

        perm : array_like (K,) (int)
            A permutation of the K discrete states

        """
        assert np.all(np.sort(perm) == np.arange(self.num_states))
        self.init_state_distn.permute(perm)
        self.transitions.permute(perm)
        self.observations.permute(perm)

    def sample(self, T, prefix=None, input=None, tag=None, with_noise=True):
        """
        Sample synthetic data from the model. Optionally, condition on a given
        prefix (preceding discrete states and data).

        Parameters
        ----------
        T : int
            number of time steps to sample

        prefix : (zpre, xpre)
            Optional prefix of discrete states (zpre) and continuous states (xpre)
            zpre must be an array of integers taking values 0...num_states-1.
            xpre must be an array of the same length that has preceding observations.

        input : (T, input_dim) array_like
            Optional inputs to specify for sampling

        tag : object
            Optional tag indicating which "type" of sampled data

        with_noise : bool
            Whether or not to sample data with noise.

        Returns
        -------
        z_sample : array_like of type int
            Sequence of sampled discrete states

        x_sample : (T x observation_dim) array_like
            Array of sampled data
        """
        K = self.num_states
        D = (self.observation_dim,) if isinstance(self.observation_dim, int) else self.observation_dim
        M = (self.input_dim,) if isinstance(self.input_dim, int) else self.input_dim
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)
        assert T > 0

        # Check the inputs
        if input is not None:
            assert input.shape == (T,) + M

        # Get the type of the observations
        dummy_data = self.observations.sample_observation(0, np.empty(0,) + D)
        dtype = dummy_data.dtype

        # Initialize the data array
        if prefix is None:
            # No prefix is given.  Sample the initial state as the prefix.
            pad = 1
            z = np.zeros(T, dtype=int)
            data = np.zeros((T,) + D, dtype=dtype)
            input = np.zeros((T,) + M) if input is None else input
            mask = np.ones((T,) + D, dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = np.exp(self.init_state_distn.log_initial_state_distn(data))
            z[0] = npr.choice(self.num_states, p=pi0)
            data[0] = self.observations.sample_observation(z[0], data[:0], input=input[0], with_noise=with_noise)

            # We only need to sample T-1 datapoints now
            T = T - 1

        else:
            # Check that the prefix is of the right type
            zpre, xpre = prefix
            pad = len(zpre)
            assert zpre.dtype == int and zpre.min() >= 0 and zpre.max() < K
            assert xpre.shape == (pad,) + D

            # Construct the states, data, inputs, and mask arrays
            z = np.concatenate((zpre, np.zeros(T, dtype=int)))
            data = np.concatenate((xpre, np.zeros((T,) + D, dtype)))
            input = np.zeros((T+pad,) + M) if input is None else np.concatenate((np.zeros((pad,) + M), input))
            mask = np.ones((T+pad,) + D, dtype=bool)

        # Fill in the rest of the data
        for t in range(pad, pad+T):
            dummy_data = dict(data=data[t-1:t+1], input=input[t-1:t+1], mask=mask[t-1:t+1])
            Pt = np.exp(self.transitions.log_transition_matrices(dummy_data))[0]
            z[t] = npr.choice(self.num_states, p=Pt[z[t-1]])
            data[t] = self.observations.sample_observation(z[t], data[:t], input=input[t], tag=tag, with_noise=with_noise)

        # Return the whole data if no prefix is given.
        # Otherwise, just return the simulated part.
        if prefix is None:
            return z, data
        else:
            return z[pad:], data[pad:]

    # def sample_new(self, T, prefix=None, covariates=None, with_noise=True):
    #     """
    #     Sample synthetic data from the model. Optionally, condition on a given
    #     prefix (preceding discrete states and data).

    #     Parameters
    #     ----------
    #     T : int
    #         number of time steps to sample

    #     prefix : (zpre, xpre)
    #         Optional prefix of discrete states (zpre) and continuous states (xpre)
    #         zpre must be an array of integers taking values 0...num_states-1.
    #         xpre must be an array of the same length that has preceding observations.

    #     covariates : dict
    #         Dictionary of covariates containing keys like 'input' and 'tag'.
    #         If given, the input must be of length T.
    #         See help(ssm) for more information.

    #     with_noise : bool
    #         Whether or not to sample data with noise.

    #     Returns
    #     -------
    #     z_sample : array_like of type int
    #         Sequence of sampled discrete states

    #     x_sample : (T x observation_dim) array_like
    #         Array of sampled data
    #     """
    #     K = self.num_states
    #     D = (self.observation_dim,) if isinstance(self.observation_dim, int) else self.observation_dim
    #     M = (self.input_dim,) if isinstance(self.input_dim, int) else self.input_dim
    #     assert isinstance(D, tuple)
    #     assert isinstance(M, tuple)
    #     assert T > 0

    #     # Check the inputs
    #     if input is not None:
    #         assert input.shape == (T,) + M

    #     # Get the type of the observations
    #     dummy_data = self.observations.sample_observation(0, np.empty(0,) + D)
    #     dtype = dummy_data.dtype

    #     # Initialize the data array
    #     if prefix is None:
    #         # No prefix is given.  Sample the initial state as the prefix.
    #         pad = 1
    #         states = np.zeros(T, dtype=int)
    #         data = np.zeros((T,) + D, dtype=dtype)
    #         input = np.zeros((T,) + M) if input is None else input
    #         mask = np.ones((T,) + D, dtype=bool)

    #         # Sample the first state from the initial distribution
    #         pi0 = np.exp(self.init_state_distn.log_initial_state_distn(data))
    #         z[0] = npr.choice(self.num_states, p=pi0)
    #         data[0] = self.observations.sample_observation(states[0], data[:0], input=input[0], with_noise=with_noise)

    #         # We only need to sample T-1 datapoints now
    #         T = T - 1

    #     else:
    #         # Check that the prefix is of the right type
    #         zpre, xpre = prefix
    #         pad = len(zpre)
    #         assert zpre.dtype == int and zpre.min() >= 0 and zpre.max() < K
    #         assert xpre.shape == (pad,) + D

    #         # Construct the states, data, inputs, and mask arrays
    #         z = np.concatenate((zpre, np.zeros(T, dtype=int)))
    #         data = np.concatenate((xpre, np.zeros((T,) + D, dtype)))
    #         input = np.zeros((T+pad,) + M) if input is None else np.concatenate((np.zeros((pad,) + M), input))
    #         mask = np.ones((T+pad,) + D, dtype=bool)

    #     # Fill in the rest of the data
    #     for t in range(pad, pad+T):
    #         dummy_data = dict(data=data[t-1:t+1], input=input[t-1:t+1], mask=mask[t-1:t+1])
    #         Pt = np.exp(self.transitions.log_transition_matrices(dummy_data))[0]
    #         z[t] = npr.choice(self.num_states, p=Pt[z[t-1]])
    #         data[t] = self.observations.sample_observation(z[t], data[:t], input=input[t], tag=tag, with_noise=with_noise)

    #     # Return the whole data if no prefix is given.
    #     # Otherwise, just return the simulated part.
    #     if prefix is None:
    #         return z, data
    #     else:
    #         return z[pad:], data[pad:]


    @check_data
    def expected_states(self, data):
        """
        Compute the posterior distribution of discrete states given
        observed data, p(z | x).

        Parameters
        ----------

        data : dictionary.
            Contains the data and any covariates. Must follow
            the dictionary layout described in help(ssm)

        """
        log_pi0 = self.init_state_distn.log_initial_state_distn(data)
        log_Ps = self.transitions.log_transition_matrices(data)
        log_likes = self.observations.log_likelihoods(data)
        return hmm_expected_states(log_pi0, log_Ps, log_likes)

    @check_data
    def most_likely_states(self, data):
        """
        Compute the most likely sequence of discrete states given
        observed data, z* = argmax p(z | x).

        Parameters
        ----------

        data : dictionary.
            Contains the data and any covariates. Must follow
            the dictionary layout described in help(ssm)

        """
        log_pi0 = self.init_state_distn.log_initial_state_distn(data)
        log_Ps = self.transitions.log_transition_matrices(data)
        log_likes = self.observations.log_likelihoods(data)
        return viterbi(log_pi0, log_Ps, log_likes)

    @check_data
    def filter(self, data):
        """
        Compute the filtered posterior distribution of discrete states given
        observed data, p(z_t | x_{1:t}).

        Parameters
        ----------

        data : dictionary.
            Contains the data and any covariates. Must follow
            the dictionary layout described in help(ssm)

        """
        log_pi0 = self.init_state_distn.log_initial_state_distn(data)
        log_Ps = self.transitions.log_transition_matrices(data)
        log_likes = self.observations.log_likelihoods(data)
        return hmm_filter(log_pi0, log_Ps, log_likes)

    @check_data
    def smooth(self, data):
        """
        Compute the smoothed observations, taking an expectation over
        the posterior distribution of the discrete states,
        E_{p(z | x)}[x | z].

        Parameters
        ----------

        data : dictionary.
            Contains the data and any covariates. Must follow
            the dictionary layout described in help(ssm)

        """
        Ez, _, _ = self.expected_states(data)
        return self.observations.smooth(Ez, data)

    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """
        return self.init_state_distn.log_prior() + \
               self.transitions.log_prior() + \
               self.observations.log_prior()

    @check_dataset
    def log_likelihood(self, dataset):
        """
        Compute the log likelihood of a dataset under the current
        model parameters.

        Parameters
        ----------

        dataset : dictionary or list of dictionaries
            The dataset to be evaluated. Each dictionary must follow
            the dictionary layout described in help(ssm)

        Returns
        -------
        ll : double
            Log likelihood of the entire dataset.
        """
        ll = 0
        for data in dataset:
            log_pi0 = self.init_state_distn.log_initial_state_distn(data)
            log_Ps = self.transitions.log_transition_matrices(data)
            log_likes = self.observations.log_likelihoods(data)
            ll += hmm_normalizer(log_pi0, log_Ps, log_likes)
            assert np.isfinite(ll)
        return ll

    @check_dataset
    def log_probability(self, dataset):
        """
        Compute the log probability (likelihood plus prior) of a dataset
        under the current model parameters.

        Parameters
        ----------

        dataset : dictionary or list of dictionaries
            The dataset to be evaluated. Each dictionary must follow
            the dictionary layout described in help(ssm)

        Returns
        -------
        lp : double
            Log probability of the entire dataset.
        """
        return self.log_likelihood(dataset) + self.log_prior()

    def expected_log_probability(self, expectations, dataset):
        """
        Compute the expected log probability of a dataset under the current
        model parameters and given expectations.

        Parameters
        ----------

        expectations : list or array_like (time_bins, num_states)
            Posterior distribution over discrete states p(z | x) for
            each data in the dataset.  Obtained from hmm.expected_states().

        dataset : dictionary or list of dictionaries
            The dataset to be evaluated. Each dictionary must follow
            the dictionary layout described in help(ssm)

        Returns
        -------
        elp : double
            Expected log probability of the entire dataset.
        """
        elp = self.log_prior()
        for (Ez, Ezzp1, _), data in zip(expectations, dataset):
            log_pi0 = self.init_state_distn.log_initial_state_distn(data)
            log_Ps = self.transitions.log_transition_matrices(data)
            log_likes = self.observations.log_likelihoods(data)

            # Compute the expected log probability
            elp += np.sum(Ez[0] * log_pi0)
            elp += np.sum(Ezzp1 * log_Ps)
            elp += np.sum(Ez * log_likes)
            assert np.isfinite(elp)
        return elp

    # Model fitting
    def _fit_sgd(self, optimizer, dataset, num_iters=1000, **kwargs):
        """
        Fit the model with maximum marginal likelihood.
        """
        T = sum([data['data'].shape[0] for data in dataset])

        def _objective(params, itr):
            self.params = params
            obj = self.log_probability(dataset)
            return -obj / T

        # Set up the progress bar
        lls = [-_objective(self.params, 0) * T]
        pbar = trange(num_iters)
        pbar.set_description("Epoch {} Itr {} LP: {:.1f}".format(0, 0, lls[-1]))

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            self.params, val, g, state = step(value_and_grad(_objective), self.params, itr, state, **kwargs)
            lls.append(-val * T)
            pbar.set_description("LP: {:.1f}".format(lls[-1]))
            pbar.update(1)

        return lls

    def _fit_stochastic_em(self, optimizer, dataset, num_epochs=100, **kwargs):
        """
        Replace the M-step of EM with a stochastic gradient update using the ELBO computed
        on a minibatch of data.
        """
        M = len(dataset)
        T = sum([data['data'].shape[0] for data in dataset])

        # A helper to grab a minibatch of data
        perm = [np.random.permutation(M) for _ in range(num_epochs)]
        def _get_minibatch(itr):
            epoch = itr // M
            m = itr % M
            i = perm[epoch][m]
            return dataset[i]

        # Define the objective (negative ELBO)
        def _objective(params, itr):
            # Grab a minibatch of data
            data = _get_minibatch(itr)
            Ti = data['data'].shape[0]

            # E step: compute expected latent states with current parameters
            Ez, Ezzp1, _ = self.expected_states(data)

            # M step: set the parameter and compute the (normalized) objective function
            self.params = params
            log_pi0 = self.init_state_distn.log_initial_state_distn(data)
            log_Ps = self.transitions.log_transition_matrices(data)
            log_likes = self.observations.log_likelihoods(data)

            # Compute the expected log probability
            # (Scale by number of length of this minibatch.)
            obj = self.log_prior()
            obj += np.sum(Ez[0] * log_pi0) * M
            obj += np.sum(Ezzp1 * log_Ps) * (T - M) / (Ti - 1)
            obj += np.sum(Ez * log_likes) * T / Ti
            assert np.isfinite(obj)

            return -obj / T

        # Set up the progress bar
        lls = [-_objective(self.params, 0) * T]
        pbar = trange(num_epochs * M)
        pbar.set_description("Epoch {} Itr {} LP: {:.1f}".format(0, 0, lls[-1]))

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            self.params, val, g, state = step(value_and_grad(_objective), self.params, itr, state, **kwargs)
            epoch = itr // M
            m = itr % M
            lls.append(-val * T)
            pbar.set_description("Epoch {} Itr {} LP: {:.1f}".format(epoch, m, lls[-1]))
            pbar.update(1)

        return lls

    def _fit_em(self, dataset, num_em_iters=100, **kwargs):
        """
        Fit the parameters with expectation maximization.

        E step: compute E[z_t] and E[z_t, z_{t+1}] with message passing;
        M-step: analytical maximization of E_{p(z | x)} [log p(x, z; theta)].
        """
        lls = [self.log_probability(dataset)]

        pbar = trange(num_em_iters)
        pbar.set_description("LP: {:.1f}".format(lls[-1]))
        for itr in pbar:
            # E step: compute expected latent states with current parameters
            expectations = [self.expected_states(data) for data in dataset]

            # M step: maximize expected log joint wrt parameters
            self.init_state_distn.m_step(expectations, dataset, **kwargs)
            self.transitions.m_step(expectations, dataset, **kwargs)
            self.observations.m_step(expectations, dataset, **kwargs)

            # Store progress
            lls.append(self.log_prior() + sum([ll for (_, _, ll) in expectations]))
            pbar.set_description("LP: {:.1f}".format(lls[-1]))

        return lls

    @check_dataset
    def fit(self, dataset, method="em", initialize=True, **kwargs):
        _fitting_methods = \
            dict(sgd=partial(self._fit_sgd, "sgd"),
                 adam=partial(self._fit_sgd, "adam"),
                 em=self._fit_em,
                 stochastic_em=partial(self._fit_stochastic_em, "adam"),
                 stochastic_em_sgd=partial(self._fit_stochastic_em, "sgd"),
                 )

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        if initialize:
            self.initialize(dataset)

        return _fitting_methods[method](dataset, **kwargs)


class BaseHSMM(BaseHMM):
    """
    Hidden semi-Markov model with non-geometric duration distributions.
    The trick is to expand the state space with "super states" and "sub states"
    that effectively count duration. We rely on the transition model to
    specify a "state map," which maps the super states (1, .., K) to
    super+sub states ((1,1), ..., (1,r_1), ..., (K,1), ..., (K,r_K)).
    Here, r_k denotes the number of sub-states of state k.
    """
    @property
    def state_map(self):
        return self.transitions.state_map

    def sample(self, T, prefix=None, input=None, tag=None, with_noise=True):
        """
        Sample synthetic data from the model. Optionally, condition on a given
        prefix (preceding discrete states and data).

        Parameters
        ----------
        T : int
            number of time steps to sample

        prefix : (zpre, xpre)
            Optional prefix of discrete states (zpre) and continuous states (xpre)
            zpre must be an array of integers taking values 0...num_states-1.
            xpre must be an array of the same length that has preceding observations.

        input : (T, input_dim) array_like
            Optional inputs to specify for sampling

        tag : object
            Optional tag indicating which "type" of sampled data

        with_noise : bool
            Whether or not to sample data with noise.

        Returns
        -------
        z_sample : array_like of type int
            Sequence of sampled discrete states

        x_sample : (T x observation_dim) array_like
            Array of sampled data
        """
        K = self.num_states
        D = (self.observation_dim,) if isinstance(self.observation_dim, int) else self.observation_dim
        M = (self.input_dim,) if isinstance(self.input_dim, int) else self.input_dim
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)
        assert T > 0

        # Check the inputs
        if input is not None:
            assert input.shape == (T,) + M

        # Get the type of the observations
        dummy_data = self.observations.sample_observation(0, np.empty(0,) + D)
        dtype = dummy_data.dtype

        # Initialize the data array
        if prefix is None:
            # No prefix is given.  Sample the initial state as the prefix.
            pad = 1
            z = np.zeros(T, dtype=int)
            data = np.zeros((T,) + D, dtype=dtype)
            input = np.zeros((T,) + M) if input is None else input
            mask = np.ones((T,) + D, dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = np.exp(self.init_state_distn.log_initial_state_distn(data, input, mask, tag))
            z[0] = npr.choice(self.num_states, p=pi0)
            data[0] = self.observations.sample_observation(z[0], data[:0], input=input[0], with_noise=with_noise)

            # We only need to sample T-1 datapoints now
            T = T - 1

        else:
            # Check that the prefix is of the right type
            zpre, xpre = prefix
            pad = len(zpre)
            assert zpre.dtype == int and zpre.min() >= 0 and zpre.max() < K
            assert xpre.shape == (pad,) + D

            # Construct the states, data, inputs, and mask arrays
            z = np.concatenate((zpre, np.zeros(T, dtype=int)))
            data = np.concatenate((xpre, np.zeros((T,) + D, dtype)))
            input = np.zeros((T+pad,) + M) if input is None else np.concatenate((np.zeros((pad,) + M), input))
            mask = np.ones((T+pad,) + D, dtype=bool)

        # Convert the discrete states to the range (1, ..., K_total)
        m = self.state_map
        K_total = len(m)
        _, starts = np.unique(m, return_index=True)
        z = starts[z]

        # Fill in the rest of the data
        for t in range(pad, pad+T):
            Pt = np.exp(self.transitions.log_transition_matrices(data[t-1:t+1], input[t-1:t+1], mask=mask[t-1:t+1], tag=tag))[0]
            z[t] = npr.choice(K_total, p=Pt[z[t-1]])
            data[t] = self.observations.sample_observation(m[z[t]], data[:t], input=input[t], tag=tag, with_noise=with_noise)

        # Collapse the states
        z = m[z]

        # Return the whole data if no prefix is given.
        # Otherwise, just return the simulated part.
        if prefix is None:
            return z, data
        else:
            return z[pad:], data[pad:]

    @check_data
    def expected_states(self, data):
        m = self.state_map
        log_pi0 = self.init_state_distn.log_initial_state_distn(data)
        log_Ps = self.transitions.log_transition_matrices(data)
        log_likes = self.observations.log_likelihoods(data)
        Ez, Ezzp1, normalizer = hmm_expected_states(replicate(log_pi0, m), log_Ps, replicate(log_likes, m))

        # Collapse the expected states
        Ez = collapse(Ez, m)
        Ezzp1 = collapse(collapse(Ezzp1, m, axis=2), m, axis=1)
        return Ez, Ezzp1, normalizer

    @check_data
    def most_likely_states(self, data):
        m = self.state_map
        log_pi0 = self.init_state_distn.log_initial_state_distn(data)
        log_Ps = self.transitions.log_transition_matrices(data)
        log_likes = self.observations.log_likelihoods(data)
        z_star = viterbi(replicate(log_pi0, m), log_Ps, replicate(log_likes, m))
        return self.state_map[z_star]

    @check_data
    def filter(self, data):
        m = self.state_map
        log_pi0 = self.init_state_distn.log_initial_state_distn(data)
        log_Ps = self.transitions.log_transition_matrices(data)
        log_likes = self.observations.log_likelihoods(data)
        pzp1 = hmm_filter(replicate(log_pi0, m), log_Ps, replicate(log_likes, m))
        return collapse(pzp1, m)

    @check_data
    def posterior_sample(self, data):
        m = self.state_map
        log_pi0 = self.init_state_distn.log_initial_state_distn(data)
        log_Ps = self.transitions.log_transition_matrices(data)
        log_likes = self.observations.log_likelihoods(data)
        z_smpl = hmm_sample(replicate(log_pi0, m), log_Ps, replicate(log_likes, m))
        return self.state_map[z_smpl]

    @check_data
    def smooth(self, data):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        m = self.state_map
        Ez, _, _ = self.expected_states(data)
        return self.observations.smooth(Ez, data)

    @check_dataset
    def log_likelihood(self, dataset):
        """
        Compute the log probability of the data under the current
        model parameters.

        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        m = self.state_map
        ll = 0
        for data in dataset:
            log_pi0 = self.init_state_distn.log_initial_state_distn(data)
            log_Ps = self.transitions.log_transition_matrices(data)
            log_likes = self.observations.log_likelihoods(data)
            ll += hmm_normalizer(replicate(log_pi0, m), log_Ps, replicate(log_likes, m))
            assert np.isfinite(ll)
        return ll

    def expected_log_probability(self, expectations, dataset):
        """
        Compute the log probability of the data under the current
        model parameters.

        :param datas: single array or list of arrays of data.
        :return total log probability of the data.
        """
        raise NotImplementedError("Need to get raw expectations for the expected transition probability.")

    def _fit_em(self, dataset, num_em_iters=100, **kwargs):
        """
        Fit the parameters with expectation maximization.

        E step: compute E[z_t] and E[z_t, z_{t+1}] with message passing;
        M-step: analytical maximization of E_{p(z | x)} [log p(x, z; theta)].
        """
        lls = [self.log_probability(dataset)]

        pbar = trange(num_em_iters)
        pbar.set_description("LP: {:.1f}".format(lls[-1]))
        for itr in pbar:
            # E step: compute expected latent states with current parameters
            expectations = [self.expected_states(data) for data in dataset]

            # Unique to HSMM: also sample the posterior for stochastic M step
            #                 of transition model
            samples = [self.posterior_sample(data) for data in dataset]

            # M step: maximize expected log joint wrt parameters
            self.init_state_distn.m_step(expectations, dataset, **kwargs)
            self.transitions.m_step(expectations, dataset, samples, **kwargs)
            self.observations.m_step(expectations, dataset, **kwargs)

            # Store progress
            lls.append(self.log_prior() + sum([ll for (_, _, ll) in expectations]))
            pbar.set_description("LP: {:.1f}".format(lls[-1]))

        return lls

    @check_dataset
    def fit(self, dataset, method="em", initialize=True, **kwargs):
        _fitting_methods = dict(em=self._fit_em)

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        if initialize:
            self.initialize(dataset)

        return _fitting_methods[method](dataset, **kwargs)


class BaseSwitchingLDS(object):
    """
    Switching linear dynamical system fit with
    stochastic variational inference on the marginal model,
    integrating out the discrete states.
    """
    def __init__(self, observation_dim, num_states, latent_dim,
                 init_state_distn, transitions, dynamics, emissions,
                 input_dim=0):
        """
        Construct a switching linear dynamical system (SLDS).

        Parameters
        ----------
        observation_dim : int > 0 or tuple of ints > 0
            The dimensionality of the observed data

        num_states : int > 0
            The number of discrete states

        latent_dim : int > 0
            The dimensionality of the continuous latent states

        init_state_distn : InitialStateDistribution object
            Object encapsulating the initial state distribution
            p(z_1)

        transitions : _Transitions object
            Object encapsulating the transition distribution
            p(z_t | z_{t-1}, ...)

        dynamics : _AutoRegressiveObservationsBase object
            Object encapsulating the dynamics distribution
            p(x_t | x_{t-1}, z_t, ...)

        emissions : _Emissions object
            Object encapsulating the emissions distribution
            p(y_t | x_t, z_t, ...)

        input_dim : int >= 0
            Dimensionality of the external inputs, optional.
        """
        self.observation_dim = observation_dim
        self.num_states = num_states
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.init_state_distn = init_state_distn
        self.transitions = transitions
        self.dynamics = dynamics
        self.emissions = emissions

    @property
    def params(self):
        return self.init_state_distn.params, \
               self.transitions.params, \
               self.dynamics.params, \
               self.emissions.params

    @params.setter
    def params(self, value):
        self.init_state_distn.params = value[0]
        self.transitions.params = value[1]
        self.dynamics.params = value[2]
        self.emissions.params = value[3]

    @check_dataset
    def initialize(self, dataset, num_em_iters=25):
        # First initialize the observation model
        self.emissions.initialize(dataset)

        # Get the initialized variational mean for the data
        latent_states = [self.emissions.invert(data) for data in dataset]
        latent_masks = [np.ones_like(states, dtype=bool) for states in latent_states]

        # Now run a few iterations of EM on a ARHMM with the variational mean
        print("Initializing with an ARHMM using {} steps of EM.".format(num_em_iters))
        arhmm = BaseHMM(self.num_states, self.latent_dim,
                        copy.deepcopy(self.init_state_distn),
                        copy.deepcopy(self.transitions),
                        copy.deepcopy(self.dynamics),
                        input_dim=self.input_dim)

        # Construct a dataset with the latent states as observations
        dummy_dataset = copy.deepcopy(dataset)
        for data, states, mask in zip(dummy_dataset, latent_states, latent_masks):
            data['data'] = states
            data['mask'] = mask

        # Fit the ARHMM and copy its parameters
        arhmm.fit(dummy_dataset, method="em", num_em_iters=num_em_iters, num_iters=10)
        self.init_state_distn = copy.deepcopy(arhmm.init_state_distn)
        self.transitions = copy.deepcopy(arhmm.transitions)
        self.dynamics = copy.deepcopy(arhmm.observations)

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        assert np.all(np.sort(perm) == np.arange(self.num_states))
        self.init_state_distn.permute(perm)
        self.transitions.permute(perm)
        self.dynamics.permute(perm)
        self.emissions.permute(perm)

    def log_prior(self):
        """
        Compute the log prior probability of the model parameters
        """
        return self.init_state_distn.log_prior() + \
               self.transitions.log_prior() + \
               self.dynamics.log_prior() + \
               self.emissions.log_prior()

    def sample(self, T, input=None, tag=None, prefix=None, with_noise=True):
        K = self.num_states
        D = (self.latent_dim,) if isinstance(self.latent_dim, int) else self.latent_dim
        M = (self.input_dim,) if isinstance(self.input_dim, int) else self.input_dim
        assert isinstance(D, tuple)
        assert isinstance(M, tuple)

        # If prefix is given, pad the output with it
        if prefix is None:
            pad = 1
            z = np.zeros(T+1, dtype=int)
            x = np.zeros((T+1,) + D)
            data = np.zeros((T+1,) + D)
            input = np.zeros((T+1,) + M) if input is None else input
            xmask = np.ones((T+1,) + D, dtype=bool)

            # Sample the first state from the initial distribution
            pi0 = np.exp(self.init_state_distn.log_initial_state_distn(data, input, xmask, tag))
            z[0] = npr.choice(K, p=pi0)
            x[0] = self.dynamics.sample_observation(z[0], x[:0], tag=tag, with_noise=with_noise)

        else:
            zhist, xhist, yhist = prefix
            pad = len(zhist)
            assert zhist.dtype == int and zhist.min() >= 0 and zhist.max() < K
            assert xhist.shape == (pad, D)
            assert yhist.shape == (pad, N)

            z = np.concatenate((zhist, np.zeros(T, dtype=int)))
            x = np.concatenate((xhist, np.zeros((T,) + D)))
            input = np.zeros((T+pad,) + M) if input is None else input
            xmask = np.ones((T+pad,) + D, dtype=bool)

        # Sample z and x
        for t in range(pad, T+pad):
            Pt = np.exp(self.transitions.log_transition_matrices(x[t-1:t+1], input[t-1:t+1], mask=xmask[t-1:t+1], tag=tag))[0]
            z[t] = npr.choice(self.num_states, p=Pt[z[t-1]])
            x[t] = self.dynamics.sample_observation(z[t], x[:t], input=input[t], tag=tag, with_noise=with_noise)

        # Sample observations given latent states
        # TODO: sample in the loop above?
        y = self.emissions.sample(z, x, input=input, tag=tag)
        return z[pad:], x[pad:], y[pad:]

    @check_slds_args
    def expected_states(self, variational_mean, data):
        dummy_data = copy.copy(data)
        dummy_data['data'] = variational_mean
        dummy_data['mask'] = np.ones_like(variational_mean, dtype=bool)

        log_pi0 = self.init_state_distn.log_initial_state_distn(dummy_data)
        log_Ps = self.transitions.log_transition_matrices(dummy_data)
        log_likes = self.dynamics.log_likelihoods(dummy_data)
        log_likes += self.emissions.log_likelihoods(data, variational_mean)
        return hmm_expected_states(log_pi0, log_Ps, log_likes)

    @check_slds_args
    def most_likely_states(self, variational_mean, data):
        dummy_data = copy.copy(data)
        dummy_data['data'] = variational_mean
        dummy_data['mask'] = np.ones_like(variational_mean, dtype=bool)

        log_pi0 = self.init_state_distn.log_initial_state_distn(dummy_data)
        log_Ps = self.transitions.log_transition_matrices(dummy_data)
        log_likes = self.dynamics.log_likelihoods(dummy_data)
        log_likes += self.emissions.log_likelihoods(data, variational_mean)
        return viterbi(log_pi0, log_Ps, log_likes)

    @check_slds_args
    def smooth(self, variational_mean, data):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        Ez, _, _ = self.expected_states(variational_mean, data)
        return self.emissions.smooth(Ez, variational_mean, data)

    @check_dataset
    def log_probability(self, dataset):
        warnings.warn("Cannot compute exact marginal log probability for the SLDS. "
                      "the ELBO instead.")
        return np.nan

    @check_variational_args
    def elbo(self, variational_posterior, dataset, n_samples=1):
        """
        Lower bound on the marginal likelihood p(y | theta)
        using variational posterior q(x; phi) where phi = variational_params
        """
        elbo = 0
        for sample in range(n_samples):
            # Sample x from the variational posterior
            latent_states = variational_posterior.sample()

            # log p(theta)
            elbo += self.log_prior()

            # log p(x, y | theta) = log \sum_z p(x, y, z | theta)
            for states, data in zip(latent_states, dataset):
                dummy_data = copy.copy(data)
                dummy_data['data'] = states
                dummy_data['mask'] = np.ones_like(states, dtype=bool)

                log_pi0 = self.init_state_distn.log_initial_state_distn(dummy_data)
                log_Ps = self.transitions.log_transition_matrices(dummy_data)
                log_likes = self.dynamics.log_likelihoods(dummy_data)
                log_likes += self.emissions.log_likelihoods(data, states)
                elbo += hmm_normalizer(log_pi0, log_Ps, log_likes)

            # -log q(x)
            elbo -= variational_posterior.log_density(latent_states)
            assert np.isfinite(elbo)

        return elbo / n_samples

    @check_variational_args
    def _surrogate_elbo(self, variational_posterior, dataset, alpha=0.75, **kwargs):
        """
        Lower bound on the marginal likelihood p(y | gamma)
        using variational posterior q(x; phi) where phi = variational_params
        and gamma = emission parameters.  As part of computing this objective,
        we optimize q(z | x) and take a natural gradient step wrt theta, the
        parameters of the dynamics model.

        Note that the surrogate ELBO is a lower bound on the ELBO above.
           E_p(z | x, y)[log p(z, x, y)]
           = E_p(z | x, y)[log p(z, x, y) - log p(z | x, y) + log p(z | x, y)]
           = E_p(z | x, y)[log p(x, y) + log p(z | x, y)]
           = log p(x, y) + E_p(z | x, y)[log p(z | x, y)]
           = log p(x, y) -H[p(z | x, y)]
          <= log p(x, y)
        with equality only when p(z | x, y) is atomic.  The gap equals the
        entropy of the posterior on z.
        """
        # log p(theta)
        elbo = self.log_prior()

        # Sample x from the variational posterior
        latent_states = variational_posterior.sample()

        # Inner optimization: find the true posterior p(z | x, y; theta).
        # Then maximize the inner ELBO wrt theta,
        #
        #    E_p(z | x, y; theta_fixed)[log p(z, x, y; theta).
        #
        # This can be seen as a natural gradient step in theta
        # space.  Note: we do not want to compute gradients wrt x or the
        # emissions parameters backward throgh this optimization step,
        # so we unbox them first.
        latent_states_unboxed = [getval(states) for states in latent_states]
        emission_params_boxed = self.emissions.params
        flat_emission_params_boxed, unflatten = flatten(emission_params_boxed)
        self.emissions.params = unflatten(getval(flat_emission_params_boxed))

        # E step: compute the true posterior p(z | x, y, theta_fixed) and
        # the necessary expectations under this posterior.
        expectations = [self.expected_states(states, data) for states, data in
                        zip(latent_states_unboxed, dataset)]

        # M step: maximize expected log joint wrt parameters
        # Note: Only do a partial update toward the M step for this sample of xs
        latent_masks = [np.ones_like(states, dtype=bool) for states in latent_states_unboxed]
        for distn in [self.init_state_distn, self.transitions, self.dynamics]:
            curr_prms = copy.deepcopy(distn.params)
            distn.m_step(expectations, latent_states_unboxed, **kwargs)
            distn.params = convex_combination(curr_prms, distn.params, alpha)

        # Box up the emission parameters again before computing the ELBO
        self.emissions.params = emission_params_boxed

        # Compute expected log likelihood E_q(z | x, y) [log p(z, x, y; theta)]
        for (Ez, Ezzp1, _), states, data in zip(expectations, latent_states, dataset):
            dummy_data = copy.copy(data)
            dummy_data['data'] = states
            dummy_data['mask'] = np.ones_like(states, dtype=bool)

            # Compute expected log likelihood (inner ELBO)
            log_pi0 = self.init_state_distn.log_initial_state_distn(dummy_data)
            log_Ps = self.transitions.log_transition_matrices(dummy_data)
            log_likes = self.dynamics.log_likelihoods(dummy_data)
            log_likes += self.emissions.log_likelihoods(data, states)

            elbo += np.sum(Ez[0] * log_pi0)
            elbo += np.sum(Ezzp1 * log_Ps)
            elbo += np.sum(Ez * log_likes)

        # -log q(x)
        elbo -= variational_posterior.log_density(latent_states)
        assert np.isfinite(elbo)

        return elbo

    def _fit_svi(self, variational_posterior, dataset,
                 learning=True, optimizer="adam", num_iters=100,
                 **kwargs):
        """
        Fit with stochastic variational inference using a
        mean field Gaussian approximation for the latent states x_{1:T}.
        """
        # Define the objective (negative ELBO)
        T = sum([data.shape[0] for data in dataset])
        def _objective(params, itr):
            if learning:
                self.params, variational_posterior.params = params
            else:
                variational_posterior.params = params

            obj = self.elbo(variational_posterior, dataset)
            return -obj / T

        # Initialize the parameters
        if learning:
            params = (self.params, variational_posterior.params)
        else:
            params = variational_posterior.params

        # Set up the progress bar
        elbos = [-_objective(params, 0) * T]
        pbar = trange(num_iters)
        pbar.set_description("ELBO: {:.1f}".format(elbos[0]))

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            params, val, g, state = step(value_and_grad(_objective), params, itr, state)
            elbos.append(-val * T)

            # TODO: Check for convergence -- early stopping

            # Update progress bar
            pbar.set_description("ELBO: {:.1f}".format(elbos[-1]))
            pbar.update()

        # Save the final parameters
        if learning:
            self.params, variational_posterior.params = params
        else:
            variational_posterior.params = params

        return elbos

    def _fit_variational_em(self, variational_posterior, dataset,
                            learning=True, alpha=.75, optimizer="adam",
                            num_iters=100, **kwargs):
        """
        Let gamma denote the emission parameters and theta denote the transition
        and initial discrete state parameters. This is a mix of EM and SVI:
            1. Sample x ~ q(x; phi)
            2. Compute L(x, theta') E_p(z | x, theta)[log p(x, z; theta')]
            3. Set theta = (1 - alpha) theta + alpha * argmax L(x, theta')
            4. Set gamma = gamma + eps * nabla log p(y | x; gamma)
            5. Set phi = phi + eps * dx/dphi * d/dx [L(x, theta) + log p(y | x; gamma) - log q(x; phi)]
        """
        # Optimize the standard ELBO when updating gamma (emissions params)
        # and phi (variational params)
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            if learning:
                self.emissions.params, variational_posterior.params = params
            else:
                variational_posterior.params = params

            obj = self._surrogate_elbo(variational_posterior, dataset, **kwargs)
            return -obj / T

        # Initialize the parameters
        if learning:
            params = (self.emissions.params, variational_posterior.params)
        else:
            params = variational_posterior.params

        # Set up the progress bar
        elbos = [-_objective(params, 0) * T]
        pbar = trange(num_iters)
        pbar.set_description("Surrogate ELBO: {:.1f}".format(elbos[0]))

        # Run the optimizer
        step = dict(sgd=sgd_step, rmsprop=rmsprop_step, adam=adam_step)[optimizer]
        state = None
        for itr in pbar:
            # Update the emission and variational posterior parameters
            params, val, g, state = step(value_and_grad(_objective), params, itr, state)
            elbos.append(-val * T)

            # Update progress bar
            pbar.set_description("Surrogate ELBO: {:.1f}".format(elbos[-1]))
            pbar.update()

        # Save the final emission and variational parameters
        if learning:
            self.emissions.params, variational_posterior.params = params
        else:
            variational_posterior.params = params

        return elbos

    @check_variational_args
    def fit(self, variational_posterior, dataset, method="svi", initialize=True, **kwargs):

        # Specify fitting methods
        _fitting_methods = dict(svi=self._fit_svi,
                                vem=self._fit_variational_em)

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        if initialize:
            self.initialize(dataset)

        _fitting_methods[method](variational_posterior, dataset, learning=True, **kwargs)

    @check_variational_args
    def approximate_posterior(self, variational_posterior, dataset, method="svi", **kwargs):
        # Specify fitting methods
        _fitting_methods = dict(svi=self._fit_svi,
                                vem=self._fit_variational_em)

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        return _fitting_methods[method](variational_posterior, dataset, learning=False, **kwargs)


class BaseLDS(BaseSwitchingLDS):
    """
    Switching linear dynamical system fit with
    stochastic variational inference on the marginal model,
    integrating out the discrete states.
    """
    def __init__(self, observation_dim, latent_dim, dynamics, emissions, input_dim=0):
        from ssm.init_state_distns import InitialStateDistribution
        from ssm.transitions import StationaryTransitions
        init_state_distn = InitialStateDistribution(1, D, M)
        transitions = StationaryTransitions(1, D, M)
        super(_LDS, self).__init__(N, 1, D, M, init_state_distn, transitions, dynamics, emissions)

    @check_slds_args
    def expected_states(self, variational_mean, data):
        return np.ones((variational_mean.shape[0], 1)), \
               np.ones((variational_mean.shape[0], 1, 1)), \
               0

    @check_slds_args
    def most_likely_states(self, variational_mean, data):
        raise NotImplementedError

    def log_prior(self):
        return self.dynamics.log_prior() + self.emissions.log_prior()

    @check_dataset
    def log_probability(self, dataset):
        warnings.warn("Log probability of LDS is not yet implemented.")
        return np.nan

    @check_variational_args
    def elbo(self, variational_posterior, dataset, n_samples=1):
        """
        Lower bound on the marginal likelihood p(y | theta)
        using variational posterior q(x; phi) where phi = variational_params
        """
        elbo = 0
        for sample in range(n_samples):
            # Sample x from the variational posterior
            latent_states = variational_posterior.sample()

            # log p(theta)
            elbo += self.log_prior()

            # Compute log p(y, x | theta)
            for states, data in zip(latent_states, dataset):
                dummy_data = copy.copy(data)
                dummy_data['data'] = states
                dummy_data['mask'] = np.ones_like(states, dtype=bool)

                elbo += np.sum(self.dynamics.log_likelihoods(dummy_data))
                elbo += np.sum(self.emissions.log_likelihoods(data, states))

            # -log q(x)
            elbo -= variational_posterior.log_density(latent_states)
            assert np.isfinite(elbo)

        return elbo / n_samples
