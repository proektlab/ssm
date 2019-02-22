from functools import partial
from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.scipy.stats import dirichlet

from ssm.util import one_hot, logistic, relu, rle, \
    fit_multiclass_logistic_regression, \
    fit_negative_binomial_integer_r
from ssm.stats import multivariate_normal_logpdf
from ssm.optimizers import adam, bfgs, rmsprop, sgd


class _Transitions(object):
    def __init__(self, num_states, observation_dim, input_dim=0):
        self.num_states = num_states
        self.observation_dim = observation_dim
        self.input_dim = input_dim

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def initialize(self, dataset):
        pass

    def permute(self, perm):
        pass

    def log_prior(self):
        return 0

    def log_transition_matrices(self, data):
        raise NotImplementedError

    def m_step(self, expectations, dataset, optimizer="bfgs", num_iters=100, **kwargs):
        """
        If M-step cannot be done in closed form for the transitions, default to BFGS.
        """
        optimizer = dict(sgd=sgd, adam=adam, rmsprop=rmsprop, bfgs=bfgs)[optimizer]

        # Maximize the expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, (expected_states, expected_joints, _) in zip(datas, expectations):
                log_Ps = self.log_transition_matrices(data)
                elbo += np.sum(expected_joints * log_Ps)
            return elbo

        # Normalize and negate for minimization
        T = sum([data['data'].shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        # Call the optimizer
        self.params = optimizer(_objective, self.params, num_iters=num_iters, **kwargs)


class StationaryTransitions(_Transitions):
    """
    Standard Hidden Markov Model with fixed initial distribution and transition matrix.
    """
    def __init__(self, num_states, observation_dim, input_dim=0):
        super(StationaryTransitions, self).\
            __init__(num_states, observation_dim, input_dim=input_dim)
        Ps = .95 * np.eye(num_states) + .05 * npr.rand(num_states, num_states)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

    @property
    def params(self):
        return (self.log_Ps,)

    @params.setter
    def params(self, value):
        self.log_Ps = value[0]

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]

    @property
    def transition_matrix(self):
        return np.exp(self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True))

    def log_transition_matrices(self, data):
        T = data['data'].shape[0]
        log_Ps = self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True)
        return np.tile(log_Ps[None, :, :], (T-1, 1, 1))

    def m_step(self, expectations, dataset, **kwargs):
        P = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-16
        P /= P.sum(axis=-1, keepdims=True)
        self.log_Ps = np.log(P)


class StickyTransitions(StationaryTransitions):
    """
    Upweight the self transition prior.

    pi_k ~ Dir(alpha + kappa * e_k)
    """
    def __init__(self, num_states, observation_dim, input_dim=0, alpha=1, kappa=100):
        super(StickyTransitions, self).\
            __init__(num_states, observation_dim, input_dim=input_dim)
        self.alpha = alpha
        self.kappa = kappa

    def log_prior(self):
        K = self.num_states
        Ps = np.exp(self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True))

        lp = 0
        for k in range(K):
            alpha = self.alpha * np.ones(K) + self.kappa * (np.arange(K) == k)
            lp += dirichlet.logpdf(Ps[k], alpha)
        return lp

    def m_step(self, expectations, dataset, **kwargs):
        expected_joints = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-8
        expected_joints += self.kappa * np.eye(self.num_states)
        P = expected_joints / expected_joints.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(P)


class InputDrivenTransitions(StickyTransitions):
    """
    Hidden Markov Model whose transition probabilities are
    determined by a generalized linear model applied to the
    exogenous input.
    """
    def __init__(self, num_states, observation_dim, input_dim=0, alpha=1, kappa=0):
        super(InputDrivenTransitions, self).\
            __init__(num_states, observation_dim, input_dim=input_dim,
                     alpha=alpha, kappa=kappa)

        # Parameters linking input to state distribution
        self.Ws = npr.randn(num_states, input_dim)

    @property
    def params(self):
        return self.log_Ps, self.Ws

    @params.setter
    def params(self, value):
        self.log_Ps, self.Ws = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.Ws = self.Ws[perm]

    def log_transition_matrices(self, data):
        T = data['data'].shape[0]
        inpt = data['input']
        assert inpt.shape == (T, self.input_dim)

        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        # Input effect
        log_Ps = log_Ps + np.dot(inpt[1:], self.Ws.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, dataset, **kwargs):
        _Transitions.m_step(self, expectations, dataset, **kwargs)


class RecurrentTransitions(InputDrivenTransitions):
    """
    Generalization of the input driven HMM in which the observations serve as future inputs
    """
    def __init__(self, num_states, observation_dim, input_dim=0, alpha=1, kappa=0):
        super(RecurrentTransitions, self).\
            __init__(num_states, observation_dim, input_dim=input_dim,
                     alpha=alpha, kappa=kappa)

        # Parameters linking past observations to state distribution
        self.Rs = np.zeros((num_states, observation_dim))

    @property
    def params(self):
        return super(RecurrentTransitions, self).params + (self.Rs,)

    @params.setter
    def params(self, value):
        self.Rs = value[-1]
        super(RecurrentTransitions, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        super(RecurrentTransitions, self).permute(perm)
        self.Rs = self.Rs[perm]

    def log_transition_matrices(self, data):
        obs = data['data']
        inpt = data['input']
        T, D = data.shape

        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        # Input effect
        log_Ps = log_Ps + np.dot(inpt[1:], self.Ws.T)[:, None, :]
        # Past observations effect
        log_Ps = log_Ps + np.dot(obs[:-1], self.Rs.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, dataset, **kwargs):
        _Transitions.m_step(self, expectations, dataset, **kwargs)


class RecurrentOnlyTransitions(_Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, num_states, observation_dim, input_dim=0):
        super(RecurrentOnlyTransitions, self).\
            __init__(num_states, observation_dim, input_dim=input_dim)

        # Parameters linking past observations to state distribution
        self.Ws = npr.randn(num_states, input_dim)
        self.Rs = npr.randn(num_states, observation_dim)
        self.r = npr.randn(num_states)

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs, self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data):
        obs = data['data']
        inpt = data['input']
        T, D = data.shape

        log_Ps = np.dot(inpt[1:], self.Ws.T)[:, None, :]              # inputs
        log_Ps = log_Ps + np.dot(obs[:-1], self.Rs.T)[:, None, :]     # past observations
        log_Ps = log_Ps + self.r                                      # bias
        log_Ps = np.tile(log_Ps, (1, self.num_states, 1))             # expand
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)      # normalize

    def m_step(self, expectations, dataset, **kwargs):
        _Transitions.m_step(self, expectations, dataset, **kwargs)


class RBFRecurrentTransitions(InputDrivenTransitions):
    """
    Recurrent transitions with radial basis functions for parameterizing
    the next state probability given current continuous data. We have,

    p(z_{t+1} = k | z_t, x_t)
        \propto N(x_t | \mu_k, \Sigma_k) \times \pi_{z_t, z_{t+1})

    where {\mu_k, \Sigma_k, \pi_k}_{k=1}^K are learned parameters.
    Equivalently,

    log p(z_{t+1} = k | z_t, x_t)
        = log N(x_t | \mu_k, \Sigma_k) + log \pi_{z_t, z_{t+1}) + const
        = -D/2 log(2\pi) -1/2 log |Sigma_k|
          -1/2 (x - \mu_k)^T \Sigma_k^{-1} (x-\mu_k)
          + log \pi{z_t, z_{t+1}}

    The difference between this and the recurrent model above is that the
    log transition matrices are quadratic functions of x rather than linear.

    While we're at it, there's no harm in adding a linear term to the log
    transition matrices to capture input dependencies.
    """
    def __init__(self, num_states, observation_dim, input_dim=0, alpha=1, kappa=0):
        super(RBFRecurrentTransitions, self).\
            __init__(num_states, observation_dim, input_dim=input_dim,
                     alpha=alpha, kappa=kappa)

        # RBF parameters
        self.mus = npr.randn(num_states, observation_dim)
        self._sqrt_Sigmas = npr.randn(num_states, observation_dim, observation_dim)

    @property
    def params(self):
        return self.log_Ps, self.mus, self._sqrt_Sigmas, self.Ws

    @params.setter
    def params(self, value):
        self.log_Ps, self.mus, self._sqrt_Sigmas, self.Ws = value

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    def initialize(self, dataset):
        # Fit a GMM to the data to set the means and covariances
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(self.num_states, covariance_type="full")
        gmm.fit(np.vstack([data['data'] for data in dataset]))
        self.mus = gmm.means_
        self._sqrt_Sigmas = np.linalg.cholesky(gmm.covariances_)

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.mus = self.mus[perm]
        self.sqrt_Sigmas = self.sqrt_Sigmas[perm]
        self.Ws = self.Ws[perm]

    def log_transition_matrices(self, data):
        obs = data['data']
        inpt = data['input']
        mask = data['mask']
        assert np.all(mask), "Recurrent models require that all data are present."
        T = obs.shape[0]
        assert inpt.shape == (T, self.input_dim)
        K, D = self.num_states, self.observation_dim

        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))

        # RBF recurrent function
        rbf = multivariate_normal_logpdf(obs[:-1, None, :], self.mus, self.Sigmas)
        log_Ps = log_Ps + rbf[:, None, :]

        # Input effect
        log_Ps = log_Ps + np.dot(inpt[1:], self.Ws.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, dataset, **kwargs):
        _Transitions.m_step(self, expectations, dataset, **kwargs)


# Allow general nonlinear emission models with neural networks
class NeuralNetworkRecurrentTransitions(_Transitions):
    def __init__(self, num_states, observation_dim, input_dim=0,
                 hidden_layer_sizes=(50,), nonlinearity="relu"):
        super(NeuralNetworkRecurrentTransitions, self).\
            __init__(num_states, observation_dim, input_dim=input_dim)

        # Baseline transition probabilities
        Ps = .95 * np.eye(num_states) + .05 * npr.rand(num_states, num_states)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

        # Initialize the NN weights
        layer_sizes = (observation_dim + input_dim,) + hidden_layer_sizes + (num_states,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

        nonlinearities = dict(
            relu=relu,
            tanh=np.tanh,
            sigmoid=logistic)
        self.nonlinearity = nonlinearities[nonlinearity]

    @property
    def params(self):
        return self.log_Ps, self.weights, self.biases

    @params.setter
    def params(self, value):
        self.log_Ps, self.weights, self.biases = value

    def permute(self, perm):
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.weights[-1] = self.weights[-1][:,perm]
        self.biases[-1] = self.biases[-1][perm]

    def log_transition_matrices(self, data):
        # Pass the data and inputs through the neural network
        x = np.hstack((data['data'][:-1], data['input'][1:]))
        for W, b in zip(self.weights, self.biases):
            y = np.dot(x, W) + b
            x = self.nonlinearity(y)

        # Add the baseline transition biases
        log_Ps = self.log_Ps[None, :, :] + y[:, None, :]

        # Normalize
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, dataset, optimizer="adam", num_iters=100, **kwargs):
        # Default to adam instead of bfgs for the neural network model.
        _Transitions.m_step(self, expectations, dataset,
                            optimizer=optimizer, num_iters=num_iters, **kwargs)


class NegativeBinomialSemiMarkovTransitions(_Transitions):
    """
    Semi-Markov transition model with negative binomial (NB) distributed
    state durations, as compared to the geometric state durations in the
    standard Markov model.  The negative binomial has higher variance than
    the geometric, but its mode can be greater than 1.

    The NB(r, p) distribution, with r a positive integer and p a probability
    in [0, 1], is this distribution over number of heads before seeing
    r tails where the probability of heads is p. The number of heads
    between each tails is an independent geometric random variable.  Thus,
    the total number of heads is the sum of r independent and identically
    distributed geometric random variables.

    We can "embed" the semi-Markov model with negative binomial durations
    in the standard Markov model by expanding the state space.  Map each
    discrete state k to r new states: (k,1), (k,2), ..., (k,r_k),
    for k in 1, ..., K. The total number of states is \sum_k r_k,
    where state k has a NB(r_k, p_k) duration distribution.

    The transition probabilities are as follows. The probability of staying
    within the same "super state" are:

    p(z_{t+1} = (k,i) | z_t = (k,i)) = p_k

    and for 0 <= j <= r_k - i

    p(z_{t+1} = (k,i+j) | z_t = (k,i)) = (1-p_k)^{j-i} p_k

    The probability of flipping (r_k - i + 1) tails in a row in state k;
    i.e. the probability of exiting super state k, is (1-p_k)^{r_k-i+1}.
    Thus, the probability of transitioning to a new super state is:

    p(z_{t+1} = (j,1) | z_t = (k,i)) = (1-p_k)^{r_k-i+1} * P[k, j]

    where P[k, j] is a transition matrix with zero diagonal.

    As a sanity check, note that the sum of probabilities is indeed 1:

    \sum_{j=i}^{r_k} p(z_{t+1} = (k,j) | z_t = (k,i))
        + \sum_{m \neq k}  p(z_{t+1} = (m, 1) | z_t = (k, i))

    = \sum_{j=0}^{r_k-i} (1-p_k)^j p_k + \sum_{m \neq k} (1-p_k)^{r_k-i+1} * P[k, j]

    = p_k (1-(1-p_k)^{r_k-i+1}) / (1-(1-p_k)) + (1-p_k)^{r_k-i+1}

    = 1 - (1-p_k)^{r_k-i+1} + (1 - p_k)^{r_k-i+1}

    = 1.

    where we used the geometric series and the fact that \sum_{j != k} P[k, j] = 1.
    """
    def __init__(self, num_states, observation_dim, input_dim=0, r_min=1, r_max=20):
        assert num_states > 1, "Explicit duration models only work if num states > 1."
        super(NegativeBinomialSemiMarkovTransitions, self).\
            __init__(num_states, observation_dim, input_dim=input_dim)

        # Initialize the super state transition probabilities
        self.Ps = npr.rand(num_states, num_states)
        np.fill_diagonal(self.Ps, 0)
        self.Ps /= self.Ps.sum(axis=1, keepdims=True)

        # Initialize the negative binomial duration probabilities
        self.r_min, self.r_max = r_min, r_max
        self.rs = npr.randint(r_min, r_max + 1, size=num_states)
        self.ps = 0.5 * np.ones(num_states)

        # Initialize the transition matrix
        self._transition_matrix = None

    @property
    def params(self):
        return (self.Ps, self.rs, self.ps)

    @params.setter
    def params(self, value):
        Ps, rs, ps = value
        assert Ps.shape == (self.num_states, self.num_states)
        assert np.allclose(np.diag(Ps), 0)
        assert np.allclose(Ps.sum(1), 1)
        assert rs.shape == (self.num_states)
        assert rs.dtype == int
        assert np.all(rs > 0)
        assert ps.shape == (self.num_states)
        assert np.all(ps > 0)
        assert np.all(ps < 1)
        self.Ps, self.rs, self.ps = Ps, rs, ps

        # Reset the transition matrix
        self._transition_matrix = None

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ps = self.Ps[np.ix_(perm, perm)]
        self.rs = self.rs[perm]
        self.ps = self.ps[perm]

        # Reset the transition matrix
        self._transition_matrix = None

    @property
    def total_num_states(self):
        return np.sum(self.rs)

    @property
    def state_map(self):
        return np.repeat(np.arange(self.num_states), self.rs)

    @property
    def transition_matrix(self):
        if self._transition_matrix is not None:
            return self._transition_matrix

        As, rs, ps = self.Ps, self.rs, self.ps

        # Fill in the transition matrix one block at a time
        K_total = self.total_num_states
        P = np.zeros((K_total, K_total))
        starts = np.concatenate(([0], np.cumsum(rs)[:-1]))
        ends = np.cumsum(rs)
        for (i, j), Aij in np.ndenumerate(As):
            block = P[starts[i]:ends[i], starts[j]:ends[j]]

            # Diagonal blocks (stay in sub-state or advance to next sub-state)
            if i == j:
                for k in range(rs[i]):
                    # p(z_{t+1} = (.,i+k) | z_t = (.,i)) = (1-p)^k p
                    # for 0 <= k <= r - i
                    block += (1 - ps[i])**k * ps[i] * np.diag(np.ones(rs[i]-k), k=k)

            # Off-diagonal blocks (exit to a new super state)
            else:
                # p(z_{t+1} = (j,1) | z_t = (k,i)) = (1-p_k)^{r_k-i+1} * A[k, j]
                block[:,0] = (1-ps[i]) ** np.arange(rs[i], 0, -1) * Aij

        assert np.allclose(P.sum(1),1)
        assert (0 <= P).all() and (P <= 1.).all()

        # Cache the transition matrix
        self._transition_matrix = P

        return P

    def log_transition_matrices(self, data):
        T = data['data'].shape[0]
        P = self.transition_matrix
        return np.tile(np.log(P)[None, :, :], (T-1, 1, 1))

    def m_step(self, expectations, dataset, samples, **kwargs):
        # Update the transition matrix between super states
        P = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-16
        np.fill_diagonal(P, 0)
        P /= P.sum(axis=-1, keepdims=True)
        self.Ps = P

        # Fit negative binomial models for each duration based on sampled states
        states, durations = map(np.concatenate, zip(*[rle(z_smpl) for z_smpl in samples]))
        for k in range(self.num_states):
            self.rs[k], self.ps[k] = \
                fit_negative_binomial_integer_r(durations[states == k], self.r_min, self.r_max)

        # Reset the transition matrix
        self._transition_matrix = None
