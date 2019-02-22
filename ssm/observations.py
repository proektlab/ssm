import copy
import warnings

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd.scipy.special import gammaln, digamma

from ssm.util import random_rotation, check_dataset, \
    logistic, logit, one_hot, generalized_newton_studentst_dof, \
    fit_linear_regression
from ssm.preprocessing import interpolate_data
from ssm.cstats import robust_ar_statistics
from ssm.optimizers import adam, bfgs, rmsprop, sgd
import ssm.stats as stats


class _Observations(object):

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

    def permute(self, perm):
        pass

    @check_dataset
    def initialize(self, dataset):
        pass

    def log_prior(self):
        return 0

    def log_likelihoods(self, data):
        raise NotImplementedError

    def sample_observation(self, z, xhist, input=None, tag=None, with_noise=True):
        raise NotImplementedError

    def m_step(self, expectations, dataset, optimizer="bfgs", **kwargs):
        """
        If M-step cannot be done in closed form for the transitions, default to SGD.
        """
        optimizer = dict(adam=adam, bfgs=bfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, (expected_states, expected_joints, _) in zip(dataset, expectations):
                lls = self.log_likelihoods(data)
                elbo += np.sum(expected_states * lls)
            return elbo

        # define optimization target
        T = sum([data['data'].shape[0] for data in dataset])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = optimizer(_objective, self.params, **kwargs)

    def smooth(self, expectations, data):
        raise NotImplementedError


class GaussianObservations(_Observations):
    def __init__(self, num_states, observation_dim, input_dim=0):
        super(GaussianObservations, self).__init__(num_states, observation_dim, input_dim)
        self.mus = npr.randn(num_states, observation_dim)
        self._sqrt_Sigmas = npr.randn(num_states, observation_dim, observation_dim)

    @property
    def params(self):
        return self.mus, self._sqrt_Sigmas

    @params.setter
    def params(self, value):
        self.mus, self._sqrt_Sigmas = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._sqrt_Sigmas = self._sqrt_Sigmas[perm]

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    @check_dataset
    def initialize(self, dataset):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate([data['data'] for data in dataset])
        km = KMeans(self.num_states).fit(data)
        self.mus = km.cluster_centers_
        Sigmas = np.array([np.cov(data[km.labels_ == k].T) for k in range(self.num_states)])
        self._sqrt_Sigmas = np.linalg.cholesky(Sigmas + 1e-8 * np.eye(self.observation_dim))

    def log_likelihoods(self, data):
        mus, Sigmas = self.mus, self.Sigmas

        mask = data['mask']
        if mask is not None and np.any(~mask) and not isinstance(mus, np.ndarray):
            raise Exception("Current implementation of multivariate_normal_logpdf for masked data"
                            "does not work with autograd because it writes to an array. "
                            "Use DiagonalGaussian instead if you need to support missing data.")

        return stats.multivariate_normal_logpdf(data['data'][:, None, :], mus, Sigmas)

    def sample_observation(self, z, xhist, input=None, tag=None, with_noise=True):
        K, D, mus = self.num_states, self.observation_dim, self.mus
        sqrt_Sigmas = self._sqrt_Sigmas if with_noise else np.zeros((K, D, D))
        return mus[z] + np.dot(sqrt_Sigmas[z], npr.randn(D))

    def m_step(self, expectations, dataset, **kwargs):
        K, D = self.num_states, self.observation_dim
        J = np.zeros((K, D))
        h = np.zeros((K, D))

        for (Ez, _, _), data in zip(expectations, dataset):
            obs = data['data']
            J += np.sum(Ez[:, :, None], axis=0)
            h += np.sum(Ez[:, :, None] * obs[:, None, :], axis=0)
        self.mus = h / J

        # Update the variance
        sqerr = np.zeros((K, D, D))
        weight = np.zeros((K,))
        for (Ez, _, _), data in zip(expectations, dataset):
            obs = data['data']
            resid = obs[:, None, :] - self.mus
            sqerr += np.sum(Ez[:, :, None, None] * resid[:, :, None, :] * resid[:, :, :, None], axis=0)
            weight += np.sum(Ez, axis=0)
        self._sqrt_Sigmas = np.linalg.cholesky(sqerr / weight[:, None, None] + 1e-8 * np.eye(D))

    def smooth(self, expectations, data):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)


class DiagonalGaussianObservations(_Observations):
    def __init__(self, num_states, observation_dim, input_dim=0):
        super(DiagonalGaussianObservations, self).__init__(num_states, observation_dim, input_dim)
        self.mus = npr.randn(num_states, observation_dim)
        self._log_sigmasq = -2 + npr.randn(num_states, observation_dim)

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @sigmasq.setter
    def sigmasq(self, value):
        assert np.all(value > 0) and value.shape == (self.num_states, self.observation_dim)
        self._log_sigmasq = np.log(value + 1e-16)

    @property
    def params(self):
        return self.mus, self._log_sigmasq

    @params.setter
    def params(self, value):
        self.mus, self._log_sigmasq = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._log_sigmasq = self._log_sigmasq[perm]

    @check_dataset
    def initialize(self, dataset):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate([data['data'] for data in dataset])
        km = KMeans(self.num_states).fit(data)
        self.mus = km.cluster_centers_
        sigmasq = np.array([np.var(data[km.labels_ == k], axis=0)
                           for k in range(self.num_states)])
        self.sigmasq = sigmasq

    def log_likelihoods(self, data):
        mus, sigmas = self.mus, self.sigmasq
        obs = data['data']
        mask = data['mask']
        return stats.diagonal_gaussian_logpdf(obs[:, None, :], mus, sigmas, mask=mask[:, None, :])

    def sample_observation(self, z, xhist, input=None, tag=None, with_noise=True):
        K, D, mus = self.num_states, self.observation_dim, self.mus
        sigmas = self.sigmasq if with_noise else np.zeros((K, D))
        return mus[z] + np.sqrt(sigmas[z]) * npr.randn(D)

    def m_step(self, expectations, dataset, **kwargs):
        x = np.concatenate([data['data'] for data in dataset])
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.num_states):
            self.mus[k] = np.average(x, axis=0, weights=weights[:,k])
            sqerr = (x - self.mus[k])**2
            self._log_sigmasq[k] = np.log(np.average(sqerr, weights=weights[:,k], axis=0) + 1e-16)

    def smooth(self, expectations, data):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)


class StudentsTObservations(_Observations):
    def __init__(self, num_states, observation_dim, input_dim=0):
        super(StudentsTObservations, self).__init__(num_states, observation_dim, input_dim)
        self.mus = npr.randn(num_states, observation_dim)
        self._log_sigmasq = -2 + npr.randn(num_states, observation_dim)
        # Student's t distribution also has a degrees of freedom parameter
        self._log_nus = np.log(4) * np.ones((num_states, observation_dim))

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @sigmasq.setter
    def sigmasq(self, value):
        assert np.all(value > 0) and value.shape == (self.num_states, self.observation_dim)
        self._log_sigmasq = np.log(value + 1e-16)

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @nus.setter
    def nus(self, value):
        assert np.all(value > 0) and value.shape == (self.num_states, self.observation_dim)
        self._log_nus = np.log(value + 1e-16)

    @property
    def params(self):
        return self.mus, self._log_sigmasq, self._log_nus

    @params.setter
    def params(self, value):
        self.mus, self._log_sigmasq, self._log_nus = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._log_sigmasq = self._log_sigmasq[perm]
        self._log_nus = self._log_nus[perm]

    @check_dataset
    def initialize(self, dataset):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate([data['data'] for data in dataset])
        km = KMeans(self.num_states).fit(data)
        self.mus = km.cluster_centers_
        self.sigmasq = np.array([np.var(data[km.labels_ == k], axis=0) for k in range(self.num_states)])

    def log_likelihoods(self, data):
        D, mus, sigmas, nus = self.observation_dim, self.mus, self.sigmasq, self.nus
        obs = data['data']
        mask = data['mask']
        return stats.independent_studentst_logpdf(obs[:, None, :], mus, sigmas, nus, mask=mask[:, None, :])

    def sample_observation(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, sigmas, nus = self.observation_dim, self.mus, self.sigmasq, self.nus
        tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        sigma = sigmas[z] / tau if with_noise else 0
        return mus[z] + np.sqrt(sigma) * npr.randn(D)

    def smooth(self, expectations, data):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)

    def m_step(self, expectations, dataset, **kwargs):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t for
        complete details.
        """
        self._m_step_mu_sigma(expectations, dataset)
        self._m_step_nu(expectations, dataset)

    def _m_step_mu_sigma(self, expectations, dataset):
        K, D = self.num_states, self.observation_dim

        # Estimate the precisions w for each data point
        E_taus = []
        for data in dataset:
            # nu: (K,D)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> tau: (T, K, D)
            obs = data['data']
            alpha = self.nus/2 + 1/2
            beta = self.nus/2 + 1/2 * (obs[:, None, :] - self.mus)**2 / self.sigmasq
            E_taus.append(alpha / beta)

        # Update the mean (notation from natural params of Gaussian)
        J = np.zeros((K, D))
        h = np.zeros((K, D))
        for E_tau, (Ez, _, _), data in zip(E_taus, expectations, dataset):
            obs = data['data']
            J += np.sum(Ez[:, :, None] * E_tau, axis=0)
            h += np.sum(Ez[:, :, None] * E_tau * obs[:, None, :], axis=0)
        self.mus = h / J

        # Update the variance
        sqerr = np.zeros((K, D))
        weight = np.zeros((K, D))
        for E_tau, (Ez, _, _), data in zip(E_taus, expectations, dataset):
            obs = data['data']
            sqerr += np.sum(Ez[:, :, None] * E_tau * (obs[:, None, :] - self.mus)**2, axis=0)
            weight += np.sum(Ez[:, :, None], axis=0)
        self.sigmasq = sqerr / weight

    def _m_step_nu(self, expectations, dataset):
        """
        The shape parameter nu determines a gamma prior.  We have

            tau_n ~ Gamma(nu/2, nu/2)
            y_n ~ N(mu, sigma^2 / tau_n)

        To update nu, we do EM and optimize the expected log likelihood using
        a generalized Newton's method.  See the notebook in doc/students_t for
        complete details.
        """
        K, D = self.num_states, self.observation_dim

        # Compute the precisions w for each data point
        E_taus = np.zeros((K, D))
        E_logtaus = np.zeros((K, D))
        weights = np.zeros(K)
        for data, (Ez, _, _) in zip(dataset, expectations):
            # nu: (K,D)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> alpha/beta: (T, K, D)
            obs = data['data']
            alpha = self.nus/2 + 1/2
            beta = self.nus/2 + 1/2 * (obs[:, None, :] - self.mus)**2 / self.sigmasq

            E_taus += np.sum(Ez[:, :, None] * (alpha / beta), axis=0)
            E_logtaus += np.sum(Ez[:, :, None] * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights[:, None]
        E_logtaus /= weights[:, None]

        for k in range(K):
            for d in range(D):
                self._log_nus[k, d] = np.log(generalized_newton_studentst_dof(E_taus[k, d], E_logtaus[k, d]))


class MultivariateStudentsTObservations(_Observations):
    def __init__(self, num_states, observation_dim, input_dim=0):
        super(MultivariateStudentsTObservations, self).\
            __init__(num_states, observation_dim, input_dim)
        self.mus = npr.randn(num_states, observation_dim)
        self._sqrt_Sigmas = npr.randn(num_states, observation_dim, observation_dim)
        self._log_nus = np.log(4) * np.ones((num_states,))

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @property
    def params(self):
        return self.mus, self._sqrt_Sigmas, self._log_nus

    @params.setter
    def params(self, value):
        self.mus, self._sqrt_Sigmas, self._log_nus = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self._sqrt_Sigmas = self._sqrt_Sigmas[perm]
        self._log_nus = self._log_nus[perm]

    @check_dataset
    def initialize(self, dataset):
        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate([data['data'] for data in dataset])
        km = KMeans(self.num_states).fit(data)
        self.mus = km.cluster_centers_
        Sigmas = np.array([np.cov(data[km.labels_ == k].T) for k in range(self.num_states)])
        self._sqrt_Sigmas = np.linalg.cholesky(Sigmas + 1e-8 * np.eye(self.observation_dim))
        self._log_nus = np.log(4) * np.ones((self.num_states,))

    def log_likelihoods(self, data):
        assert np.all(data['mask']), "MultivariateStudentsTObservations does not support missing data"
        D, mus, Sigmas, nus = self.observation_dim, self.mus, self.Sigmas, self.nus
        return stats.multivariate_studentst_logpdf(data['data'][:, None, :], mus, Sigmas, nus)

    def m_step(self, expectations, dataset, **kwargs):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t for
        complete details.
        """
        self._m_step_mu_sigma(expectations, dataset)
        self._m_step_nu(expectations, dataset)

    def _m_step_mu_sigma(self, expectations, dataset):
        K, D = self.num_states, self.observation_dim

        # Estimate the precisions w for each data point
        E_taus = []
        for data in dataset:
            # nu: (K,)  mus: (K, D)  Sigmas: (K, D, D)  y: (T, D)  -> tau: (T, K)
            y = data['data']
            alpha = self.nus/2 + D/2
            beta = self.nus/2 + 1/2 * stats.batch_mahalanobis(self._sqrt_Sigmas, y[:, None, :] - self.mus)
            E_taus.append(alpha / beta)

        # Update the mean (notation from natural params of Gaussian)
        J = np.zeros((K,))
        h = np.zeros((K, D))
        for E_tau, (Ez, _, _), data in zip(E_taus, expectations, dataset):
            y = data['data']
            J += np.sum(Ez * E_tau, axis=0)
            h += np.sum(Ez[:, :, None] * E_tau[:, :, None] * y[:, None, :], axis=0)
        self.mus = h / J[:, None]

        # Update the variance
        sqerr = np.zeros((K, D, D))
        weight = np.zeros((K,))
        for E_tau, (Ez, _, _), data in zip(E_taus, expectations, dataset):
            # sqerr += np.sum(Ez[:, :, None] * E_tau * (y[:, None, :] - self.mus)**2, axis=0)
            y = data['data']
            resid = y[:, None, :] - self.mus
            sqerr += np.einsum('tk,tk,tki,tkj->kij', Ez, E_tau, resid, resid)
            weight += np.sum(Ez, axis=0)

        self._sqrt_Sigmas = np.linalg.cholesky(sqerr / weight[:, None, None] + 1e-8 * np.eye(D))

    def _m_step_nu(self, expectations, dataset):
        """
        The shape parameter nu determines a gamma prior.  We have

            tau_n ~ Gamma(nu/2, nu/2)
            y_n ~ N(mu, Sigma / tau_n)

        To update nu, we do EM and optimize the expected log likelihood using
        a generalized Newton's method.  See the notebook in doc/students_t for
        complete details.
        """
        K, D = self.num_states, self.observation_dim

        # Compute the precisions w for each data point
        E_taus = np.zeros(K)
        E_logtaus = np.zeros(K)
        weights = np.zeros(K)
        for data, (Ez, _, _) in zip(dataset, expectations):
            # nu: (K,)  mus: (K, D)  Sigmas: (K, D, D)  y: (T, D)  -> alpha/beta: (T, K)
            y = data['data']
            alpha = self.nus/2 + D/2
            beta = self.nus/2 + 1/2 * stats.batch_mahalanobis(self._sqrt_Sigmas, y[:, None, :] - self.mus)

            E_taus += np.sum(Ez * (alpha / beta), axis=0)
            E_logtaus += np.sum(Ez * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights
        E_logtaus /= weights

        for k in range(K):
            self._log_nus[k] = np.log(generalized_newton_studentst_dof(E_taus[k], E_logtaus[k]))

    def sample_observation(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, Sigmas, nus = self.observation_dim, self.mus, self.Sigmas, self.nus
        tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
        sqrt_Sigma = np.linalg.cholesky(Sigmas[z] / tau) if with_noise else 0
        return mus[z] + np.dot(sqrt_Sigma, npr.randn(D))

    def smooth(self, expectations, data):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(self.mus)


class BernoulliObservations(_Observations):

    def __init__(self, num_states, observation_dim, input_dim=0):
        super(BernoulliObservations, self).__init__(num_states, observation_dim, input_dim)
        self.logit_ps = npr.randn(num_states, observation_dim)

    @property
    def params(self):
        return self.logit_ps

    @params.setter
    def params(self, value):
        self.logit_ps = value

    def permute(self, perm):
        self.logit_ps = self.logit_ps[perm]

    @check_dataset
    def initialize(self, dataset):

        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate([data['data'] for data in dataset])
        km = KMeans(self.num_states).fit(data)
        ps = np.clip(km.cluster_centers_, 1e-3, 1-1e-3)
        self.logit_ps = logit(ps)

    def log_likelihoods(self, data):
        obs = data['obs']
        mask = data['mask']
        return stats.bernoulli_logpdf(obs[:, None, :], self.logit_ps, mask=mask[:, None, :])

    def sample_observation(self, z, xhist, input=None, tag=None, with_noise=True):
        ps = 1 / (1 + np.exp(self.logit_ps))
        return npr.rand(self.observation_dim) < ps[z]

    def m_step(self, expectations, dataset, **kwargs):
        x = np.concatenate([data['data'] for data in dataset])
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.num_states):
            ps = np.clip(np.average(x, axis=0, weights=weights[:,k]), 1e-3, 1-1e-3)
            self.logit_ps[k] = logit(ps)

    def smooth(self, expectations, data):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        ps = 1 / (1 + np.exp(self.logit_ps))
        return expectations.dot(ps)


class PoissonObservations(_Observations):

    def __init__(self, num_states, observation_dim, input_dim=0):
        super(PoissonObservations, self).__init__(num_states, observation_dim, input_dim)
        self.log_lambdas = npr.randn(num_states, observation_dim)

    @property
    def params(self):
        return self.log_lambdas

    @params.setter
    def params(self, value):
        self.log_lambdas = value

    def permute(self, perm):
        self.log_lambdas = self.log_lambdas[perm]

    @check_dataset
    def initialize(self, dataset):

        # Initialize with KMeans
        from sklearn.cluster import KMeans
        data = np.concatenate([data['data'] for data in dataset])
        km = KMeans(self.num_states).fit(data)
        self.log_lambdas = np.log(km.cluster_centers_ + 1e-3)

    def log_likelihoods(self, data):
        lambdas = np.exp(self.log_lambdas)
        obs = data['data']
        mask = data['mask']
        return stats.poisson_logpdf(obs[:, None, :], lambdas, mask=mask[:, None, :])

    def sample_observation(self, z, xhist, input=None, tag=None, with_noise=True):
        lambdas = np.exp(self.log_lambdas)
        return npr.poisson(lambdas[z])

    def m_step(self, expectations, dataset, **kwargs):
        x = np.concatenate([data['data'] for data in dataset])
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.num_states):
            self.log_lambdas[k] = np.log(np.average(x, axis=0, weights=weights[:,k]) + 1e-16)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        return expectations.dot(np.exp(self.log_lambdas))


class CategoricalObservations(_Observations):

    def __init__(self, num_states, observation_dim, input_dim=0, num_classes=3):
        """
        @param C:  number of classes in the categorical observations
        """
        super(CategoricalObservations, self).__init__(num_states, observation_dim, input_dim)
        self.num_classes = num_classes
        self.logits = npr.randn(num_states, observation_dim, num_classes)

    @property
    def params(self):
        return self.logits

    @params.setter
    def params(self, value):
        self.logits = value

    def permute(self, perm):
        self.logits = self.logits[perm]

    @check_dataset
    def initialize(self, dataset):
        pass

    def log_likelihoods(self, data):
        obs = data['data']
        mask = data['mask']
        return stats.categorical_logpdf(obs[:, None, :], self.logits, mask=mask[:, None, :])

    def sample_observation(self, z, xhist, input=None, tag=None, with_noise=True):
        ps = np.exp(self.logits - logsumexp(self.logits, axis=2, keepdims=True))
        return np.array([npr.choice(self.num_classes, p=ps[z, d]) for d in range(self.observation_dim)])

    def m_step(self, expectations, dataset, **kwargs):
        x = np.concatenate([data['data'] for data in dataset])
        weights = np.concatenate([Ez for Ez, _, _ in expectations])
        for k in range(self.num_states):
            # compute weighted histogram of the class assignments
            xoh = one_hot(x, self.num_classes)                                # T x D x C
            ps = np.average(xoh, axis=0, weights=weights[:, k]) + 1e-3        # D x C
            ps /= np.sum(ps, axis=-1, keepdims=True)
            self.logits[k] = np.log(ps)

    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        raise NotImplementedError


class _AutoRegressiveObservationsBase(_Observations):
    """
    Base class for autoregressive observations of the form,

    E[x_t | x_{t-1}, z_t=k, u_t]
        = \sum_{l=1}^{L} A_k^{(l)} x_{t-l} + b_k + V_k u_t.

    where L is the number of lags and u_t is the input.
    """
    def __init__(self, num_states, observation_dim, input_dim=0, lags=1):
        super(_AutoRegressiveObservationsBase, self).__init__(num_states, observation_dim, input_dim)

        # Distribution over initial point
        self.mu_init = np.zeros((num_states, observation_dim))

        # AR parameters
        assert lags > 0
        self.lags = lags
        self.bs = npr.randn(num_states, observation_dim)
        self.Vs = npr.randn(num_states, observation_dim, input_dim)

        # Inheriting classes may treat _As differently
        self._As = None

    @property
    def As(self):
        return self._As

    @As.setter
    def As(self, value):
        self._As = value

    @property
    def params(self):
        return self.As, self.bs, self.Vs

    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs = value

    def permute(self, perm):
        self.mu_init = self.mu_init[perm]
        self.As = self.As[perm]
        self.bs = self.bs[perm]
        self.Vs = self.Vs[perm]

    def _compute_mus(self, data):
        obs = data['data']
        inpt = data['input']
        mask = data['mask']
        assert np.all(mask), "ARHMM cannot handle missing data"

        K, D, L = self.num_states, self.observation_dim, self.lags
        T = obs.shape[0]
        As, bs, Vs = self.As, self.bs, self.Vs

        # Instantaneous inputs
        mus = np.matmul(Vs[None, ...], inpt[L:, None, :, None])[:, :, :, 0]

        # Lagged data
        for l in range(self.lags):
            Als = As[None, :, :, l*D:(l+1)*D]
            lagged_obs = obs[L-l-1:-l-1, None, :, None]
            mus = mus + np.matmul(Als, lagged_obs)[:, :, :, 0]

        # Bias
        mus = mus + bs

        # Pad with the initial condition
        mus = np.concatenate((self.mu_init * np.ones((L, K, D)), mus))

        assert mus.shape == (T, K, D)
        return mus

    def _compute_second_moment_mus(self, data):
        """
        Compute the second order statistics of the AR mean. We assume
        that time steps are independent with Gaussian distributions.
        The mean is given by `data` and the second moment by `E_xxT`.

        Let mu_t = c_t + d_t where c_t = \sum_{l=1}^L A_l x_{t-l} and d_t = Vu_t + b
        The first term is a random variable; the second is deterministic.

        E[mu_t mu_t^T] = E[c_t c_t^T] + 2 E[c_t] d_t + d_t d_t^T

        E[c_t] = \sum_{l=1}^L A_l E[x_{t-l}]

        E[c_t c_t^T] = \sum_{l=1}^L A_l E[x_{t-l} x_{t-l}^T] A_l^T
                       + \sum_{l=1}^L \sum_{j=1}^L A_l E[x_{t-l}] E[x_{t-j}]^T A_j^T
        """
        obs = data['data']
        inpt = data['input']
        mask = data['mask']

        K, D, L, = self.num_states, self.observation_dim, self.lags
        mu0, As, bs, Vs = self.mu_init, self.As, self.bs, self.Vs
        T = obs.shape[0]

        # Compute second moments
        E_xxT = cov + obs[:, :, None] * obs[:, None, :]

        # Split the dynamics matrix
        As = np.split(As, np.arange(1, L) * L, axis=2)

        # Construct views of the lagged data and moments
        l_obs = [obs[L-l-1:-l-1] for l in range(L)]
        l_xxT = [E_xxT[L-l-1:-l-1] for l in range(L)]

        # Pad with the initial condition
        mu0_mu0T = np.einsum('ki,kj->kij', mu0, mu0)

        # Compute the constant term
        d = np.einsum('kdm,tm->tkd', Vs, inpt[L:]) + bs
        ddT = np.einsum('tki,tkj->tkij', d, d)

        # Compute the terms that depend on the data
        E_c = np.zeros((T-L, K, D))
        E_ccT = np.zeros((T-L, K, D, D))
        for l in range(self.lags):
            E_c += np.einsum('kij,tj->tki', As[l], l_obs[l])
            E_ccT += np.einsum('kij,tjm,knm->tkin', As[l], l_xxT[l])

            # Add the cross terms
            for j in range(self.lags):
                E_ccT += np.einsum('kij,tj,tm,knm->tkin', As[l], l_obs[l], l_obs[j], As[j])

        # Combine terms to compute E_mu_mu^T
        E_mu_muT = E_ccT + 2 * np.einsum('tki,tkj->tkij', E_c, d) + ddT

        # Compute initial condition and predictions
        E_mu_muT = np.concatenate((np.tile(mu0_mu0T[None, ...], (L, 1, 1, 1)), E_mu_muT))

        assert E_mu_muT.shape == (T, K, D, D)
        return E_mu_muT

    def smooth(self, expectations, data):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
        mus = self._compute_mus(data)
        return (expectations[:, :, None] * mus).sum(1)


class AutoRegressiveObservations(_AutoRegressiveObservationsBase):
    """
    AutoRegressive observation model with Gaussian noise.

        (x_t | z_t = k, u_t) ~ N(A_k x_{t-1} + b_k + V_k u_t, S_k)

    where S_k is a positive definite covariance matrix.

    The parameters are fit via maximum likelihood estimation.
    """
    def __init__(self, num_states, observation_dim, input_dim=0, lags=1):
        super(AutoRegressiveObservations, self).__init__(num_states, observation_dim, input_dim)

        # Initialize the dynamics and the noise covariances
        self._As = .95 * np.array([
                np.column_stack([random_rotation(observation_dim),
                                 np.zeros((observation_dim, (lags-1) * observation_dim))])
            for _ in range(num_states)])

        self._sqrt_Sigmas_init = np.tile(np.eye(observation_dim)[None, ...], (num_states, 1, 1))
        self._sqrt_Sigmas = npr.randn(num_states, observation_dim, observation_dim)

    @property
    def Sigmas_init(self):
        return np.matmul(self._sqrt_Sigmas_init, np.swapaxes(self._sqrt_Sigmas_init, -1, -2))

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    @Sigmas.setter
    def Sigmas(self, value):
        assert value.shape == (self.num_states, self.observation_dim, self.observation_dim)
        self._sqrt_Sigmas = np.linalg.cholesky(value)

    @property
    def params(self):
        return super(AutoRegressiveObservations, self).params + (self._sqrt_Sigmas,)

    @params.setter
    def params(self, value):
        self._sqrt_Sigmas = value[-1]
        super(AutoRegressiveObservations, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(AutoRegressiveObservations, self).permute(perm)
        self._sqrt_Sigmas = self._sqrt_Sigmas[perm]

    def initialize(self, dataset, localize=True):
        from sklearn.linear_model import LinearRegression

        # Sample time bins for each discrete state
        # Use the data to cluster the time bins if specified.
        Ts = [data['data'].shape[0] for data in dataset]
        if localize:
            from sklearn.cluster import KMeans
            km = KMeans(self.num_states)
            km.fit(np.vstack([data['data'] for data in dataset]))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])
            zs = [z[:-self.lags] for z in zs]               # remove the ends
        else:
            zs = [npr.choice(self.num_states, size=T-self.lags) for T in Ts]

        # Initialize the weights with linear regression
        Sigmas = []
        for k in range(self.num_states):
            ts = [np.where(z == k)[0] for z in zs]
            Xs = [np.column_stack([data['data'][t + l] for l in range(self.lags)] + \
                                  [data['input'][t]])
                  for t, data in zip(ts, dataset)]
            ys = [data['data'][t+self.lags] for t, data in zip(ts, dataset)]

            # Solve the linear regression
            coef_, intercept_, Sigma = fit_linear_regression(Xs, ys)
            self.As[k] = coef_[:, :self.observation_dim * self.lags]
            self.Vs[k] = coef_[:, self.observation_dim * self.lags:]
            self.bs[k] = intercept_
            Sigmas.append(Sigma)

        # Set the variances all at once to use the setter
        self.Sigmas = np.array(Sigmas)

    def log_likelihoods(self, data):
        """
        Compute the log likelihood of the data given the input.
        If data covariance is specified via the cov argument,
        compute the expected log likelihood of the data instead.

        Parameters
        ----------

        data : array_like (T, D)
            Array of observed data. If cov is specified, this is
            assumed to be the mean of a Gaussian distribution of
            the observed data.

        input : array_like (T, M)
            Array of inputs that bias autoregressive model.

        mask : array_like (T, D), bool
            Mask of the data.  This must be all True for autoregressive
            modles.  We do not handle missing data.

        tag : object
            Unused.

        cov : array_like (T, D, D)
            Optional covariance of the data.

        Returns
        -------

        lps : array_like (T,)
            Log probability of the data, or expected log probability
            if covariances are specified.
        """
        assert np.all(data['mask']), \
            "Cannot compute likelihood of autoregressive obsevations with missing data."
        L = self.lags
        obs = data['data']
        mus = self._compute_mus(data)

        if 'covariance' not in  data:
            # Compute the likelihood of the initial data and remainder separately
            ll_init = stats.multivariate_normal_logpdf(obs[:L, None, :], mus[:L], self.Sigmas_init)
            ll_ar = stats.multivariate_normal_logpdf(obs[L:, None, :], mus[L:], self.Sigmas)
            return np.row_stack((ll_init, ll_ar))

        # Use the second order moments to evaluate expected log likelihood
        cov = data['covariance']
        E_xxT =  cov + obs[:, :, None] * obs[:, None, :]
        E_mu_muT = self._compute_second_moment_mus(data)

        # Compute the likelihood of the initial data and remainder separately
        ll_init = stats.expected_multivariate_normal_logpdf(
            obs[:L, None, :], E_xxT[:L, None, :, :],
            mus[:L], E_mu_muT[:L], self.Sigmas_init)

        ll_ar = stats.expected_multivariate_normal_logpdf(
            obs[L:, None, :], E_xxT[L:, None, :, :],
            mus[L:], E_mu_muT[L:], self.Sigmas)

        return np.row_stack((ll_init, ll_ar))


    def m_step(self, expectations, dataset, **kwargs):
        K, D, M, L = self.num_states, self.observation_dim, self.input_dim, self.lags
        # Collect data for this dimension
        xs, ys, Ezs = [], [], []
        for (Ez, _, _), data in zip(expectations, dataset):
            # Only use data if it is complete
            obs = data['data']
            mask = data['mask']
            inpt = data['input']

            if not np.all(mask):
                raise Exception("Encountered missing data in AutoRegressiveObservations!")

            xs.append(
                np.hstack([obs[L-l-1:-l-1] for l in range(L)] +
                          [inpt[L:, :self.input_dim],
                          np.ones((obs.shape[0]-L, 1))]))
            ys.append(obs[L:])
            Ezs.append(Ez[L:])

        # M step: Fit the weighted linear regressions for each K and D
        J = np.tile(1e-8 * np.eye(D * L + M + 1)[None, :, :], (K, 1, 1))
        h = np.zeros((K, D * L + M + 1, D))
        for x, y, Ez in zip(xs, ys, Ezs):
            J += np.einsum('tk, ti, tj -> kij', Ez, x, x)
            h += np.einsum('tk, ti, td -> kid', Ez, x, y)

        mus = np.linalg.solve(J, h)
        self.As = np.swapaxes(mus[:, :D*L, :], 1, 2)
        self.Vs = np.swapaxes(mus[:, D*L:D*L+M, :], 1, 2)
        self.bs = mus[:, -1, :]

        # Update the covariance
        sqerr = np.zeros((K, D, D))
        weight = np.zeros(K)
        for x, y, Ez in zip(xs, ys, Ezs):
            yhat = np.matmul(x[None, :, :], mus)
            resid = y[None, :, :] - yhat
            sqerr += np.einsum('tk,kti,ktj->kij', Ez, resid, resid)
            weight += np.sum(Ez, axis=0)

        # self._sqrt_Sigmas = np.linalg.cholesky(sqerr / weight[:, None, None] + 1e-8 * np.eye(D))
        self.Sigmas = sqerr / weight[:, None, None] + 1e-8 * np.eye(D)

    def sample_observation(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, Vs = self.observation_dim, self.As, self.bs, self.Vs

        if xhist.shape[0] < self.lags:
            # Sample from the initial distribution
            S = np.linalg.cholesky(self.Sigmas_init[z]) if with_noise else 0
            return self.mu_init[z] + np.dot(S, npr.randn(D))
        else:
            # Sample from the autoregressive distribution
            mu = Vs[z].dot(input) + bs[z]
            for l in range(self.lags):
                Al = As[z][:,l*D:(l+1)*D]
                mu += Al.dot(xhist[-l-1])

            S = np.linalg.cholesky(self.Sigmas[z]) if with_noise else 0
            return mu + np.dot(S, npr.randn(D))


class AutoRegressiveDiagonalNoiseObservations(AutoRegressiveObservations):
    """
    AutoRegressive observation model with diagonal Gaussian noise.

        (x_t | z_t = k, u_t) ~ N(A_k x_{t-1} + b_k + V_k u_t, S_k)

    where

        S_k = diag([sigma_{k,1}, ..., sigma_{k, D}])

    The parameters are fit via maximum likelihood estimation.
    """
    def __init__(self, num_states, observation_dim, input_dim=0, lags=1):
        super(AutoRegressiveDiagonalNoiseObservations, self).\
            __init__(num_states, observation_dim, input_dim)

        # Initialize the dynamics and the noise covariances
        self._As = .95 * np.array([
                np.column_stack([random_rotation(observation_dim),
                                 np.zeros((observation_dim, (lags-1) * observation_dim))])
            for _ in range(num_states)])

        # Get rid of the square root parameterization and replace with log diagonal
        del self._sqrt_Sigmas_init
        del self._sqrt_Sigmas
        self._log_sigmasq_init = np.zeros((num_states, observation_dim))
        self._log_sigmasq = np.zeros((num_states, observation_dim))

    @property
    def sigmasq_init(self):
        return np.exp(self._log_sigmasq_init)

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @property
    def Sigmas_init(self):
        return np.array([np.diag(np.exp(log_s)) for log_s in self._log_sigmasq_init])

    @property
    def Sigmas(self):
        return np.array([np.diag(np.exp(log_s)) for log_s in self._log_sigmasq])

    @Sigmas.setter
    def Sigmas(self, value):
        assert value.shape == (self.num_states, self.observation_dim, self.observation_dim)
        sigmasq = np.array([np.diag(S) for S in value])
        assert np.all(sigmasq > 0)
        self._log_sigmasq = np.log(sigmasq)

    @property
    def params(self):
        return super(AutoRegressiveObservations, self).params + (self._log_sigmasq,)

    @params.setter
    def params(self, value):
        self._log_sigmasq = value[-1]
        super(AutoRegressiveObservations, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(AutoRegressiveObservations, self).permute(perm)
        self._log_sigmasq_init = self._log_sigmasq_init[perm]
        self._log_sigmasq = self._log_sigmasq[perm]


class IndependentAutoRegressiveObservations(_AutoRegressiveObservationsBase):
    def __init__(self, num_states, observation_dim, input_dim=0, lags=1):
        super(IndependentAutoRegressiveObservations, self).\
            __init__(num_states, observation_dim, input_dim)

        self._As = np.concatenate((.95 * np.ones((num_states, observation_dim, 1)),
                                   np.zeros((num_states, observation_dim, lags-1))),
                                  axis=2)
        self._log_sigmasq_init = np.zeros((num_states, observation_dim))
        self._log_sigmasq = np.zeros((num_states, observation_dim))

    @property
    def sigmasq_init(self):
        return np.exp(self._log_sigmasq_init)

    @property
    def sigmasq(self):
        return np.exp(self._log_sigmasq)

    @property
    def As(self):
        return np.array([
                np.column_stack([np.diag(Ak[:,l]) for l in range(self.lags)])
            for Ak in self._As
        ])

    @As.setter
    def As(self, value):
        # TODO: extract the diagonal components
        raise NotImplementedError

    @property
    def params(self):
        return self._As, self.bs, self.Vs, self._log_sigmasq

    @params.setter
    def params(self, value):
        self._As, self.bs, self.Vs, self._log_sigmasq = value

    def permute(self, perm):
        self.mu_init = self.mu_init[perm]
        self._As = self._As[perm]
        self.bs = self.bs[perm]
        self.Vs = self.Vs[perm]
        self._log_sigmasq_init = self._log_sigmasq_init[perm]
        self._log_sigmasq = self._log_sigmasq[perm]

    def initialize(self, dataset):
        # Initialize with linear regressions
        from sklearn.linear_model import LinearRegression
        data = np.concatenate([data['data'] for data in dataset])
        inpt = np.concatenate([data['input'] for data in dataset])
        T = data.shape[0]

        for k in range(self.num_states):
            for d in range(self.observation_dim):
                ts = npr.choice(T-self.lags, replace=False, size=(T-self.lags)//self.num_states)
                x = np.column_stack([data[ts + l, d:d+1] for l in range(self.lags)] + [inpt[ts, :self.input_dim]])
                y = data[ts+self.lags, d:d+1]
                lr = LinearRegression().fit(x, y)

                self.As[k, d] = lr.coef_[:, :self.lags]
                self.Vs[k, d] = lr.coef_[:, self.lags:self.lags+self.input_dim]
                self.bs[k, d] = lr.intercept_

                resid = y - lr.predict(x)
                sigmas = np.var(resid, axis=0)
                self._log_sigmasq[k, d] = np.log(sigmas + 1e-16)

    def _compute_mus(self, data):
        """
        Re-implement compute_mus for this class since we can do it much
        more efficiently than in the general AR case.
        """
        K, D, M, L = self.num_states, self.observation_dim, self.input_dim, self.lags
        As, bs, Vs = self.As, self.bs, self.Vs

        obs = data['data']
        inpt = data['input']
        T = obs.shape[0]


        # Instantaneous inputs, lagged data, and bias
        mus = np.matmul(Vs[None, ...], inpt[L:, None, :M, None])[:, :, :, 0]
        for l in range(L):
            mus += As[:, :, l] * data[L-l-1:-l-1, None, :]
        mus += bs

        # Pad with the initial condition
        mus = np.concatenate((self.mu_init * np.ones((L, K, D)), mus))
        assert mus.shape == (T, K, D)

        return mus

    def log_likelihoods(self, data):
        L = self.lags
        obs = data['data']
        mus = self._compute_mus(data)

        # Compute the likelihood of the initial data and remainder separately
        ll_init = stats.diagonal_gaussian_logpdf(obs[:L, None, :], mus[:L], self.sigmasq_init)
        ll_ar = stats.diagonal_gaussian_logpdf(obs[L:, None, :], mus[L:], self.sigmasq)
        return np.row_stack((ll_init, ll_ar))

    def m_step(self, expectations, dataset, **kwargs):
        K, D, M, L = self.num_states, self.observation_dim, self.input_dim, self.lags

        for d in range(D):
            # Collect data for this dimension
            xs, ys, weights = [], [], []
            for (Ez, _, _), data in zip(expectations, dataset):
                obs = data['data']
                mask = data['mask']
                inpt = data['input']
                T = obs.shape[0]

                # Only use data if it is complete
                if np.all(mask[:, d]):
                    xs.append(
                        np.hstack([obs[L-l-1:-l-1, d:d+1] for l in range(L)]
                                  + [inpt[L:, :M], np.ones((T-L, 1))]))
                    ys.append(data[L:, d])
                    weights.append(Ez[L:])

            xs = np.concatenate(xs)
            ys = np.concatenate(ys)
            weights = np.concatenate(weights)

            # If there was no data for this dimension then skip it
            if len(xs) == 0:
                self.As[:, d, :] = 0
                self.Vs[:, d, :] = 0
                self.bs[:, d] = 0
                continue

            # Otherwise, fit a weighted linear regression for each discrete state
            for k in range(K):
                # Check for zero weights (singular matrix)
                if np.sum(weights[:, k]) < self.lags + M + 1:
                    self.As[k, d] = 1.0
                    self.Vs[k, d] = 0
                    self.bs[k, d] = 0
                    self._log_sigmasq[k, d] = 0
                    continue

                # Solve for the most likely A,V,b (no prior)
                Jk = np.sum(weights[:, k][:, None, None] * xs[:,:,None] * xs[:, None,:], axis=0)
                hk = np.sum(weights[:, k][:, None] * xs * ys[:, None], axis=0)
                muk = np.linalg.solve(Jk, hk)

                self.As[k, d] = muk[:L]
                self.Vs[k, d] = muk[L:L+M]
                self.bs[k, d] = muk[-1]

                # Update the variances
                yhats = xs.dot(np.concatenate((self.As[k, d], self.Vs[k, d], [self.bs[k, d]])))
                sqerr = (ys - yhats)**2
                sigma = np.average(sqerr, weights=weights[:, k], axis=0) + 1e-16
                self._log_sigmasq[k, d] = np.log(sigma)

    def sample_observation(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, sigmas = self.observation_dim, self.As, self.bs, self.sigmasq

        # Sample the initial condition
        if xhist.shape[0] < self.lags:
            sigma_init = self.sigmasq_init[z] if with_noise else 0
            return self.mu_init[z] + np.sqrt(sigma_init) * npr.randn(D)

        # Otherwise sample the AR model
        muz = bs[z].copy()
        for lag in range(self.lags):
            muz += As[z, :, lag] * xhist[-lag - 1]

        sigma = sigmas[z] if with_noise else 0
        return muz + np.sqrt(sigma) * npr.randn(D)


# Robust autoregressive models with diagonal Student's t noise
class _RobustAutoRegressiveObservationsMixin(object):
    """
    Mixin for AR models where the noise is distributed according to a
    multivariate t distribution,

        epsilon ~ t(0, Sigma, nu)

    which is equivalent to,

        tau ~ Gamma(nu/2, nu/2)
        epsilon | tau ~ N(0, Sigma / tau)

    We use this equivalence to perform the M step (update of Sigma and tau)
    via an inner expectation maximization algorithm.

    This mixin mus be used in conjunction with either AutoRegressiveObservations or
    AutoRegressiveDiagonalNoiseObservations, which provides the parameterization for
    Sigma.  The mixin does not capitalize on structure in Sigma, so it will pay
    a small complexity penalty when used in conjunction with the diagonal noise model.
    """
    def __init__(self, num_states, observation_dim, input_dim=0, lags=1):
        super(_RobustAutoRegressiveObservationsMixin, self).\
            __init__(num_states, observation_dim, input_dim=input_dim, lags=lags)
        self._log_nus = np.log(4) * np.ones(num_states)

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @property
    def params(self):
        return super(_RobustAutoRegressiveObservationsMixin, self).params + (self._log_nus,)

    @params.setter
    def params(self, value):
        self._log_nus = value[-1]
        super(_RobustAutoRegressiveObservationsMixin, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        super(_RobustAutoRegressiveObservationsMixin, self).permute(perm)
        self._log_nus = self._log_nus[perm]

    def log_likelihoods(self, data):
        assert np.all(data['mask']), \
            "Cannot compute likelihood of autoregressive obsevations with missing data."
        obs = data['data']
        mus = self._compute_mus(data)

        # Compute the likelihood of the initial data and remainder separately
        L = self.lags
        ll_init = stats.multivariate_normal_logpdf(obs[:L, None, :], mus[:L], self.Sigmas_init)
        ll_ar = stats.multivariate_studentst_logpdf(obs[L:, None, :], mus[L:], self.Sigmas, self.nus)
        return np.row_stack((ll_init, ll_ar))

    def m_step(self, expectations, dataset, num_em_iters=1, optimizer="adam", num_iters=10, **kwargs):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t
        for complete details.
        """
        self._m_step_ar(expectations, dataset, num_em_iters)
        self._m_step_nu(expectations, dataset, optimizer, num_iters, **kwargs)

    def _m_step_ar(self, expectations, dataset, num_em_iters):
        K, D, M, L = self.num_states, self.observation_dim, self.input_dim, self.lags

        # Collect data for this dimension
        xs, ys, Ezs = [], [], []
        for (Ez, _, _), data in zip(expectations, dataset):
            obs = data['data']
            mask = data['mask']
            inpt = data['input']
            T = obs.shape[0]

            # Only use data if it is complete
            if not np.all(mask):
                raise Exception("Encountered missing data in AutoRegressiveObservations!")

            xs.append(
                np.hstack([obs[L-l-1:-l-1] for l in range(L)]
                          + [inpt[L:, :M], np.ones((T-L, 1))]))
            ys.append(obs[L:])
            Ezs.append(Ez[L:])

        for itr in range(num_em_iters):
            # E Step: compute expected precision for each data point given current parameters
            taus = []
            for x, y in zip(xs, ys):
                Afull = np.concatenate((self.As, self.Vs, self.bs[:, :, None]), axis=2)
                mus = np.matmul(Afull[None, :, :, :], x[:, None, :, None])[:, :, :, 0]

                # nu: (K,)  mus: (T, K, D)  sigmas: (K, D, D)  y: (T, D)  -> tau: (T, K)
                alpha = self.nus / 2 + D/2
                sqrt_Sigmas = np.linalg.cholesky(self.Sigmas)
                beta = self.nus / 2 + 1/2 * stats.batch_mahalanobis(sqrt_Sigmas, y[:, None, :] - mus)
                taus.append(alpha / beta)

            # M step: Fit the weighted linear regressions for each K and D
            # This is exactly the same as the M-step for the AutoRegressiveObservations,
            # but it has an extra scaling factor of tau applied to the weight.
            J = np.tile(1e-8 * np.eye(D * L + M + 1)[None, :, :], (K, 1, 1))
            h = np.zeros((K, D * L + M + 1, D))
            for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
                weight = Ez * tau
                J += np.einsum('tk, ti, tj -> kij', weight, x, x)
                h += np.einsum('tk, ti, td -> kid', weight, x, y)

            mus = np.linalg.solve(J, h)
            self.As = np.swapaxes(mus[:, :D * L, :], 1, 2)
            self.Vs = np.swapaxes(mus[:, D * L:D * L + M, :], 1, 2)
            self.bs = mus[:, -1, :]

            # Update the covariance
            sqerr = np.zeros((K, D, D))
            weight = np.zeros(K)
            for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
                yhat = np.matmul(x[None, :, :], mus)
                resid = y[None, :, :] - yhat
                sqerr += np.einsum('tk,kti,ktj->kij', Ez * tau, resid, resid)
                weight += np.sum(Ez, axis=0)

            self.Sigmas = sqerr / weight[:, None, None] + 1e-8 * np.eye(D)

    def _m_step_nu(self, expectations, dataset, optimizer, num_iters, **kwargs):
        """
        Update the degrees of freedom parameter of the multivariate t distribution
        using a generalized Newton update. See notes in the ssm repo.
        """
        K, D, L = self.num_states, self.observation_dim, self.lags
        E_taus = np.zeros(K)
        E_logtaus = np.zeros(K)
        weights = np.zeros(K)
        for (Ez, _, _,), data in zip(expectations, dataset):
            obs = data['data']

            # nu: (K,)  mus: (K, D)  Sigmas: (K, D, D)  y: (T, D)  -> tau: (T, K)
            mus = self._compute_mus(data)
            alpha = self.nus/2 + D/2
            sqrt_Sigma = np.linalg.cholesky(self.Sigmas)
            beta = self.nus/2 + 1/2 * stats.batch_mahalanobis(sqrt_Sigma, obs[L:, None, :] - mus[L:])

            E_taus += np.sum(Ez[L:, :] * alpha / beta, axis=0)
            E_logtaus += np.sum(Ez[L:, :] * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights
        E_logtaus /= weights

        for k in range(K):
            self._log_nus[k] = np.log(generalized_newton_studentst_dof(E_taus[k], E_logtaus[k]))

    def sample_observation(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, Sigmas, nus = self.observation_dim, self.As, self.bs, self.Sigmas, self.nus
        if xhist.shape[0] < self.lags:
            S = np.linalg.cholesky(self.Sigmas_init[z]) if with_noise else 0
            return self.mu_init[z] + np.dot(S, npr.randn(D))
        else:
            mu = bs[z].copy()
            for l in range(self.lags):
                mu += As[z][:,l*D:(l+1)*D].dot(xhist[-l-1])

            tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
            S = np.linalg.cholesky(Sigmas[z] / tau) if with_noise else 0
            return mu + np.dot(S, npr.randn(D))


class RobustAutoRegressiveObservations(_RobustAutoRegressiveObservationsMixin, AutoRegressiveObservations):
    """
    AR model where the noise is distributed according to a multivariate t distribution,

        epsilon ~ t(0, Sigma, nu)

    which is equivalent to,

        tau ~ Gamma(nu/2, nu/2)
        epsilon | tau ~ N(0, Sigma / tau)

    Here, Sigma is a general covariance matrix.
    """
    pass


class RobustAutoRegressiveDiagonalNoiseObservations(
    _RobustAutoRegressiveObservationsMixin, AutoRegressiveDiagonalNoiseObservations):
    """
    AR model where the noise is distributed according to a multivariate t distribution,

        epsilon ~ t(0, Sigma, nu)

    which is equivalent to,

        tau ~ Gamma(nu/2, nu/2)
        epsilon | tau ~ N(0, Sigma / tau)

    Here, Sigma is a diagonal covariance matrix.
    """
    pass

# Robust autoregressive models with diagonal Student's t noise
class AltRobustAutoRegressiveDiagonalNoiseObservations(AutoRegressiveDiagonalNoiseObservations):
    """
    An alternative formulation of the robust AR model where the noise is
    distributed according to a independent scalar t distribution,

    For each output dimension d,

        epsilon_d ~ t(0, sigma_d^2, nu_d)

    which is equivalent to,

        tau_d ~ Gamma(nu_d/2, nu_d/2)
        epsilon_d | tau_d ~ N(0, sigma_d^2 / tau_d)

    """
    def __init__(self, num_states, observation_dim, input_dim=0, lags=1):
        super(AltRobustAutoRegressiveDiagonalNoiseObservations, self).\
            __init__(num_states, observation_dim, input_dim=input_dim, lags=lags)
        self._log_nus = np.log(4) * np.ones((num_states, observation_dim))

    @property
    def nus(self):
        return np.exp(self._log_nus)

    @property
    def params(self):
        return self.As, self.bs, self.Vs, self._log_sigmasq, self._log_nus

    @params.setter
    def params(self, value):
        self.As, self.bs, self.Vs, self._log_sigmasq, self._log_nus = value

    def permute(self, perm):
        super(AltRobustAutoRegressiveDiagonalNoiseObservations, self).permute(perm)
        self.inv_nus = self.inv_nus[perm]

    def log_likelihoods(self, data):
        L = self.lags
        obs = data['data']
        mask = data['mask']
        assert np.all(mask), \
            "Cannot compute likelihood of autoregressive obsevations with missing data."
        mus = self._compute_mus(data)

        # Compute the likelihood of the initial data and remainder separately
        ll_init = stats.diagonal_gaussian_logpdf(obs[:L, None, :], mus[:L], self.sigmasq_init)
        ll_ar = stats.independent_studentst_logpdf(obs[L:, None, :], mus[L:], self.sigmasq, self.nus)
        return np.row_stack((ll_init, ll_ar))

    def m_step(self, expectations, dataset, num_em_iters=1, optimizer="adam", num_iters=10, **kwargs):
        """
        Student's t is a scale mixture of Gaussians.  We can estimate its
        parameters using the EM algorithm. See the notebook in doc/students_t
        for complete details.
        """
        self._m_step_ar(expectations, dataset, num_em_iters)
        self._m_step_nu(expectations, dataset, optimizer, num_iters, **kwargs)

    def _m_step_ar(self, expectations, dataset, num_em_iters):
        K, D, M, L = self.num_states, self.observation_dim, self.input_dim, self.lags

        # Collect data for this dimension
        xs, ys, Ezs = [], [], []
        for (Ez, _, _), data in zip(expectations, dataset):
            obs = data['data']
            mask = data['mask']
            inpt = data['input']
            T = obs.shape[0]

            # Only use data if it is complete
            if not np.all(mask):
                raise Exception("Encountered missing data in AutoRegressiveObservations!")

            xs.append(
                np.hstack([data[L-l-1:-l-1] for l in range(L)]
                          + [inpt[L:, :M], np.ones((data.shape[0]-L, 1))]))
            ys.append(data[L:])
            Ezs.append(Ez[L:])

        for itr in range(num_em_iters):
            # E Step: compute expected precision for each data point given current parameters
            taus = []
            for x, y in zip(xs, ys):
                # mus = self._compute_mus(data, input, mask, tag)
                # sigmas = self._compute_sigmas(data, input, mask, tag)
                Afull = np.concatenate((self.As, self.Vs, self.bs[:, :, None]), axis=2)
                mus = np.matmul(Afull[None, :, :, :], x[:, None, :, None])[:, :, :, 0]

                # nu: (K,D)  mus: (T, K, D)  sigmas: (K, D)  y: (T, D)  -> tau: (T, K, D)
                alpha = self.nus / 2 + 1/2
                beta = self.nus / 2 + 1/2 * (y[:, None, :] - mus)**2 / self.sigmasq
                taus.append(alpha / beta)

            # M step: Fit the weighted linear regressions for each K and D
            J = np.tile(np.eye(D * L + M + 1)[None, None, :, :], (K, D, 1, 1))
            h = np.zeros((K, D,  D * L + M + 1,))
            for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
                robust_ar_statistics(Ez, tau, x, y, J, h)

            mus = np.linalg.solve(J, h)
            self.As = mus[:, :, :D * L]
            self.Vs = mus[:, :, D * L:D * L + M]
            self.bs = mus[:, :, -1]

            # Fit the variance
            sqerr = 0
            weight = 0
            for x, y, Ez, tau in zip(xs, ys, Ezs, taus):
                yhat = np.matmul(x[None, :, :], np.swapaxes(mus, -1, -2))
                sqerr += np.einsum('tk, tkd, ktd -> kd', Ez, tau, (y - yhat)**2)
                weight += np.sum(Ez, axis=0)
            self._log_sigmasq = np.log(sqerr / weight[:, None] + 1e-16)

    def _m_step_nu(self, expectations, dataset, optimizer, num_iters, **kwargs):
        K, D, L = self.num_states, self.observation_dim, self.lags
        E_taus = np.zeros((K, D))
        E_logtaus = np.zeros((K, D))
        weights = np.zeros(K)
        for (Ez, _, _,), data in zip(expectations, dataset):
            obs = data['data']
            mask = data['mask']
            inpt = data['input']
            T = obs.shape[0]

            # nu: (K,D)  mus: (K, D)  sigmas: (K, D)  y: (T, D)  -> w: (T, K, D)
            mus = self._compute_mus(obs, inpt, mask, tag)
            alpha = self.nus/2 + 1/2
            beta = self.nus/2 + 1/2 * (obs[L:, None, :] - mus[L:])**2 / self.sigmasq

            E_taus += np.sum(Ez[L:, :, None] * alpha / beta, axis=0)
            E_logtaus += np.sum(Ez[L:, :, None] * (digamma(alpha) - np.log(beta)), axis=0)
            weights += np.sum(Ez, axis=0)

        E_taus /= weights[:, None]
        E_logtaus /= weights[:, None]

        for k in range(K):
            for d in range(D):
                self._log_nus[k, d] = np.log(generalized_newton_studentst_dof(E_taus[k, d], E_logtaus[k, d]))

    def sample_observation(self, z, xhist, input=None, tag=None, with_noise=True):
        D, As, bs, sigmasq, nus = self.observation_dim, self.As, self.bs, self.sigmasq, self.nus
        if xhist.shape[0] < self.lags:
            sigma_init = self.sigmasq_init[z] if with_noise else 0
            return self.mu_init[z] + np.sqrt(sigma_init) * npr.randn(D)
        else:
            mu = bs[z].copy()
            for l in range(self.lags):
                mu += As[z][:,l*D:(l+1)*D].dot(xhist[-l-1])

            tau = npr.gamma(nus[z] / 2.0, 2.0 / nus[z])
            var = sigmasq[z] / tau if with_noise else 0
            return mu + np.sqrt(var) * npr.randn(D)


class VonMisesObservations(_Observations):
    def __init__(self, num_states, observation_dim, input_dim=0):
        super(VonMisesObservations, self).__init__(num_states, observation_dim, input_dim)
        self.mus = npr.randn(num_states, observation_dim)
        self.log_kappas = np.log(-1*npr.uniform(low=-1, high=0, size=(num_states, observation_dim)))

    @property
    def params(self):
        return self.mus, self.log_kappas

    @params.setter
    def params(self, value):
        self.mus, self.log_kappas = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self.log_kappas = self.log_kappas[perm]

    @check_dataset
    def initialize(self, dataset):
        # TODO: add spherical k-means for initialization
        pass

    def log_likelihoods(self, data):
        mus, kappas = self.mus, np.exp(self.log_kappas)
        obs = data['data']
        mask = data['mask']
        return stats.vonmises_logpdf(obs[:, None, :], mus, kappas, mask=mask[:, None, :])

    def sample_observation(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, kappas = self.observation_dim, self.mus, np.exp(self.log_kappas)
        return npr.vonmises(self.mus[z], kappas[z], D)

    def m_step(self, expectations, dataset, **kwargs):

        x = np.concatenate([data['data'] for data in dataset])
        weights = np.concatenate([Ez for Ez, _, _ in expectations])  # T x D
        assert x.shape[0] == weights.shape[0]

        # convert angles to 2D representation and employ closed form solutions
        x_k = np.stack((np.sin(x), np.cos(x)), axis=1)  # T x 2 x D

        r_k = np.tensordot(weights.T, x_k, axes=1)  # K x 2 x D
        r_norm = np.sqrt(np.sum(np.power(r_k, 2), axis=1))  # K x D

        mus_k = np.divide(r_k, r_norm[:, None])  # K x 2 x D
        r_bar = np.divide(r_norm, np.sum(weights, 0)[:, None])  # K x D

        mask = (r_norm.sum(1) == 0)
        mus_k[mask] = 0
        r_bar[mask] = 0

        # Approximation
        kappa0 = r_bar * (self.observation_dim + 1 - np.power(r_bar, 2)) / (1 - np.power(r_bar, 2))  # K,D

        kappa0[kappa0 == 0] += 1e-6

        for k in range(self.num_states):
            self.mus[k] = np.arctan2(*mus_k[k])  #
            self.log_kappas[k] = np.log(kappa0[k])  # K, D

    def smooth(self, expectations, data):
        mus = self.mus
        return expectations.dot(mus)
