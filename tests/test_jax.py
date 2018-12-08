import numpy as onp

import jax.config
jax.config.config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import random
from jax import jit, grad 
from jax.scipy.misc import logsumexp

import numpy.random as npr

from ssm.messages import forward_pass, backward_pass, grad_hmm_normalizer

@jit 
def forward_pass_np(log_pi0, log_Ps, log_likes):
    T, K = log_likes.shape
    alphas = []
    alphas.append(log_likes[0] + log_pi0)
    for t in range(T-1):
        anext = logsumexp(alphas[t] + log_Ps[t].T, axis=1)
        anext += log_likes[t+1]
        alphas.append(anext)
    return np.array(alphas)

@jit
def hmm_normalizer_np(log_pi0, log_Ps, ll):
    alphas = forward_pass_np(log_pi0, log_Ps, ll)    
    Z = logsumexp(alphas[-1])
    return Z


def make_parameters(T, K):
    log_pi0 = -np.log(K) * np.ones(K)
    As = npr.rand(T-1, K, K)
    As /= As.sum(axis=2, keepdims=True)
    log_Ps = np.log(As)
    ll = npr.randn(T, K)
    return log_pi0, log_Ps, ll

if __name__ == "__main__":
    T = 10
    K = 3

    log_pi0, log_Ps, lls = make_parameters(T, K)
    np_log_pi0, np_log_Ps, np_lls = onp.copy(log_pi0), onp.copy(log_Ps), onp.copy(lls) 

    # Construct numpy arrays and compute gradients with cython
    dlog_pi0, dlog_Ps, dlls = onp.zeros_like(log_pi0), onp.zeros_like(log_Ps), onp.zeros_like(lls)
    alphas = onp.zeros((T, K))
    forward_pass(-onp.log(K) * onp.ones(K), onp.copy(log_Ps), onp.copy(lls), alphas)
    grad_hmm_normalizer(onp.copy(log_Ps), alphas, dlog_pi0, dlog_Ps, dlls)

    # Compare manual grads to JAX grads
    assert np.allclose(dlog_pi0, grad(hmm_normalizer_np, argnums=0)(log_pi0, log_Ps, lls))
    assert np.allclose(dlog_Ps, grad(hmm_normalizer_np, argnums=1)(log_pi0, log_Ps, lls))
    assert np.allclose(dlls, grad(hmm_normalizer_np, argnums=2)(log_pi0, log_Ps, lls))
