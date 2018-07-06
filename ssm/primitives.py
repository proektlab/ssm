# Define an autograd extension for HMM normalizer
import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from autograd.extend import primitive, defvjp
from functools import partial

from ssm.messages import forward_pass, backward_pass, grad_hmm_normalizer

@primitive
def hmm_normalizer(log_pi0, log_Ps, ll):
    T, K = ll.shape
    alphas = np.zeros((T, K))
    forward_pass(log_pi0, log_Ps, ll, alphas)    
    return logsumexp(alphas[-1])
    
def _make_grad_hmm_normalizer(argnum, ans, log_pi0, log_Ps, ll):
    # Unbox the inputs if necessary
    unbox = lambda x: x if isinstance(x, np.ndarray) else x._value
    log_pi0 = unbox(log_pi0)
    log_Ps = unbox(log_Ps)
    ll = unbox(ll)

    dlog_pi0 = np.zeros_like(log_pi0)
    dlog_Ps= np.zeros_like(log_Ps)
    dll = np.zeros_like(ll)
    T, K = ll.shape
    
    # Forward pass to get alphas
    alphas = np.zeros((T, K))
    forward_pass(log_pi0, log_Ps, ll, alphas)
    grad_hmm_normalizer(log_Ps, alphas, dlog_pi0, dlog_Ps, dll)
    
    if argnum == 0:
        return lambda g: g * dlog_pi0
    if argnum == 1:
        return lambda g: g * dlog_Ps
    if argnum == 2:
        return lambda g: g * dll

defvjp(hmm_normalizer, 
       partial(_make_grad_hmm_normalizer, 0),
       partial(_make_grad_hmm_normalizer, 1),
       partial(_make_grad_hmm_normalizer, 2))


def hmm_expected_states(log_pi0, log_Ps, ll):
    T, K = ll.shape
    alphas = np.zeros((T, K))
    forward_pass(log_pi0, log_Ps, ll, alphas)    
    betas = np.zeros((T, K))
    backward_pass(log_Ps, ll, betas)    

    expected_states = alphas + betas
    expected_states -= logsumexp(expected_states, axis=1, keepdims=True)
    expected_states = np.exp(expected_states)
    
    expected_joints = alphas[:-1,:,None] + betas[1:,None,:] + ll[1:,None,:] + log_Ps
    expected_joints -= expected_joints.max((1,2))[:,None, None]
    expected_joints = np.exp(expected_joints)
    
    return expected_states, expected_joints


### LDS helpers
#
# We need to evaluate log N(y | J, h) where J is a block tridiagonal precision matrix
# 
# J: TD x TD matrix
# h: TD arry
#
# log N(y | J, h) = -TD/2 log 2\pi + 1/2 log |J| - 1/2 y^T J y - 1/2 h^T y - 1/2 h^T J^{-1} h
#                 = -TD/2 log 2\pi + \sum L_{ii} - 1/2 y^T J y - 1/2 h^T y - 1/2 h^T J^{-1} h
#
# where L = cholesky(J, lower=True)
# 
# In order to compute this, we need gradients of cholesky_banded and solveh_banded, 
# both of which are available in scipy.linalg.
###
import scipy.linalg as spla
def grad_solveh_banded(argnum, ans, ab, b):
    """
    Gradient of solveh_banded, which solves Ax=b when A is banded and Hermitian.
    """
    updim = lambda x: x if x.ndim == 2 else x[...,None]

    # d(A^-1 b)/dA * g = -(A^{-T} g) (A^-1 b)^T
    # where g = dL/d(A^-1 b) is the same shape as b
    def dy_dab(g):
        nAinv_g = -updim(solveh_banded(ab, g))
        ans = updim(ans)

        # It would be costly to take the outer product of these two just 
        # to throw away all but the banded parts. Instead, compute only
        # the necessary pieces. 
        # NOTE: THIS ASSUMES LOWER=TRUE!!!
        # NOTE: THIS IS NOT AUTOMATICALLY DIFFERENTIABLE BY AUTOGRAD!!!
        H, W = ab.shape
        out = np.zeros((H, W))
        out[0] = np.sum(nAinv_g * ans, axis=1)
        for h in range(1, H):
            # Multiply by 2 to account for upper and lower contribution
            out[h, :W-h] = 2 * np.sum(nAinv_g[h:] * ans[:-W-h], axis=1)
        return out

    # d(A^-1 b)/db * g = A^{-T} g
    # where g = dL/d(A^-1 b) is the same shape as b
    dy_db = lambda g: solveh_banded(ab, g)

    return dy_dab if argnum == 0 else dy_db

solveh_banded = primitive(spla.solveh_banded)
defvjp(solveh_banded, partial(grad_solveh_banded, 0), partial(grad_solveh_banded, 1))


def grad_cholesky_banded(Lab, ab):
    # Based on Iain Murray's note http://arxiv.org/abs/1602.07527
    # scipy's dtrtrs wrapper, solve_triangular, doesn't broadcast along leading
    # dimensions, so we just call a generic LU solve instead of directly using
    # backsubstitution (also, we factor twice...)
    solve_trans = lambda a, b: solve(T(a), b)
    
    def conjugate_solve(L, X):
        # X -> L^{-T} X L^{-1}
        return solve_trans(L, T(solve_trans(L, T(X))))

    def vjp(Gab):
        # Gab is the same shape as Lab
        H, W = Lab.shape
        assert Gab.shape == (H, W)

        # First we need to compute dot(L.T, G)
        # this is multiplying the columns of L and G, both of 
        # which are represented as banded matrices. Since both 
        # matrices are banded, their product is as well.  
        #
        # F = L^T G -> F_ij = \sum_h L_{hi} G_{hj}
        # 
        # but L_hi = 0 if h < i  or h > i + H
        # and G_hj = 0 if h < j  or h > j + H
        #
        # so the nonzero terms are h > max(i, j) and h < min (i + H, j + H).
        # We are only interested in the lower triangular part where i >= j,
        # so our working range is i < h < j + H. 
        #
        # This implies F_{ij} = 0 if i > j + H. In other words, the resulting 
        # matrix also has a bandwidth of H.
        Phi = np.zeros((H, W))
        for h in range(H):
            Phi[h] = np.sum(Lab[:, h:] * Gab[:, :(W-h)], axis=0)

        # Divide the diagonal by 2
        Phi[0] /= 2

        # tmp = solve(T(L), T(phi)) = (L.T)^{-1} Phi.T; shape is W x W
        # L^{-T} is upper triangular
        # Phi.T is a banded upper triangular matrix
        Uab = transpose_lower_banded_matrix(Lab)
        spla.solve_banded((0, H-1), Uab, Phi.T)

        # S = solve(T(L), T(tmp))
        S = conjugate_solve(L, phi(LTg))
        return (S + T(S)) / 2.
    return vjp
# defvjp(cholesky, grad_cholesky)

def logdet_banded_symmetric(ab):
    # Get the Cholesky factorization of the banded symmetric matrix 
    # in lower form (ab represents the lower bands).  Log determinant
    # is then twice the sum of the log of the diagonal.
    Lab = cholesky_banded(ab, lower=True)
    return 2 * np.sum(np.log(Lab[0]))


def transpose_lower_banded_matrix(Lab):
    # This is painful
    Uab = np.flipud(Lab)
    u = Uab.shape[0] - 1
    for i in range(1,u+1):
        Uab[-(i+1), i:] = Uab[-(i+1), :-i]
        Uab[-(i + 1), :i] = 0
    return Uab

def convert_block_tridiag_to_banded(H_diag, H_upper_diag, lower=True):
    """
    convert blocks to banded matrix representation required for scipy.
    we are using the "lower form."
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html
    """
    T, D, _ = H_diag.shape
    assert H_diag.ndim == 3 and H_diag.shape[2] == D
    assert H_upper_diag.shape == (T - 1, D, D)
    H_lower_diag = np.swapaxes(H_upper_diag, -2, -1)

    ab = np.zeros((2 * D, T * D))

    # Fill in blocks along the diagonal
    for d in range(D):
        # Get indices of (-d)-th diagonal of H_diag
        i = np.arange(d, D)
        j = np.arange(0, D - d)
        h = np.column_stack((H_diag[:, i, j], np.zeros((T, d))))
        ab[d] = h.ravel()

    # Fill in lower left corner of blocks below the diagonal
    for d in range(0, D):
        # Get indices of (-d)-th diagonal of H_diag
        i = np.arange(d, D)
        j = np.arange(0, D - d)
        h = np.column_stack((H_lower_diag[:, i, j], np.zeros((T - 1, d))))
        ab[D + d, :D * (T - 1)] = h.ravel()

    # Fill in upper corner of blocks below the diagonal
    for d in range(1, D):
        # Get indices of (+d)-th diagonal of H_lower_diag
        i = np.arange(0, D - d)
        j = np.arange(d, D)
        h = np.column_stack((np.zeros((T - 1, d)), H_lower_diag[:, i, j]))
        ab[D - d, :D * (T - 1)] += h.ravel()

    return ab if lower else transpose_lower_banded_matrix(ab)


def scipy_solve_symm_block_tridiag(H_diag, H_upper_diag, v, ab=None):
    """
    use scipy.linalg.solve_banded to solve a symmetric block tridiagonal system
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html
    """
    from scipy.linalg import solveh_banded
    ab = convert_block_tridiag_to_banded(H_diag, H_upper_diag) \
        if ab is None else ab
    x = solveh_banded(ab, v.ravel(), lower=True)
    return x.reshape(v.shape)


def scipy_sample_block_tridiag(H_diag, H_upper_diag, size=1, ab=None, z=None):
    from scipy.linalg import cholesky_banded, solve_banded

    ab = convert_block_tridiag_to_banded(H_diag, H_upper_diag, lower=False) \
        if ab is None else ab

    Uab = cholesky_banded(ab, lower=False)
    z = np.random.randn(ab.shape[1], size) if z is None else z

    # If lower = False, we have (U^T U)^{-1} = U^{-1} U^{-T} = AA^T = Sigma
    # where A = U^{-1}.  Samples are Az = U^{-1}z = x, or equivalently Ux = z.
    return solve_banded((0, Uab.shape[0]-1), Uab, z)
