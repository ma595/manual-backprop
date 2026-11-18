import numpy as np
from torch.nn import BatchNorm1d
from torch import tensor
import torch

import torch.nn as nn

from src.ops import *

torch.manual_seed(0)
np.random.seed(0)

# Set X_new
# X_new = np.random.randn(*X.shape).astype(np.float32)
# # Set non-zero upstream gradient
# dldp_np = np.random.randn(*X.shape).astype(np.float32)


def init_data(examples, feats):
    torch.manual_seed(0)
    return torch.rand((examples, feats), dtype=torch.float64)


def fwd_backwd_bn(
    X,
    dldp,
    gamma=None,
    bias=None,
    eps=1e-5,
):

    if X.shape[0] < 2:
        raise ValueError(
            f"batch norm needs more than 1 sample per feature, got shape {X.shape}"
        )

    # bn can be expressed simply as this op:
    # y_ij = γ_j x̂_ij + β_j
    # dl/dx̂_ij = γ_j g_ij
    # dl/dγ_j  = Σ_i g_ij x̂_ij
    # dl/dβ_j  = Σ_i g_ij
    # dl/dX    = (depends on dl/dx̂, so on γ)
    # bias corresponds to beta
    # gamma is 'weight' in torch land
    eps = np.array(eps)
    if gamma is None:
        gamma = np.ones((X.shape[1],), dtype=np.float64)
    if bias is None:
        bias = np.zeros((X.shape[1],), np.float64)

    ## Summation
    sum0 = np.sum(X, axis=0)  # reduction # collapse: vector of size 5 i.
    mean = (1.0 / len(X)) * sum0  # compute mean
    num = sub(X, mean)  # shape (3,5) # broadcasting (possible extra op)
    # Denominator
    ## Variance
    b = pow(num, 2.0)  # shape (3,5) # applied elementwise # num
    sum0_ = np.sum(b, axis=0)  # collapse: vector of size 5 i.e. feats
    var = (1.0 / len(X)) * sum0_  # shape (5,)
    var_eps = add(var, eps)  # Elementwise: (5,) + scalar broadcast
    denom_vec = power(var_eps, -0.5)  # Pure Power: (3,5)

    # --- OUTPUT ---
    xhat = hadamard(num, denom_vec)  # (3,5) * (3,5) # num
    xhat_gamma = hadamard(xhat, gamma)  # (3*5) * (1*5)
    bn_gamma_bias = add(
        xhat_gamma, bias
    )  # Ex: if gamma were ones we could put xhat in instead

    fwd = bn_gamma_bias

    ## BACKWARD:
    # bn_gamma_bias = add(bn_gamma, bias_)
    dldbn_gamma, grad_bias = vjp_add(xhat_gamma, bias, dldp)  # we return dlbn_gamma

    dldbn, grad_gamma = vjp_hadamard(xhat, gamma, dldbn_gamma)

    # bn = hadamard(num, denom_b)  # (3,5) * (3,5)
    dldnum, dldenom_b = vjp_hadamard(num, denom_vec, dldbn)

    # denom_vec = power(var_eps, -0.5)  # Pure Power: (5,)
    dldvar_eps = vjp_power(var_eps, -0.5, dldenom_b)

    # var_eps = add(var, eps_b)  # Elementwise: (5,) + scalar
    dldvar, dldeps = vjp_add(var, eps, dldvar_eps)

    # var = (1.0 / len(X)) * sum0_  # (5,)
    dldsum0_ = vjp_scale(1.0 / len(X), dldvar)

    ## Numerator
    # b = pow(num, 2)  # size (3,5) # applied elementwise
    dldnum += vjp_power(num, 2.0, dldsum0_)

    # num = sub(X, meanb)  # size (3,5) # broadcasting (possible extra op)
    grad_X, dldmean = vjp_sub(
        X, mean, dldnum
    )  # size (3,5) # broadcasting (possible extra op)

    # mean = (1.0 / len(X)) * sum0  # compute mean
    dldsum0 = vjp_scale(1.0 / len(X), dldmean)

    # sum0 = np.sum(X, axis=0)  # reduction # collapse: vector of size 5 i.
    grad_X += dldsum0

    return fwd, grad_X, grad_gamma, grad_bias


def main():
    features = 5
    examples = 3

    X = init_data(examples, features)
    # test_bn_fwd(X)
    # fwd, dldX, dldgamma, dldbias = fwd_backwd(X_new, dldp_np)
    # test_fwd_bckwd_bn(X) # now moved to tests.py
    # np.pow(a,-1)


if __name__ == "__main__":
    main()
