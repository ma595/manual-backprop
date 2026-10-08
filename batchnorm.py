import numpy as np
from torch.nn import BatchNorm1d
from torch import tensor
import torch

import torch.nn as nn

from src.ops import *

torch.manual_seed(0)
np.random.seed(42)

# Set X_new
# X_new = np.random.randn(*X.shape).astype(np.float32)
# # Set non-zero upstream gradient
# dldp_np = np.random.randn(*X.shape).astype(np.float32)


def init_data(examples, feats):
    torch.manual_seed(0)
    return torch.rand((examples, feats), dtype=torch.float64)


def bn_manual(X, eps=1e-5):
    gamma = np.ones((X.shape[1],))
    bias = np.zeros((X.shape[1],))
    ## Summation
    sum0 = np.sum(X, axis=0)  # reduction # collapse: vector of size 5 i.
    mean = (1.0 / len(X)) * sum0  # compute mean
    num = sub(X, mean)  # size (3,5) # broadcasting (possible extra op)

    # Denominator
    ## Variance
    b = pow(num, 2)  # size (3, 5) # applied elementwise # num
    sum0_ = np.sum(b, axis=0)  # collapse: vector of size 5 i.e. feats
    var = (1.0 / len(X)) * sum0_  # (5,)
    var_eps = add(var, eps)  # Elementwise: (5,) + scalar
    denom_vec = power(var_eps, -0.5)  # Pure Power: (5,)

    # --- OUTPUT ---
    bn = hadamard(num, denom_vec)  # (3, 5) * (3, 5) # num
    bn_gamma = hadamard(bn, gamma)  # (3 * 5) * (1 * 5)

    bn_gamma_bias = add(bn_gamma, bias)

    return bn_gamma_bias


def fwd_backwd_bn(
    X,
    dldp,
    eps=1e-5,
):
    eps = np.array(eps)
    gamma = np.ones((X.shape[1],), dtype=np.float32)
    bias = np.zeros((X.shape[1],), np.float32)
    ## Summation
    sum0 = np.sum(X, axis=0)  # reduction # collapse: vector of size 5 i.
    mean = (1.0 / len(X)) * sum0  # compute mean
    num = sub(X, mean)  # size (3,5) # broadcasting (possible extra op)

    # Denominator
    ## Variance
    b = pow(num, 2.0)  # size (3, 5) # applied elementwise # num
    sum0_ = np.sum(b, axis=0)  # collapse: vector of size 5 i.e. feats
    var = (1.0 / len(X)) * sum0_  # (5,)
    var_eps = add(var, eps)  # Elementwise: (5,) + scalar broadcast
    denom_vec = power(var_eps, -0.5)  # Pure Power: (5,)

    # --- OUTPUT ---
    bn = hadamard(num, denom_vec)  # (3, 5) * (3, 5) # num
    bn_gamma = hadamard(bn, gamma)  # (3 * 5) * (1 * 5)
    bn_gamma_bias = add(bn_gamma, bias)

    fwd = bn_gamma_bias

    ## BACKWARD:
    # bn_gamma_bias = add(bn_gamma, bias_)
    dldbn_gamma, grad_bias = vjp_add(bn_gamma, bias, dldp)  # we return dlbn_gamma

    dldbn, grad_gamma = vjp_hadamard(bn, gamma, dldbn_gamma)

    # Collapse gamma (3, 5) -> gamma (5,) for output
    grad_gamma = unbroadcast(grad_gamma, gamma.shape)

    # bn = hadamard(num, denom_b)  # (3, 5) * (3, 5)
    dldnum, dldenom_b = vjp_hadamard(num, denom_vec, dldbn)

    # denom_vec = power(var_eps, -0.5)  # Pure Power: (5,)
    dldvar_eps = vjp_power(var_eps, -0.5, dldenom_b)

    # var_eps = add(var, eps_b)  # Elementwise: (5,) + scalar
    dldvar, dldeps = vjp_add(var, eps, dldvar_eps)

    # var = (1.0 / len(X)) * sum0_  # (5,)
    dldsum0_ = vjp_scale(1.0 / len(X), dldvar)

    ## Numerator
    # b = pow(num, 2)  # size (3, 5) # applied elementwise
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


def test_bn_fwd(data):
    bn = BatchNorm1d(num_features=data.shape[1])
    data_cpy = data.detach().cpu().numpy()
    assert np.allclose(bn(data).detach().cpu().numpy(), bn_manual(X=data_cpy))


def test_backward_bn(data):
    data_cpy = data.detach().clone().numpy()
    data.requires_grad = True
    model = BatchNorm1d(num_features=data.shape[1], dtype=torch.float64)
    model.zero_grad()
    fwd_pt = model(data)

    dldp_np = np.random.randn(*data.shape).astype(np.float64)
    fwd_pt.backward(tensor(dldp_np))
    params = model.parameters()
    params_l = list(params)

    grads = dict()

    for name, param in model.named_parameters():
        if param.requires_grad:
            grads[name] = param.grad

    fwd, dldX, dldgamma, dldbias = fwd_backwd_bn(data_cpy, dldp_np)

    grads["X"] = data.grad

    # now do assertions
    np.testing.assert_allclose(grads["weight"].detach().cpu().numpy(), dldgamma)
    np.testing.assert_allclose(grads["bias"].detach().cpu().numpy(), dldbias)
    np.testing.assert_allclose(grads["X"].detach().cpu().numpy(), dldX)
    np.testing.assert_allclose(fwd_pt.detach().cpu().numpy(), fwd)


def main():
    features = 5
    examples = 3

    X = init_data(examples, features)
    # test_bn_fwd(X)
    # fwd, dldX, dldgamma, dldbias = fwd_backwd(X_new, dldp_np)
    test_backward_bn(X)
    # np.pow(a,-1)


if __name__ == "__main__":
    main()
