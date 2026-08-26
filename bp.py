import torch
import numpy as np
from torch.nn import BatchNorm1d
from torch import tensor
import torch

import torch.nn as nn

from ops import * 

torch.manual_seed(0)
np.random.seed(42)

# Set X_new
# X_new = np.random.randn(*X.shape).astype(np.float32)
# # Set non-zero upstream gradient
# dldp_np = np.random.randn(*X.shape).astype(np.float32)


def init_data(examples, feats):
    torch.manual_seed(0)
    return torch.rand((examples, feats))


def bn_manual(X, eps=1e-5):
    gamma = np.ones((X.shape[1],))
    bias = np.zeros((X.shape[1],))
    ## Summation
    sum0 = np.sum(X, axis=0)  # reduction # collapse: vector of size 5 i.
    mean = (1.0 / len(X)) * sum0  # compute mean
    meanb = np.broadcast_to(mean, X.shape)
    num = sub(X, meanb)  # size (3,5) # broadcasting (possible extra op)

    # Denominator
    ## Variance
    b = pow(num, 2)  # size (3, 5) # applied elementwise # num
    sum0_ = np.sum(b, axis=0)  # collapse: vector of size 5 i.e. feats
    var = (1.0 / len(X)) * sum0_  # (5,)
    eps_b = np.broadcast_to(eps, var.shape)  # Broadcast: () -> (5,)
    var_eps = add(var, eps_b)  # Elementwise: (5,) + scalar
    denom_vec = power(var_eps, -0.5)  # Pure Power: (5,)
    denom_b = np.broadcast_to(denom_vec, X.shape)  # Broadcast:  (5,) -> (3, 5)

    # --- OUTPUT ---
    bn = hadamard(num, denom_b)  # (3, 5) * (3, 5) # num
    gamma_ = np.broadcast_to(gamma, X.shape)  # (5,) -> (3, 5)
    bn_gamma = hadamard(bn, gamma_)  # (3 * 5) * (1 * 5)

    bias_ = np.broadcast_to(bias, bn_gamma.shape)
    bn_gamma_bias = add(bn_gamma, bias_)

    return bn_gamma_bias


def fwd_backwd_bn(
    X,
    dldp,
    eps=1e-5,
):
    gamma = np.ones((X.shape[1],), dtype=np.float32)
    bias = np.zeros((X.shape[1],), np.float32)
    ## Summation
    sum0 = np.sum(X, axis=0)  # reduction # collapse: vector of size 5 i.
    mean = (1.0 / len(X)) * sum0  # compute mean
    meanb = np.broadcast_to(mean, X.shape)
    num = sub(X, meanb)  # size (3,5) # broadcasting (possible extra op)

    # Denominator
    ## Variance
    b = pow(num, 2.0)  # size (3, 5) # applied elementwise # num
    sum0_ = np.sum(b, axis=0)  # collapse: vector of size 5 i.e. feats
    var = (1.0 / len(X)) * sum0_  # (5,)
    eps_b = np.broadcast_to(eps, var.shape)  # Broadcast: () -> (5,)
    var_eps = add(var, eps_b)  # Elementwise: (5,) + scalar
    denom_vec = power(var_eps, -0.5)  # Pure Power: (5,)
    denom_b = np.broadcast_to(denom_vec, X.shape)  # Broadcast:  (5,) -> (3, 5)

    # --- OUTPUT ---
    bn = hadamard(num, denom_b)  # (3, 5) * (3, 5) # num
    gamma_ = np.broadcast_to(gamma, X.shape)  # (5,) -> (3, 5)
    bn_gamma = hadamard(bn, gamma_)  # (3 * 5) * (1 * 5)
    bias_ = np.broadcast_to(bias, bn_gamma.shape)
    bn_gamma_bias = add(bn_gamma, bias_)

    fwd = bn_gamma_bias

    ## BACKWARD:
    # bn_gamma_bias = add(bn_gamma, bias_)
    dldbn_gamma, dlbias_ = vjp_add(bn_gamma, bias_, dldp)  # we return dlbn_gamma

    # bias_ = np.broadcast_to(bias, bn_gamma.shape)
    dldbias = vjp_broadcast(dlbias_)  # this reduces

    # bn_gamma = hadamard(bn, gamma_)  # (3 * 5) * (1 * 5)
    dldbn, dldgamma_ = vjp_hadamard(bn, gamma_, dldbn_gamma)

    # print("dldbn_gamma ", dldbn_gamma)
    # Collapse gamma_ (3, 5) -> gamma (5,)
    # gamma_ = np.broadcast_to(gamma, X.shape)  # (5,) -> (3, 5)
    dldgamma = vjp_broadcast(dldgamma_)
    # print("dldgamma reduced", dldgamma)

    ## DO NOT CARE BELOW THIS
    # bn = hadamard(num, denom_b)  # (3, 5) * (3, 5)
    dldnum, dldenom_b = vjp_hadamard(num, denom_b, dldbn)

    ## Variance
    # denom_b = np.broadcast_to(denom_vec, X.shape)  # Broadcast:  (5,) -> (3, 5)
    dldnom_vec = vjp_broadcast(dldenom_b)

    # denom_vec = power(var_eps, -0.5)  # Pure Power: (5,)
    dldvar_eps = vjp_power(var_eps, -0.5, dldnom_vec)

    # var_eps = add(var, eps_b)  # Elementwise: (5,) + scalar
    dldvar, dldeps_b = vjp_add(var, eps_b, dldvar_eps)

    # eps_b = np.broadcast_to(eps, var.shape)  # Broadcast: () -> (5,)
    # var = (1.0 / len(X)) * sum0_  # (5,)
    dldsum0_ = vjp_scale(1.0 / len(X), dldvar)

    # sum0_ = np.sum(b, axis=0)  # collapse: vector of size 5 i.e. feats
    dldb = vjp_reduction(dldsum0_, b.shape)

    ## Numerator
    # b = pow(num, 2)  # size (3, 5) # applied elementwise
    dldnum += vjp_power(num, 2.0, dldb)

    # num = sub(X, meanb)  # size (3,5) # broadcasting (possible extra op)
    dldX, dldmeanb = vjp_sub(
        X, meanb, dldnum
    )  # size (3,5) # broadcasting (possible extra op)

    # meanb = np.broadcast_to(mean, X.shape)
    dldmean = vjp_broadcast(dldmeanb)

    # mean = (1.0 / len(X)) * sum0  # compute mean
    dldsum0 = vjp_scale(1.0 / len(X), dldmean)

    # sum0 = np.sum(X, axis=0)  # reduction # collapse: vector of size 5 i.
    dldX += vjp_reduction(dldsum0, X.shape)

    return fwd, dldX, dldgamma, dldbias


def test_bn_fwd(data):
    bn = BatchNorm1d(num_features=data.shape[1])
    data_cpy = data.detach().cpu().numpy()
    assert np.allclose(bn(data).detach().cpu().numpy(), bn_manual(X=data_cpy))


def test_backward_bn(data):
    data_cpy = data.detach().clone().numpy()
    data.requires_grad = True
    model = BatchNorm1d(num_features=data.shape[1], dtype=torch.float32)
    model.zero_grad()
    fwd_pt = model(data)

    dldp_np = np.random.randn(*data.shape).astype(np.float32)
    fwd_pt.backward(tensor(dldp_np))
    params = model.parameters()
    params_l = list(params)
    # print(fwd_pt)

    grads = dict()

    for name, param in model.named_parameters():
        if param.requires_grad:
            grads[name] = param.grad

    fwd, dldX, dldgamma, dldbias = fwd_backwd_bn(data_cpy, dldp_np)

    grads["X"] = data.grad

    # now do assertions
    assert np.allclose(grads["weight"].detach().cpu().numpy(), dldgamma)
    assert np.allclose(grads["bias"].detach().cpu().numpy(), dldbias)
    assert np.allclose(grads["X"].detach().cpu().numpy(), dldX)
    assert np.allclose(fwd_pt.detach().cpu().numpy(), fwd)


def main():
    features = 5
    examples = 3

    X = init_data(examples, features)
    # test_bn_fwd(X)
    # fwd, dldX, dldgamma, dldbias = fwd_backwd(X_new, dldp_np)
    test_backward_bn(X)
    # print(dldgamma, '\n', dldbias)
    # np.pow(a,-1)


if __name__ == "__main__":
    main()
