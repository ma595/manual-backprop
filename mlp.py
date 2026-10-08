import numpy as np
import torch
from src.ops import *
from torch.nn.functional import sigmoid

torch.manual_seed(0)


def initialise_params(X):
    in_feats = X.shape[1]
    out_feats = 32
    W = 0.1 * torch.rand((out_feats, in_feats), dtype=torch.float64)
    W.requires_grad = True
    b = 0.1 * torch.rand((out_feats), dtype=torch.float64)
    b.requires_grad = True

    inh_feats = out_feats
    outh_feats = 2

    B = 0.1 * torch.rand((outh_feats, inh_feats), dtype=torch.float64)
    B.requires_grad = True
    c = 0.1 * torch.rand((outh_feats), dtype=torch.float64)
    c.requires_grad = True

    return ((W, b), (B, c))


def ffn(params, X, Y):
    ((W, b), (B, c)) = params

    s = X @ W.T
    u = s + b

    # print(u.shape)
    z = sigmoid(u)

    t = z @ B.T
    y = t + c
    lo = 1 / (2 * y.shape[0] * y.shape[1]) * torch.sum((y - Y) * (y - Y))
    assert lo.shape == torch.Size([])
    return lo


def fwdbck(W, X, Y):
    ((W, b), (B, c)) = W


    # forward
    s = X @ W.T
    u = s + b
    # z = sigmoid(u)
    from src.ops import sigmoid as snpy
    z = snpy(u)
    t = z @ B.T
    y = t + c

    # compute loss
    lo = 1.0 / (2 * y.shape[0] * y.shape[1]) * np.sum((y - Y) * (y - Y))
    # print(lo.shape)
    # assert lo.shape == torch.Size([])

    # backward
    dlossdloss = 1
    # this propagates the error (not the loss)
    dldyhat = vjp_mse(y, Y, dldp=dlossdloss)  ## 2nd order tensor
    dldt, dldc = vjp_add(a=t, b=c, dldp=dldyhat)  ## 2nd order, 1st order

    dldz, dldB = vjp_linear(a=z, b=B, dldp=dldt)
    # print(dldz.shape, z.shape)
    dldu = vjp_sigmoid(a=u, dldp=dldz)

    dlds, dldb = vjp_add(a=s, b=b, dldp=dldu)

    dldX, dldW = vjp_linear(a=X, b=W, dldp=dlds)

    # all gradients for all params computed, ready to return
    return ((dldW, dldb), (dldB, dldc)), lo




def test_single_step(X_train, Y_train):
    params = initialise_params(X_train)
    loss = ffn(params, X_train, Y_train)
    loss.backward()
    ((W, b), (B, c)) = params
    # W_new = fwdbck(W, X, Y)
    # Verify against PyTorch's native autograd:

    W_np = W.detach().numpy()
    b_np = b.detach().numpy()
    B_np = B.detach().numpy()
    c_np = c.detach().numpy()

    params_np = ((W_np, b_np), (B_np, c_np))

    grads, loss = fwdbck(params_np, X_train.detach().numpy(), Y_train.detach().numpy())
    (dldW, dldb), (dldB, dldc) = grads

    # print(W)
    # assert torch.allclose
    # print(type(W.grad), type(dldW))
    # print(W.shape, dldW.shape)

    np.testing.assert_allclose(dldW, W.grad.numpy())
    np.testing.assert_allclose(dldb, b.grad.numpy())
    np.testing.assert_allclose(dldB, B.grad.numpy())
    np.testing.assert_allclose(dldc, c.grad.numpy())


if __name__ == "__main__":
    from src.ds import create_XY_data
    tempX, tempY = create_XY_data(10)
    test_single_step(tempX, tempY)
