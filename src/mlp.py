import numpy as np
import torch
from src.ops import *

torch.manual_seed(0)


def initialise_params(X, Y, hidden=32):
    in_feats, out_feats = X.shape[1], Y.shape[1]
    W = 0.1 * torch.rand((hidden, in_feats), dtype=torch.float64)
    W.requires_grad = True
    b = 0.1 * torch.rand((hidden,), dtype=torch.float64)
    b.requires_grad = True

    B = 0.1 * torch.rand((out_feats, hidden), dtype=torch.float64)
    B.requires_grad = True
    c = 0.1 * torch.rand((out_feats,), dtype=torch.float64)
    c.requires_grad = True

    return ((W, b), (B, c))


def ffn(params, X, Y):
    ((W, b), (B, c)) = params

    s = X @ W.T
    u = s + b

    # print(u.shape)
    z = torch.sigmoid(u)

    t = z @ B.T
    y = t + c
    lo = 1 / (2 * y.shape[0] * y.shape[1]) * torch.sum((y - Y) * (y - Y))
    assert lo.shape == torch.Size([])
    return lo


def fwd_backwd(W, X, Y):
    ((W, b), (B, c)) = W

    # forward
    s = X @ W.T
    u = s + b
    # z = sigmoid(u)
    z = sigmoid(u)
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

    dldu = vjp_sigmoid(a=u, dldp=dldz)

    dlds, dldb = vjp_add(a=s, b=b, dldp=dldu)

    dldX, dldW = vjp_linear(a=X, b=W, dldp=dlds)

    # all gradients for all params computed, ready to return
    return ((dldW, dldb), (dldB, dldc)), lo


if __name__ == "__main__":
    from src.dataset import create_XY_data

    tempX, tempY = create_XY_data(10)
    print(tempX.shape)
    print(tempY.shape)

    # test_single_step(tempX, tempY)
