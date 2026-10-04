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
    z = sigmoid(u)
    t = z @ B.T
    y = t + c

    # compute loss
    lo = 1.0 / (2 * y.shape[0] * y.shape[1]) * torch.sum((y - Y) * (y - Y))
    assert lo.shape == torch.Size([])

    # backward
    dlossdloss = 1
    # this propagates the error (not the loss)
    dldyhat = vjp_mse(y, Y, dldp=dlossdloss)  ## 2nd order tensor
    dldt, dldc = vjp_add(a=t, b=c, dldp=dldyhat)  ## 2nd order, 1st order

    dldc = unbroadcast(dldc, c)

    dldz, dldB = vjp_linear(a=z, b=B, dldp=dldt)
    # print(dldz.shape, z.shape)
    dldu = vjp_sigmoid(a=u, dldp=dldz)

    dlds, dldb = vjp_add(a=s, b=b, dldp=dldu)

    dldb = unbroadcast(dldb, b.shape)

    dldX, dldW = vjp_linear(a=X, b=W, dldp=dlds)

    # all gradients for all params computed, ready to return
    return ((dldW, dldb), (dldB, dldc)), lo


def create_XY_data(n_examples):
    torch.manual_seed(0)
    # n_feats = 3
    # n_outputs = 2
    # train_set = torch.rand((n_examples, n_feats), dtype=torch.float64)
    # X = torch.rand(size=(n_examples, n_feats), dtype=torch.float64)

    x1 = torch.rand(n_examples, dtype=torch.float64)
    x2 = torch.rand(n_examples, dtype=torch.float64)
    theta = torch.rand(n_examples, dtype=torch.float64) * 2 * torch.pi

    def rotation(theta):
        return torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ]
        )

    # rotation(theta)
    rotations = torch.tensor(
        list(map(lambda t: [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], theta)),
        dtype=torch.float64,
    )

    # stack it in column direction
    x1x2 = torch.stack((x1, x2), dim=1)
    print(rotations.shape, x1x2.unsqueeze(2).shape)

    # Multiplying the rotation matrix by the inputs in vectorised way:
    # Attempt 1: Using bmm
    # we need to unsqueeze here for bmm to work - this adds another dimension so
    # the batch matrix multiplication works
    Y = torch.bmm(rotations, x1x2.unsqueeze(2))
    # Attempt 2: Using einsum
    ## or since rotations is (10, 2, 2) and x1x2.unsqueeze is (10, 2)
    Y = torch.einsum("bik, bk -> bi", rotations, x1x2.squeeze())
    # Attempt 3: Manual slow way
    Y = torch.stack(
        [rotation(t) @ torch.stack([x, y], dim=0) for t, x, y in zip(theta, x1, x2)],
        dim=0,
    )
    return torch.stack((x1, x2, theta), dim=1), Y


def test_single_step(X_train, Y_train):
    params = initialise_params(X_train)
    loss = ffn(params, X_train, Y_train)
    loss.backward()
    ((W, b), (B, c)) = params
    # W_new = fwdbck(W, X, Y)
    # Verify against PyTorch's native autograd:
    grads, loss = fwdbck(params, X_train, Y_train)
    (dldW, dldb), (dldB, dldc) = grads

    # print(W)
    # assert torch.allclose
    # print(type(W.grad), type(dldW))
    # print(W.shape, dldW.shape)

    assert torch.allclose(dldW, W.grad)

    print(dldb.shape)
    print((b.grad).shape)
    assert torch.allclose(dldb, b.grad)
    assert torch.allclose(dldB, B.grad)
    assert torch.allclose(dldc, c.grad)


if __name__ == "__main__":
    tempX, tempY = create_XY_data(10)
    test_single_step(tempX, tempY)
