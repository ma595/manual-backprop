from src.ops import *
from src.batchnorm import fwd_bn, fwd_backwd_bn
import torch
import numpy as np

import pytest

torch.manual_seed(0)
np.random.seed(0)

MM_SHAPES = [
    ((3, 4), (4, 5)),
    ((2, 3, 4), (2, 4, 5)),  # batched, same batch dims
    ((2, 3, 4), (4, 5)),  # b broadcast over batch
    ((1, 3, 4), (2, 4, 5)),  # a broadcast over batch
]


@pytest.mark.parametrize("shape_a, shape_b", MM_SHAPES, ids=str)
def test_vjp_mm(shape_a, shape_b):
    # create some test data

    a = np.random.rand(*shape_a)
    b = np.random.rand(*shape_b)

    p = a @ b
    dldp = np.random.rand(*p.shape)

    print(f"dldp.shape={dldp.shape}")

    dlda, dldb = vjp_mm(a, b, dldp)

    a_torch = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor(b, dtype=torch.float64, requires_grad=True)

    fwd = a_torch @ b_torch

    fwd.backward(torch.tensor(dldp, dtype=torch.float64))

    # print(list(fwd.parameters()))
    print(a_torch.grad.shape, dlda.shape)
    print(b_torch.grad.shape, dldb.shape)
    print(dldb)
    assert a_torch.grad.shape == dlda.shape
    assert b_torch.grad.shape == dldb.shape

    np.testing.assert_allclose(a_torch.grad.detach().cpu().numpy(), dlda)
    np.testing.assert_allclose(b_torch.grad.detach().cpu().numpy(), dldb)


LINEAR_SHAPES = [
    ((3, 4), (5, 4)),
    ((2, 3, 4), (5, 4)),  # b broadcast over batch
]


@pytest.mark.parametrize("shape_a, shape_b", LINEAR_SHAPES, ids=str)
def test_vjp_linear(shape_a, shape_b):
    # create some test data
    # b is the weight tensor here

    a = np.random.rand(*shape_a)
    b = np.random.rand(*shape_b)

    p = linear(a, b)
    dldp = np.random.rand(*p.shape)

    dlda, dldb = vjp_linear(a, b, dldp)

    # now for the pytorch bit (these parts aren't actually used)
    a_torch = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor(b, dtype=torch.float64, requires_grad=True)

    fwd = linear(a_torch, b_torch)
    fwd.backward(torch.tensor(dldp, dtype=torch.float64))

    assert a_torch.grad.shape == dlda.shape
    assert b_torch.grad.shape == dldb.shape

    np.testing.assert_allclose(a_torch.grad.detach().cpu().numpy(), dlda)
    np.testing.assert_allclose(b_torch.grad.detach().cpu().numpy(), dldb)

    # try from Linear instead
    from torch.nn import Linear

    print("shape_b ", shape_b)

    fwd = Linear(
        in_features=shape_b[1],
        out_features=shape_b[0],
        bias=False,
        dtype=torch.float64,
    )

    with torch.no_grad():
        fwd.weight.copy_(b_torch)  # use your W, shape (5, 4)

    # create a new tensor otherwise we take derivative twice
    a_torch = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    out = fwd(a_torch)

    out.backward(torch.tensor(dldp, dtype=torch.float64))

    np.testing.assert_allclose(a_torch.grad.detach().cpu().numpy(), dlda)
    np.testing.assert_allclose(fwd.weight.grad.detach().cpu().numpy(), dldb)


BROADCAST_SHAPES = [
    ((3, 5), (3, 5)),
    ((3, 5), (5,)),
    ((3, 1), (1, 5)),
    ((), (5,)),  # scalar * vector
]

# BINARY_OPS = [(add, vjp_add), (sub, vjp_sub), (hadamard, vjp_hadamard)]
BINARY_OPS = [(hadamard, vjp_hadamard), (add, vjp_add), (sub, vjp_sub)]


@pytest.mark.parametrize("op, vjp_op", BINARY_OPS, ids=["hadamard", "add", "sub"])
@pytest.mark.parametrize("shape_a, shape_b", BROADCAST_SHAPES, ids=str)
def test_vjp_hadamard(shape_a, shape_b, op, vjp_op):
    a = np.random.rand(*shape_a)
    b = np.random.rand(*shape_b)

    p = op(a, b)
    dldp = np.random.rand(*p.shape)

    print(f"dldp.shape={dldp.shape}")

    dlda, dldb = vjp_op(a, b, dldp)

    a_torch = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor(b, dtype=torch.float64, requires_grad=True)

    fwd = op(a_torch, b_torch)

    fwd.backward(torch.tensor(dldp, dtype=torch.float64))

    # print(list(fwd.parameters()))
    print("a ", a_torch.grad.shape, dlda.shape)
    print("b ", b_torch.grad.shape, dldb.shape)
    print(dldb)
    assert a_torch.grad.shape == dlda.shape
    assert b_torch.grad.shape == dldb.shape

    np.testing.assert_allclose(a_torch.grad.detach().cpu().numpy(), dlda)
    np.testing.assert_allclose(b_torch.grad.detach().cpu().numpy(), dldb)


def test_fwd_bckwd_bn():

    num_examples = 3
    num_feats = 5
    shape = (num_examples, num_feats)

    # create torch data
    data = torch.rand(*shape, dtype=torch.float64)

    gamma = np.random.rand(
        data.shape[1],
    )
    bias = np.random.rand(
        data.shape[1],
    )

    data_cpy = data.detach().clone().numpy()
    dldp_np = np.random.randn(*data.shape).astype(np.float64)
    fwd, dldX, dldgamma, dldbias = fwd_backwd_bn(data_cpy, dldp_np, gamma, bias)

    data.requires_grad = True
    model = torch.nn.BatchNorm1d(num_features=data.shape[1], dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(gamma))
        model.bias.copy_(torch.from_numpy(bias))
    model.zero_grad()
    fwd_pt = model(data)
    fwd_pt.backward(torch.tensor(dldp_np))
    # params = model.parameters()
    # params_l = list(params)

    grads = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            grads[name] = param.grad

    grads["X"] = data.grad

    # now do assertions
    np.testing.assert_allclose(grads["weight"].detach().cpu().numpy(), dldgamma)
    np.testing.assert_allclose(grads["bias"].detach().cpu().numpy(), dldbias)
    np.testing.assert_allclose(grads["X"].detach().cpu().numpy(), dldX)
    np.testing.assert_allclose(fwd_pt.detach().cpu().numpy(), fwd)


if __name__ == "__main__":
    # shapes = ((3, 4), (4, 5))
    # shapes = ((2, 3, 4), (2, 4, 5))
    # shapes = ((2, 3, 4), (4, 5))     # b broadcast over batch
    # # test_vjp_mm(*shapes)

    # # test_forward
    # a = np.random.rand(*shapes[0])
    # b = np.random.rand(*shapes[1])
    # a_torch = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    # b_torch = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    # p = a @ b

    # print(a.shape, a.swapaxes(-2, -1).shape)
    # print(b.shape, b.swapaxes(-2, -1).shape)

    # test_vjp_mm(*shapes)

    # test_vjp_hadamard((), (1, 5))
    test_vjp_linear((3, 4), (5, 4))
