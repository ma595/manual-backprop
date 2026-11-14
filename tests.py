from src.ops import *
import torch
import numpy as np

import pytest

torch.manual_seed(0)
np.random.seed(0)

MM_SHAPES = [
    ((3, 4), (4, 5)),
    ((2, 3, 4), (2, 4, 5)),  # batched, same batch dims
    ((2, 3, 4), (4, 5)),     # b broadcast over batch
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


# @pytest.mark.parametrize("shape_a, shape_b", MM_SHAPES, ids=str)
# def test_vjp_linear(shape_a, shape_b):
#     # create some test data

#     a = np.random.rand(*shape_a)
#     b = np.random.rand(*shape_b)

#     p = a @ b
#     dldp = np.random.rand(*p.shape)

#     dlda, dldb = vjp_linear(a, b.T, dldp)

#     a_torch = torch.tensor(a, dtype=torch.float64, requires_grad=True)
#     b_torch = torch.tensor(b.T, dtype=torch.float64, requires_grad=True)

#     fwd = linear(a, b)

#     fwd.backward(torch.tensor(dldp, dtype=torch.float64))

#     # print(list(fwd.parameters()))

#     assert a_torch.grad.shape == dlda.shape
#     assert b_torch.grad.shape == dldb.shape

#     np.testing.assert_allclose(a_torch.grad.detach().cpu().numpy(), dlda)
#     np.testing.assert_allclose(b_torch.grad.detach().cpu().numpy(), dldb)


BROADCAST_SHAPES = [
    ((3, 5), (3, 5)),
    ((3, 5), (5,)),
    ((3, 1), (1, 5)),
    ((), (5,)),  # scalar * vector
]


@pytest.mark.parametrize("shape_a, shape_b", BROADCAST_SHAPES, ids=str)
def test_vjp_hadamard(shape_a, shape_b):
    a = np.random.rand(*shape_a)
    b = np.random.rand(*shape_b)

    p = a * b
    dldp = np.random.rand(*p.shape)

    print(f"dldp.shape={dldp.shape}")

    dlda, dldb = vjp_hadamard(a, b, dldp)

    a_torch = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    b_torch = torch.tensor(b, dtype=torch.float64, requires_grad=True)

    fwd = a_torch * b_torch

    fwd.backward(torch.tensor(dldp, dtype=torch.float64))

    # print(list(fwd.parameters()))
    print("a ", a_torch.grad.shape, dlda.shape)
    print("b ", b_torch.grad.shape, dldb.shape)
    print(dldb)
    assert a_torch.grad.shape == dlda.shape
    assert b_torch.grad.shape == dldb.shape

    np.testing.assert_allclose(a_torch.grad.detach().cpu().numpy(), dlda)
    np.testing.assert_allclose(b_torch.grad.detach().cpu().numpy(), dldb)

    # dldp = np.random.rand()

    # hadamard(


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

    test_vjp_hadamard((), (1,5))
