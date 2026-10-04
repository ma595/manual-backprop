import numpy as np
from torch.nn.functional import sigmoid


def sub(a, b):
    return a - b


def add(a, b):
    return a + b


def power(a, pow):
    # return np.array([j ** pow for i in a for j in i])
    return a**pow


def linear(X, W):
    # W is expressed as (out, in)
    return X * W.T


def mm(a, b):
    return a @ b


def hadamard(a, b):
    return a * b


def vjp_add(a, b, dldp):
    return unbroadcast(dldp, a.shape), unbroadcast(dldp, b.shape)


def vjp_sub(a, b, dldp):
     dlda, dldb = vjp_add(a, b, dldp)
     return dlda, -dldb


# the fused transpose + mm implementation
# more efficient than vjp_mm for linear layer
def vjp_linear(a, b, dldp):
    # i.e.
    # dldX = dldp @ W
    # dldW = dldp.T @ X
    # return dldX, dldW
    dlda = dldp @ b
    dldb = dldp.T @ a
    return dlda, dldb


def vjp_mm(a, b, dldp):
    dlda = dldp @ b.T
    dldb = a.T @ dldp
    return dlda, dldb


def vjp_mse(yhat, Y, dldp):
    return 1.0 / (Y.shape[0] * Y.shape[1]) * dldp * (yhat - Y)


def unbroadcast(grad, shape):
    # forward SUM  p_j = Σ_i a_ij -> dlda = broadcast(dldp, a.shape)
    # forward broadcast: p_ij = a_j -> dlda_j = Σ_i dldp_ij   (sum the stretched axis)

    # Broadcasting rules in numpy:
    # A=(4,2) + B=(2,)
    # Add leading dimension
    # (4,2) + (1,2) (add B to every row of A)
    # i.e. go from r.h.s to l.h.s
    # can only broadcast if the dim is equal or one of dims is 1.
    # But this doesn't work for:
    # (4,2) + (4,)
    # Add leading dimension
    # (4,2) + (1,4) doesn't work going from r.h.s to l.h.s
    # To make it work:
    # (4,2) + B[:, np.axis]
    # (4,2) + (4,1)

    # Loop 1 handles reversing the
    # (a) prepended leading axes by repeatedly dropping extra
    # dimensions off the left until the ranks match.
    # Loop 2 handles reversing the (b) stretched size-1 axes
    # by summing across any dimension that was originally size
    # 1 while preserving its rank using keepdims=True.

    # A good example to think of is
    # B = shape(100,2,2) and shape is (1,2)
    # first we encounter condition 1 (once):
    # (2, 2) + (1,2)
    # and condition 2:
    # (2, 1) -> (1, 2) - keeping dims.

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)  # leading axis
    for ax, n in enumerate(shape):
        if n == 1:
            grad = grad.sum(axis=ax, keepdims=True)
    return grad


# p = scale * a_{ij}
# dlda = dldp *
def vjp_scale(scale, dldp):
    # dlda = dldp * dpda
    dlda = dldp * scale
    return dlda


def vjp_hadamard(a, b, dldp):
    # print(dldp.shape, b.shape, a.shape)
    dlda = dldp * b
    dldb = dldp * a
    return dlda, dldb


def vjp_power(a, alpha, dldp):
    dlda = (
        alpha * dldp * pow(a, alpha - 1.0)
    )  # local grad or jacob is alpha * pow(a, alpha-1)
    return dlda


def vjp_sigmoid(a, dldp):
    # print(dldp.shape, a.shape)
    dlda = dldp * sigmoid(a) * (1 - sigmoid(a))
    return dlda


# UNUSED - we rely on python broadcasting rules
# def vjp_reduction(A, shape):
#     return np.broadcast_to(A, shape)
