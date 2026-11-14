import numpy as np


def sub(a, b):
    return a - b


def add(a, b):
    return a + b


def power(a, pow):
    # return np.array([j ** pow for i in a for j in i])
    return a**pow


def linear(X, W):
    # W is expressed as (out, in)
    return X @ W.T


def mm(a, b):
    return a @ b


def hadamard(a, b):
    return a * b

# p = sigmoid(a)
def sigmoid(a):
    return 1.0 / (1 + np.exp(-a))

# p = a + b
def vjp_add(a, b, dldp):
    return unbroadcast(dldp, a.shape), unbroadcast(dldp, b.shape)

# p = a - b
def vjp_sub(a, b, dldp):
    dlda, dldb = vjp_add(a, b, dldp)
    return dlda, -dldb


# the fused transpose + mm implementation
# more efficient than vjp_mm for linear layer

# p = a @ b.Y
def vjp_linear(a, b, dldp):
    # linear is p = a @ W.T
    # but we pass in:
    # a = X, b = W
    #

    # dldX = dldp @ W
    # dldW = dldp.T @ X
    # return dldX, dldW
    dlda = dldp @ b
    dldb = dldp.T @ a
    return dlda, dldb

# p = a @ b 
def vjp_mm(a, b, dldp):
    # dl/db_{ij} = a.T_{ik} dldp_{kj} (sum over batch dim of a)
    # if g = dldp
    # for a single update dldW_11 = X_11 g_11 + X_21 g_21 
    # i.e. each parameter is used once per batch (for a batch of two we update the gradient by 2 adds)
    # for a = (2,3,4) and b = (4,5)
    # )
    dlda = unbroadcast(dldp @ b.swapaxes(-2, -1), a.shape)
    dldb = unbroadcast(a.swapaxes(-2, -1) @ dldp, b.shape)
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


# p = a * b 
def vjp_hadamard(a, b, dldp):
    dlda = unbroadcast(dldp * b, np.shape(a)) # np.shape now works for scalars
    dldb = unbroadcast(dldp * a, np.shape(b))
    return dlda, dldb

# p = pow(a, alpha)
def vjp_power(a, alpha, dldp):
    dlda = (
        alpha * dldp * pow(a, alpha - 1.0)
    )  # local grad or jacob is alpha * pow(a, alpha-1)
    return dlda

# p = sigmoid(a)
def vjp_sigmoid(a, dldp):
    dlda = dldp * sigmoid(a) * (1 - sigmoid(a))
    return dlda


# UNUSED - we rely on python broadcasting rules
# def vjp_reduction(A, shape):
#     return np.broadcast_to(A, shape)
