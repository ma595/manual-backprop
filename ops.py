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
    return X * W.T


def mm(a, b):
    return a @ b


def hadamard(a, b):
    return a * b


def vjp_add(a, b, dldp):
    dlda = dldp
    dldb = dldp
    return dlda, dldb


def vjp_sub(a, b, dldp):
    dlda = dldp
    dldb = dldp
    return dlda, -dldb


def vjp_linear(X, W, dldp):
    # i.e. 
    dldX = dldp @ W
    dldW = dldp.T @ X
    return dldX, dldW


def vjp_mm(a, b, dldp):
    dlda = dldp @ b.T
    dldb = a.T @ dldp
    return dlda, dldb


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
        alpha * dldp * alpha * pow(a, alpha - 1.0)
    )  # local grad or jacob is alpha * pow(a, alpha-1)
    return dlda


def vjp_reduction(A, shape):
    return np.broadcast_to(A, shape)


def vjp_broadcast(A):
    return np.sum(A, axis=0)
