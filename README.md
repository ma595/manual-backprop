# manual-backprop

Reverse-mode autodiff by hand in NumPy: per-op VJPs (vector-Jacobian products), broadcasting
aware, with gradients checked against PyTorch.

This is an exploratory repo. I wanted to understand VJPs properly, so each op's backward pass is
derived on paper in index notation and then written out by hand. There is no autograd engine here
(yet): the backward pass for each model is a hand-ordered chain of VJP calls.

Inspired by [Backpropagation and multivariable
calculus](https://robotchinwag.com/posts/backpropagation-and-multivariable-calculus/) and [this
video](https://www.youtube.com/watch?v=DGF1FtfbO6o) by Andrew Fitzgibbon at RSECon. 

## What is here

`src/ops.py` holds the forward ops and their VJPs:

- elementwise: `add`, `sub`, `hadamard`, `power`, `sigmoid`, scaling by a constant
- matmuls: `mm` (`a @ b`) and `linear` (`X @ W.T`, with `W` stored as `(out, in)`)
- loss: mean squared error
- `unbroadcast`, which sums a gradient back down to an input's shape. NumPy broadcasting in the
  forward pass copies values, so the backward pass has to add the copies' gradients back together.
  This one helper handles that for every op, instead of writing explicit broadcast and reduction
  ops.

Two models are built from these ops:

- `batchnorm.py`: batch norm forward and backward, written as a chain of primitive ops.
- `mlp.py`: a two-layer MLP (linear, sigmoid, linear, MSE) on a 2D rotation dataset. The data
  comes from `src/dataset.py`.

## What is checked

Both models are compared against PyTorch in float64 with `np.testing.assert_allclose` at its
default tolerance (`rtol=1e-7`):

- batch norm: the forward output and the gradients for `X`, `gamma` and `beta`, against
  `torch.nn.BatchNorm1d`
- MLP: the gradients for all four parameters, against `loss.backward()`

Limitations:

- The batch norm check uses PyTorch's default `gamma = 1` and `beta = 0`, so it cannot catch a
  mistake in how `gamma` and `beta` are applied in the forward pass.
- The individual ops are only tested through these two models. There are no per-op tests yet.

## Running it

```sh
pip install -r requirements.txt
python batchnorm.py
python mlp.py
```

Each script exits silently if the gradients match and raises an `AssertionError` if they do not.

## Notebooks

The derivations are in `notebooks/`:

- `torch_linear.ipynb`: the MLP's VJPs for a single example in index notation. The Jacobian is
  built two ways (from VJPs with one-hot vectors, and from finite differences) to check them.
- `torch_linear_batch_derivation.ipynb`: the same derivation for batched inputs, then the rotation
  test problem and training.
- `torch_batch_norm_derivation.ipynb`: batch norm broken into primitive ops, and the VJPs of
  broadcast and reduction.

## Next steps

- per-op tests against finite differences, run with pytest
- random `gamma` and `beta` in the batch norm test
- a tape, so the backward pass is generated from the forward pass instead of written out by hand
