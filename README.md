# manual-backprop

Reverse-mode autodiff by hand in NumPy: per-op VJPs, broadcasting-aware, verified against PyTorch.

Inspired by https://robotchinwag.com/posts/backpropagation-and-multivariable-calculus/ and https://www.youtube.com/watch?v=DGF1FtfbO6o this is an exploratory repo to understand vjps in more detail. 

We derive the following vjps using tensor calculus. Derivations are shown in the notebooks. 

- elementwise ops: hadamard, power
- matmuls
- ops of ops: batch norm


