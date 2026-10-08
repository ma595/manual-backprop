import numpy as np
import torch

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
    # print(rotations.shape, x1x2.unsqueeze(2).shape)

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
