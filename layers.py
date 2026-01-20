# layers.py
"""
Custom PyTorch layer: LearnedAffine

y = x * gamma + beta

- gamma and beta are learned parameters
- Broadcast across the batch dimension
"""

import torch
import torch.nn as nn


class LearnedAffine(nn.Module):
    """
    Simple custom layer: y = x * gamma + beta

    Expected input shape: (batch_size, dim)
    Parameters:
        gamma: shape (dim,)
        beta: shape (dim,)
    """
    def __init__(self, dim: int):
        super().__init__()
        # Initialize gamma to ones and beta to zeros
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply element-wise affine transform.

        x: Tensor of shape (batch_size, dim)
        returns: Tensor of shape (batch_size, dim)
        """
        # Broadcasting: gamma and beta are (dim,)
        # x is (batch_size, dim)
        return x * self.gamma + self.beta


def _sanity_check():
    """
    Tiny sanity check for LearnedAffine.

    Checks:
    - Input and output shapes match
    - Parameter count matches expected value (2 * dim)
    """
    dim = 16
    batch_size = 4

    layer = LearnedAffine(dim)
    x = torch.randn(batch_size, dim)
    y = layer(x)

    print("=== LearnedAffine Sanity Check ===")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected param count: {2 * dim}")
    actual_params = sum(p.numel() for p in layer.parameters())
    print(f"Actual param count:   {actual_params}")

    assert x.shape == y.shape, "Input and output shapes should match"
    assert actual_params == 2 * dim, "Parameter count should be 2 * dim"
    print("Sanity check passed.")


if __name__ == "__main__":
    _sanity_check()
