from typing import Union, List

import torch
from torch import nn, Size

from torch.nn import Module


class MyLayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], *,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()

        # Convert `normalized_shape` to `torch.Size`
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        self.num_batches_tracked = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.running_batch_var = nn.Parameter(torch.zeros(normalized_shape[0]), requires_grad=False)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # Create parameters for $\gamma$ and $\beta$ for gain and bias
        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor):
        exponential_average_factor = 0.0

        if self.training:
            self.num_batches_tracked.data += 1
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)

        # Sanity check to make sure the shapes match
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]

        # The dimensions to calculate the mean and variance on
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]

        # Calculate the mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X]$
        mean = x.mean(dim=dims, keepdim=True)
        # Calculate the squared mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X^2]$
        mean_x2 = (x ** 2).mean(dim=dims, keepdim=True)
        # Variance of all element $Var[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$
        var = mean_x2 - mean ** 2

        with torch.no_grad():
            if self.training:
                batch_var = x.var([0, 2, 3], unbiased=False)
                n = x.numel() / x.size(1)
                self.running_batch_var.data = exponential_average_factor * batch_var * n / (n - 1) + (1 - exponential_average_factor) * self.running_batch_var
            else:
                var = self.running_batch_var[None, :, None, None]

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift $$\text{LN}(x) = \gamma \hat{X} + \beta$$
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        return x_norm
