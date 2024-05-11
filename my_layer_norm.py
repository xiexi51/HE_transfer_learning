from typing import Union, List

import torch
from torch import nn, Size

from torch.nn import Module

def Cheb(input, a, b, cn=3):
    device = input.device

    if cn == 0:
        return 1 / torch.sqrt(input)
    elif cn == 2:
        c = torch.tensor([ 1.96851567, -1.07614785], device=device)
    elif cn == 3:
        c = torch.tensor([ 1.08122509, -0.71963818,  0.27800576], device=device)
    elif cn == 4:
        c = torch.tensor([ 2.23027669, -1.66487833,  0.82379455, -0.35346759], device=device)
    elif cn == 5:
        c = torch.tensor([ 2.2708268 , -1.7526131 ,  0.93297679, -0.50419129,  0.22388761], device=device)
    elif cn == 6:
        c = torch.tensor([ 2.29051981, -1.79490704,  0.98453865, -0.57316873,  0.32200415, -0.14559046], device=device)
    else:
        raise ValueError("Chebyshev polynomial of order {} not implemented".format(cn))
    
    with torch.no_grad():
        x = (2*input - a - b)/(b - a)

        x2 = 2*x

        if len(c) == 1:
            c0 = c[0]
            c1 = 0
        elif len(c) == 2:
            c0 = c[0]
            c1 = c[1]
        else:
            x2 = 2*x
            c0 = c[-2]
            c1 = c[-1]
            for i in range(3, len(c) + 1):
                tmp = c0
                c0 = c[-i] - c1
                c1 = tmp + c1*x2

    return c0 + c1*x

class MyLayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], *,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()

        self.number = 0
        self.cn = 3

        # Convert `normalized_shape` to `torch.Size`
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        self.num_batches_tracked = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.running_var_mean = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # Create parameters for $\gamma$ and $\beta$ for gain and bias
        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

        self.train_var_list = []
        self.test_var_list = []


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
        
        if self.training:
            if len(self.train_var_list) < 1000:
                self.train_var_list.append(var + self.eps)
        else:
            if len(self.test_var_list) < 1000:
                self.test_var_list.append(var + self.eps)

        if self.training:
            # with torch.no_grad():
            #     self.running_var_mean.data = exponential_average_factor * torch.mean(var) + (1 - exponential_average_factor) * self.running_var_mean
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        else:
            var_mean = var.mean()
            x_norm = (x - mean) * Cheb((var / var_mean + self.eps), 0.1, 3, cn=self.cn) / torch.sqrt(var_mean)

            # x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift $$\text{LN}(x) = \gamma \hat{X} + \beta$$
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        return x_norm
    


