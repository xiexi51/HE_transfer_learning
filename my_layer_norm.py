from typing import Union, List
import torch
from torch import nn, Size
from torch.nn import Module
from numpy.polynomial import Chebyshev
import numpy as np
from utils import CustomSettings

class MyCheb():
    def __init__(self):
        self.store_degree = None
        self.store_a = None
        self.store_b = None
        self.coeffs = None

    def get_coeffs(self, degree, a, b, device):
        f = lambda x: x**-0.5
        f_mapped = lambda x: f((b - a)/2 * x + (b + a)/2)
        nodes = Chebyshev.basis(degree + 1).roots()
        nodes_mapped = (b - a)/2 * nodes + (b + a)/2
        values = f(nodes_mapped)
        self.coeffs = torch.tensor(Chebyshev.fit(nodes, values, degree).convert().coef, device=device)
        self.store_degree = degree
        self.store_a = a
        self.store_b = b
    
    def calculate(self, input, degree, a, b):
        if degree == -1:
            return 1 / torch.sqrt(input)
        if not (degree == self.store_degree and a == self.store_a and b == self.store_b):
            self.get_coeffs(degree, a, b, input.device)
        
        with torch.no_grad():
            x = (2*input - a - b)/(b - a)
            x2 = 2*x

            if len(self.coeffs) == 1:
                c0 = self.coeffs[0]
                c1 = 0
            elif len(self.coeffs) == 2:
                c0 = self.coeffs[0]
                c1 = self.coeffs[1]
            else:
                x2 = 2*x
                c0 = self.coeffs[-2]
                c1 = self.coeffs[-1]
                for i in range(3, len(self.coeffs) + 1):
                    tmp = c0
                    c0 = self.coeffs[-i] - c1
                    c1 = tmp + c1*x2
        return c0 + c1*x


class MyLayerNorm(Module):
    def __init__(self, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.is_setup = False
        self.number = 0
        self.cheb_params = [4, 0.1, 5]
        self.training_use_cheb = False
        self.cheb = MyCheb()
        self.use_running_var_mean = False
        self.var_norm_boundary = 3
        self.momentum = None

        self.use_quad = True
        self.trainable_quad_finetune = True
        self.quad_coeffs = [0.03, 10, 0.2]
        self.quad_finetune_factors = [0.0001, 0.1, 0.001]
        self.quad_finetune_param = None

        self.num_batches_tracked = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.running_var_mean = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.normalized_shape = None
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        

        self.train_var_list = []
        self.test_var_list = []

        self.counts_train = None
        self.counts_test = None

        self.total_counts_train = None
        self.total_counts_test = None

        self.epoch_train_var_mean = 0
        self.epoch_train_var_sum = 0
        self.epoch_train_var_mean_count = 0

        self.epoch_test_var_mean = 0
        self.epoch_test_var_sum = 0
        self.epoch_test_var_mean_count = 0

        self.saved_var = None

    def setup(self, custom_settings: CustomSettings):
        self.is_setup = True
        self.cheb_params = custom_settings.cheb_params
        self.training_use_cheb = custom_settings.training_use_cheb
        self.var_norm_boundary = custom_settings.var_norm_boundary
        self.ln_momentum = custom_settings.ln_momentum
        self.use_quad = custom_settings.ln_use_quad
        self.trainable_quad_finetune = custom_settings.ln_trainable_quad_finetune
        self.quad_coeffs = custom_settings.ln_quad_coeffs
        self.quad_finetune_factors = custom_settings.ln_quad_finetune_factors
        if self.trainable_quad_finetune:
            self.quad_finetune_param = nn.Parameter(torch.zeros(3), requires_grad=True)
        else:
            self.quad_finetune_param = nn.Parameter(torch.zeros(3), requires_grad=False)

    def forward(self, x: torch.Tensor):
        assert self.is_setup, "MyLayerNorm needs to be explicitly setup before forward pass."

        if self.normalized_shape is None:
            self.normalized_shape = x.size()[1:]
            # Create parameters for $\gamma$ and $\beta$ for gain and bias
            if self.elementwise_affine:
                self.gain = nn.Parameter(torch.ones(self.normalized_shape))
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

        exponential_average_factor = 0.0

        if self.training:
            self.num_batches_tracked.data += 1
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            if self.momentum is not None:
                if exponential_average_factor < self.momentum:
                    exponential_average_factor = self.momentum

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

        self.saved_var = var
        
        var_mean = var.mean()

        if self.training:
            self.epoch_train_var_mean += var_mean.item()
            self.epoch_train_var_sum += var.sum().item()
            self.epoch_train_var_mean_count += 1
        else:
            self.epoch_test_var_mean += var_mean.item()
            self.epoch_test_var_sum += var.sum().item()
            self.epoch_test_var_mean_count += 1

        with torch.no_grad():
            if self.training:
                self.running_var_mean.data = exponential_average_factor * var_mean + (1 - exponential_average_factor) * self.running_var_mean

            if not self.training or self.use_running_var_mean:
                var_normed = var / self.running_var_mean
                
                bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf]
                counts, _ = np.histogram(var_normed.detach().cpu().numpy(), bins=bins)

                if self.training:
                    if self.counts_train is None:
                        self.counts_train = counts
                    else:
                        self.counts_train += counts
                else:
                    if self.counts_test is None:
                        self.counts_test = counts
                    else:
                        self.counts_test += counts

        # if self.training:
        #     if len(self.train_var_list) < 1000:
        #         self.train_var_list.append(var + self.eps)
        # else:
        #     if len(self.test_var_list) < 1000:
        #         self.test_var_list.append(var + self.eps)

        if self.training and not self.training_use_cheb:
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        else:
            if self.training and not self.use_running_var_mean:
                var_normed = var / var_mean
                var_rescale = torch.sqrt(var_mean)
            else:
                var_normed = var / self.running_var_mean
                var_rescale = torch.sqrt(self.running_var_mean)
            
            if self.use_quad:
                _a = self.quad_coeffs[0] + self.quad_finetune_param[0] * self.quad_finetune_factors[0]
                _b = self.quad_coeffs[1] + self.quad_finetune_param[1] * self.quad_finetune_factors[1]
                _c = self.quad_coeffs[2] + self.quad_finetune_param[2] * self.quad_finetune_factors[2]
                cheb_result = _a * (var_normed - _b)**2 + _c
            else:
                cheb_result = self.cheb.calculate(var_normed + self.eps, int(self.cheb_params[0]), self.cheb_params[1], self.cheb_params[2])

            if self.training:
                var_mask = var_normed > self.var_norm_boundary
                cheb_result[var_mask] = 1.0 / torch.sqrt(var_normed[var_mask] + self.eps)

            x_norm = (x - mean) * cheb_result / (var_rescale * 0.35)

        # Scale and shift $$\text{LN}(x) = \gamma \hat{X} + \beta$$
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        return x_norm
    
    def save_counts_to_total(self):
        if self.total_counts_train is None:
            self.total_counts_train = self.counts_train
        else:
            self.total_counts_train += self.counts_train
        self.counts_train = None
    
        if self.total_counts_test is None:
            self.total_counts_test = self.counts_test
        else:
            self.total_counts_test += self.counts_test
        self.counts_test = None
    


