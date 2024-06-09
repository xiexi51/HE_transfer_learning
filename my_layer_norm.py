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

        self.ln_x_scaler = 1
        self.var_norm_scaler = 1

        self.ln_group_size = 64
        self.group_num = 1
        self.ln_momentum = None

        self.filter_var_mean = -1
        self.filter_var_mean_times = 0

        self.mask = 0

        self.norm_type = "my_layernorm"
        self.origin_norm = None

        self.ln_use_quad = True
        self.ln_trainable_quad_finetune = True
        self.ln_quad_coeffs = [0.03, 10, 0.2]
        self.ln_quad_finetune_factors = [0.0001, 0.1, 0.001]
        self.quad_finetune_param = None

        self.register_buffer('num_batches_tracked', torch.zeros(1))
        # self.running_var_mean = None

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

        self.ln_x_scaler = custom_settings.ln_x_scaler
        self.var_norm_scaler = custom_settings.var_norm_scaler

        self.ln_group_size = custom_settings.ln_group_size

        self.norm_type = custom_settings.norm_type

        self.ln_use_quad = custom_settings.ln_use_quad
        self.ln_trainable_quad_finetune = custom_settings.ln_trainable_quad_finetune
        self.ln_quad_coeffs = custom_settings.ln_quad_coeffs
        self.ln_quad_finetune_factors = custom_settings.ln_quad_finetune_factors
        if self.ln_trainable_quad_finetune:
            self.quad_finetune_param = nn.Parameter(torch.zeros(3), requires_grad=True)
        else:
            self.quad_finetune_param = nn.Parameter(torch.zeros(3), requires_grad=False)

    def forward(self, x: torch.Tensor):
        assert self.is_setup, "MyLayerNorm needs to be explicitly setup before forward pass."

        if self.normalized_shape is None:
            self.normalized_shape = x.size()[1:]

        x *= self.ln_x_scaler

        if self.norm_type == "layernorm":
            if self.origin_norm is None:
                self.origin_norm = nn.LayerNorm(self.normalized_shape, eps=self.eps, elementwise_affine=self.elementwise_affine)
            return self.origin_norm(x)
        elif self.norm_type == "batchnorm":
            if self.origin_norm is None:
                self.origin_norm = nn.BatchNorm2d(x.size()[1], eps=self.eps, track_running_stats=False)
            return self.origin_norm(x)
        
        if not hasattr(self, 'running_var_mean'):
            if self.ln_group_size > 0:
                self.group_num = x.shape[1] // self.ln_group_size
                self.register_buffer('running_var_mean', torch.ones(self.group_num))
            else:
                self.register_buffer('running_var_mean', torch.ones(1))
            # Create parameters for $\gamma$ and $\beta$ for gain and bias
            if self.norm_type == "my_layernorm" and self.elementwise_affine:
                self.gain = nn.Parameter(torch.ones(self.normalized_shape))
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

        # if self.epoch_train_var_mean is None:
        #     self.epoch_train_var_mean = 0
        #     self.epoch_train_var_sum = 0
        #     self.epoch_train_var_mean_count = 0
        
        # if self.epoch_test_var_mean is None:
        #     self.epoch_test_var_mean = 0
        #     self.epoch_test_var_sum = 0
        #     self.epoch_test_var_mean_count = 0        

        exponential_average_factor = 0.0

        if self.training:
            self.num_batches_tracked += 1
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            if self.ln_momentum is not None:
                if exponential_average_factor < self.ln_momentum:
                    exponential_average_factor = self.ln_momentum

        dims = [-(i + 1) for i in range(len(self.normalized_shape))]

        if self.ln_group_size > 0:
            assert x.shape[1] % self.ln_group_size == 0, f"Number of channels must be divisible by {self.ln_group_size}."
            x_grouped = x.view(x.shape[0], -1, self.ln_group_size, *x.shape[2:])
            mean = x_grouped.mean(dim=dims, keepdim=False)
            mean_x2 = (x_grouped ** 2).mean(dim=dims, keepdim=False)
        else:
            mean = x.mean(dim=dims, keepdim=True)
            mean_x2 = (x ** 2).mean(dim=dims, keepdim=True)
        var = mean_x2 - mean ** 2

        # x_norm = torch.zeros_like(x_reshaped, dtype=x_reshaped.dtype, device=x_reshaped.device)
        
        var_mean = var.mean(dim=0).squeeze()

        self.saved_var_mean = var_mean

        # if self.training and self.filter_var_mean:
        #     if var_mean > self.running_var_mean * 10:
        #         x_norm = (x - mean) / torch.sqrt(var + self.eps)
        #         if self.elementwise_affine:
        #             x_norm = self.gain * x_norm + self.bias
        #         return x_norm
        
        # if self.training and self.filter_var_mean:
        #     mask = var_mean <= self.running_var_mean * self.var_norm_boundary
        # else:
        #     mask = torch.ones(var_mean.shape, dtype=torch.bool, device=var_mean.device)
        
        if self.ln_group_size > 0:
            mean = mean.repeat_interleave(self.ln_group_size, dim=1).unsqueeze(-1).unsqueeze(-1)
            var = var.repeat_interleave(self.ln_group_size, dim=1).unsqueeze(-1).unsqueeze(-1)

        if self.training and self.filter_var_mean > 0:
            if (var_mean > self.running_var_mean * self.filter_var_mean).any():
                self.filter_var_mean_times = 1
                x_norm = (x - mean) / torch.sqrt(var + self.eps)
                if self.elementwise_affine:
                    x_norm = self.gain * x_norm + self.bias
                return x_norm
            else:
                self.filter_var_mean_times = 0

        if self.training:
            self.epoch_train_var_mean += var_mean.mean().item()
            self.epoch_train_var_sum += 0
            self.epoch_train_var_mean_count += 1
        else:
            self.epoch_test_var_mean += var_mean.mean().item()
            self.epoch_test_var_sum += 0
            self.epoch_test_var_mean_count += 1    

        if self.ln_group_size > 0:
            var_mean = var_mean.repeat_interleave(self.ln_group_size).unsqueeze(-1).unsqueeze(-1)
            _running_var_mean = self.running_var_mean.repeat_interleave(self.ln_group_size).unsqueeze(-1).unsqueeze(-1)
        else:
            _running_var_mean = self.running_var_mean
        
        with torch.no_grad():
            if not self.training or self.use_running_var_mean:
                var_normed_counts = (var / _running_var_mean).squeeze().detach().cpu().numpy()
                
                bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf]
                counts, _ = np.histogram(var_normed_counts, bins=bins)

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

        if self.training and not self.training_use_cheb:
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        else:
            if self.training and not self.use_running_var_mean:
                var_normed = self.var_norm_scaler * var / var_mean
                var_rescale = torch.sqrt(var_mean / self.var_norm_scaler)
            else:
                var_normed = self.var_norm_scaler * var / _running_var_mean
                var_rescale = torch.sqrt(_running_var_mean / self.var_norm_scaler)
            
            if self.ln_use_quad:
                _a = self.ln_quad_coeffs[0] + self.quad_finetune_param[0] * self.ln_quad_finetune_factors[0]
                _b = self.ln_quad_coeffs[1] + self.quad_finetune_param[1] * self.ln_quad_finetune_factors[1]
                _c = self.ln_quad_coeffs[2] + self.quad_finetune_param[2] * self.ln_quad_finetune_factors[2]
                cheb_result = _a * (var_normed - _b) ** 2 + _c
            else:
                cheb_result = self.cheb.calculate(var_normed + self.eps, int(self.cheb_params[0]), self.cheb_params[1], self.cheb_params[2])

            if self.training:
                var_mask = var_normed > self.var_norm_boundary
                cheb_result[var_mask] = 1 / torch.sqrt(var_normed[var_mask] + self.eps)

            x_norm = (x - mean) * (cheb_result / var_rescale * (1 - self.mask) + 1 / torch.sqrt(var + self.eps) * self.mask)

        with torch.no_grad():
            if self.training:
                self.running_var_mean = (1 - exponential_average_factor) * self.running_var_mean + exponential_average_factor * self.saved_var_mean

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


def process_layer(layer, counts_type, f):
    if isinstance(layer, MyLayerNorm):
        layer_counts = getattr(layer, f"counts_{counts_type}", None)
        if layer_counts is not None:
            counts_ratio = layer_counts / layer_counts.sum()
            epoch_var_mean = getattr(layer, f"epoch_{counts_type}_var_mean", 0) / getattr(layer, f"epoch_{counts_type}_var_mean_count", 1)
            epoch_var_sum = getattr(layer, f"epoch_{counts_type}_var_sum", 0) / getattr(layer, f"epoch_{counts_type}_var_mean_count", 1)
            f.write(f"{layer.number} {layer.normalized_shape} ev {epoch_var_mean:.2f} evs {epoch_var_sum:.2f} rv_mean {layer.running_var_mean.mean().item():.2f} rv_max {layer.running_var_mean.max().item():.2f} rv_min {layer.running_var_mean.min().item():.2f}: " + " ".join(map(lambda x: "{:.5f}".format(x), counts_ratio)) + "\n")
    elif len(list(layer.children())) > 0:
        for child in layer.children():
            process_layer(child, counts_type, f)

def process_counts(model, epoch, log_file, sum_counts, counts_type):
    if sum_counts is not None:
        sum_counts_ratio = sum_counts / sum_counts.sum()
        print(f"{counts_type}: " + " ".join(map(lambda x: "{:.5f}".format(x), sum_counts_ratio)))
        with open(log_file, "a") as f:
            f.write(f"Epoch: {epoch}, {counts_type.capitalize()} counts ratio:\n")
            f.write("Sum: " + " ".join(map(lambda x: "{:.5f}".format(x), sum_counts_ratio)) + "\n")
            for layer in model.children():
                process_layer(layer, counts_type, f)

def get_ln_statistics(model, epoch, log_file):
    sum_train_counts = None
    sum_test_counts = None
    for layer in model.modules():
        if isinstance(layer, MyLayerNorm):
            if layer.counts_train is not None:
                sum_train_counts = layer.counts_train if sum_train_counts is None else sum_train_counts + layer.counts_train
            if layer.counts_test is not None:
                sum_test_counts = layer.counts_test if sum_test_counts is None else sum_test_counts + layer.counts_test
                
    process_counts(model, epoch, log_file, sum_train_counts, "train")
    process_counts(model, epoch, log_file, sum_test_counts, "test")