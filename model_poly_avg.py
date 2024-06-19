import torch
import torch.nn as nn
import torch.nn.functional as F
from model import fix_relu_poly, general_relu_poly, star_relu
from utils import STEFunction
from my_layer_norm import MyLayerNorm
from utils import CustomSettings
import numpy as np
    
class custom_relu(nn.Module):
    def __init__(self, custom_settings):
        super().__init__()
        self.custom_settings = custom_settings
        self.relu = None
        self.mask = 0
        self.dropout = nn.Dropout2d(custom_settings.relu_dropout) if custom_settings.relu_dropout > 0 else None
        # self.saved_input_mean = None
        # self.saved_input_var = None
        # self.saved_output_mean = None
        # self.saved_output_var = None

        self.avg_times = 0
        self.avg_input_mean = 0
        self.avg_input_var = 0
        self.avg_output_mean = 0
        self.avg_output_var = 0
        self.input_vars = []

    def reset_stats(self):
        self.avg_times = 0
        self.avg_input_mean = 0
        self.avg_input_var = 0
        self.avg_output_mean = 0
        self.avg_output_var = 0
        self.input_vars = []
    
    def forward(self, x):
        # self.saved_input_mean = x.mean()
        # self.saved_input_var = x.var()
        self.avg_times += 1
        self.avg_input_mean += x.mean().item()
        self.avg_input_var += x.var().item()
        self.input_vars.append(x.var().item())

        if self.relu is None:
            num_channels = x.shape[1]
            if self.custom_settings.relu_type == "channel":
                self.relu = general_relu_poly(if_channel=True, if_pixel=False, weight_inits=self.custom_settings.poly_weight_inits, 
                                              factors=self.custom_settings.poly_factors, num_channels=num_channels)
            elif self.custom_settings.relu_type == "fix":
                self.relu = fix_relu_poly(if_pixel=False, factors=self.custom_settings.poly_factors)
            elif self.custom_settings.relu_type == "star":
                self.relu = star_relu(num_channels)
            else:
                self.relu = nn.ReLU()

        if isinstance(self.relu, nn.ReLU) or isinstance(self.relu, star_relu):
            x = self.relu(x)
        else:
            x = self.relu(x, self.mask)
        
        if self.dropout is not None:
            x = self.dropout(x)

        # self.saved_output_mean = x.mean()
        # self.saved_output_var = x.var()

        self.avg_output_mean += x.mean().item()
        self.avg_output_var += x.var().item()

        return x

    def get_relu_density(self):
        return self.relu.get_relu_density(self.mask)

def get_act_statistics(model, epoch, log_file):
    with open(log_file, "a") as f:
        f.write(f"Epoch: {epoch}\n")
        for layer in model.modules():
            if isinstance(layer, custom_relu):
                if layer.avg_times > 0:
                    avg_input_mean = layer.avg_input_mean / layer.avg_times
                    avg_input_var = layer.avg_input_var / layer.avg_times

                    input_vars_np = np.array(layer.input_vars)
                    q1 = np.percentile(input_vars_np, 25)
                    median = np.percentile(input_vars_np, 50)
                    q3 = np.percentile(input_vars_np, 75)

                    f.write(f"Layer: {layer.custom_settings.relu_type} avg_input_mean: {avg_input_mean:.2f} avg_input_var: {avg_input_var:.2f} median_input_var: {median:.2f} q1_input_var: {q1:.2f} q3_input_var: {q3:.2f}\n")
                    layer.reset_stats()

class Conv2dPruned(nn.Conv2d):
    def __init__(self, custom_settings, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.prune_type = custom_settings.prune_type
        self.prune_1_1_kernel = custom_settings.prune_1_1_kernel
        self.granularity = 8
        self.group_granularity = 8
        self.weight_aux = None
        if self.prune_type == "pixel":
            self.weight_aux = nn.Parameter(torch.rand_like(self.weight))
        elif self.prune_type == "channel":
            self.weight_aux = nn.Parameter(torch.rand(out_channels))
        elif self.prune_type == "fixed_channel":
            self.weight_aux = nn.Parameter(torch.rand(out_channels), requires_grad=False)
        elif self.prune_type == "group_pixel":
            if not (self.weight.shape[-2] == 1 and self.weight.shape[-1] == 1) or self.prune_1_1_kernel:
                if groups > 1 and groups == in_channels:
                    self.weight_aux = nn.Parameter(torch.rand(groups // self.group_granularity, self.weight.shape[-2], self.weight.shape[-1]))
                else:
                    if in_channels < self.granularity:
                        self.weight_aux = nn.Parameter(torch.rand(1, self.weight.shape[-2], self.weight.shape[-1]))
                    else:
                        self.weight_aux = nn.Parameter(torch.rand(in_channels // self.granularity, self.weight.shape[-2], self.weight.shape[-1]))

    def generate_mask_from_weight_aux(self, threshold):
        if self.prune_type == "pixel":
            mask = STEFunction.apply(self.weight_aux)
        elif self.prune_type == "channel":
            mask = STEFunction.apply(self.weight_aux)
            mask = mask.view(-1, 1, 1, 1).expand_as(self.weight)
        elif self.prune_type == "fixed_channel":
            mask = (threshold > self.weight_aux).float() 
            mask = mask.view(-1, 1, 1, 1).expand_as(self.weight) 
        elif self.prune_type == "group_pixel" and self.weight_aux is not None:
            if self.groups > 1:
                weight_aux_expanded = self.weight_aux.repeat_interleave(self.group_granularity, dim=0)
                weight_aux_expanded = weight_aux_expanded.unsqueeze(1)
            else:
                # Expand weight_aux to match the shape of self.weight
                weight_aux_expanded = self.weight_aux.repeat_interleave(self.granularity, dim=0)
                weight_aux_expanded = weight_aux_expanded[:self.weight.shape[1]]
                weight_aux_expanded = weight_aux_expanded.unsqueeze(0).expand(self.weight.shape[0], -1, -1, -1)
            # Apply STEFunction to create the mask
            mask = STEFunction.apply(weight_aux_expanded)
        else:
            mask = 1
        return mask

    def forward(self, x, threshold):
        mask = self.generate_mask_from_weight_aux(threshold)
        pruned_weight = self.weight * mask
        return F.conv2d(x, pruned_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def get_conv_density(self):
        if self.prune_type is None or self.weight_aux is None:
            return self.weight.numel(), self.weight.numel()
        mask = self.generate_mask_from_weight_aux(1)
        total = mask.numel()
        active = torch.sum(mask)
        return total, active

class BasicBlockAvgCustom(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, custom_settings, out_features):
        super().__init__()
        if custom_settings.norm_type == "my_layernorm":
            self.norm1 = MyLayerNorm()
            self.norm2 = MyLayerNorm()
        elif custom_settings.norm_type == "batchnorm":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
        else:
            self.norm1 = nn.LayerNorm([planes, out_features, out_features])
            self.norm2 = nn.LayerNorm([planes, out_features, out_features])
            
        self.conv1 = Conv2dPruned(custom_settings, in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = Conv2dPruned(custom_settings, planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            if custom_settings.norm_type == "my_layernorm":
                self.shortcut_norm = MyLayerNorm()
            elif custom_settings.norm_type == "batchnorm":
                self.shortcut_norm = nn.BatchNorm2d(self.expansion*planes)
            else:
                self.shortcut_norm = nn.LayerNorm([self.expansion*planes, out_features, out_features])
            self.shortcut = nn.Sequential(
                Conv2dPruned(custom_settings, in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                self.shortcut_norm
            )

        self.relu1 = custom_relu(custom_settings)
        self.relu2 = custom_relu(custom_settings)

    def forward(self, x, mask, threshold):
        fms = []

        if len(self.shortcut._modules) > 0:
            shortcut = self.shortcut[0](x, threshold)
            shortcut = self.shortcut[1](shortcut)
        else:
            shortcut = self.shortcut(x)

        out = self.conv1(x, threshold)
        out = self.norm1(out)
        out = self.relu1(out, mask)
        fms.append(out)
        
        out = self.conv2(out, threshold)
        out = self.norm2(out)
        out += shortcut
        out = self.relu2(out, mask)
        fms.append(out)
        return out, fms

    def get_relu_density(self, mask):
        total1, relu1 = self.relu1.get_relu_density(mask)
        total2, relu2 = self.relu2.get_relu_density(mask)
        return total1 + total2, relu1 + relu2

class ResNetAvgCustom(nn.Module):
    def __init__(self, block, num_blocks, num_classes, custom_settings, if_wide):
        super().__init__()
        self.custom_settings = custom_settings
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.in_planes = 64
        self.conv1 = Conv2dPruned(custom_settings, 3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if custom_settings.norm_type == "my_layernorm":
            self.norm1 = MyLayerNorm()
        elif custom_settings.norm_type == "batchnorm":
            self.norm1 = nn.BatchNorm2d(64)
        else:
            self.norm1 = nn.LayerNorm([64, 112, 112])

        if not if_wide:
            self.layer1 = self._create_blocks(block, 64, num_blocks[0], stride=1, out_features=56)
            self.layer2 = self._create_blocks(block, 128, num_blocks[1], stride=2, out_features=28)
            self.layer3 = self._create_blocks(block, 256, num_blocks[2], stride=2, out_features=14)
            self.layer4 = self._create_blocks(block, 512, num_blocks[3], stride=2, out_features=7)
            self.linear = nn.Linear(512*block.expansion, num_classes)
        else:
            self.layer1 = self._create_blocks(block, 128, num_blocks[0], stride=1, out_features=56)
            self.layer2 = self._create_blocks(block, 256, num_blocks[1], stride=2, out_features=28)
            self.layer3 = self._create_blocks(block, 512, num_blocks[2], stride=2, out_features=14)
            self.layer4 = self._create_blocks(block, 1024, num_blocks[3], stride=2, out_features=7)
            self.linear = nn.Linear(1024*block.expansion, num_classes)

        self.relu1 = custom_relu(custom_settings)
        self.if_forward_with_fms = False

        self.num_layernorms = 0
        i = 0
        for module in self.modules():
            if isinstance(module, MyLayerNorm):
                module.number = i
                i += 1
                self.num_layernorms += 1
                module.setup(self.custom_settings)

    def _create_blocks(self, block, planes, num_blocks, stride, out_features):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(block(self.in_planes, planes, stride, self.custom_settings, out_features))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*blocks)
        
    def forward(self, x_mask_threshold):
        x, mask, threshold = x_mask_threshold

        fms = []
        
        out = self.conv1(x, threshold)
        out = self.norm1(out)
        out = self.relu1(out, mask)
        fms.append(out)

        out = self.avgpool(out)
        
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers:
            for block in layer:
                out, _fms = block(out, mask, threshold)
                fms += _fms

        featuremap = out

        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)

        out = self.linear(out)

        if self.if_forward_with_fms:
            return (out, fms, featuremap)
        else:
            return (out, featuremap)

    def get_relu_density(self, mask):
        total, relu = self.relu1.get_relu_density(mask)
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers:
            for block in layer:
                _total, _relu = block.get_relu_density(mask)
                total += _total
                relu += _relu
        return total, relu
    
    def get_conv_density(self):
        total = 0
        active = 0
        for module in self.modules():
            if isinstance(module, Conv2dPruned):
                _total, _active = module.get_conv_density()
                if _total > 0:
                    total += _total
                    active += _active
        return total, active
    
    def get_ln_statistics(self, epoch, log_file):
        sum_train_counts = None
        sum_test_counts = None
        for layer in self.modules():
            if isinstance(layer, MyLayerNorm):
                if layer.counts_train is not None:
                    sum_train_counts = layer.counts_train if sum_train_counts is None else sum_train_counts + layer.counts_train
                if layer.counts_test is not None:
                    sum_test_counts = layer.counts_test if sum_test_counts is None else sum_test_counts + layer.counts_test
                
        if sum_train_counts is not None:
            sum_train_counts_ratio = sum_train_counts / sum_train_counts.sum()
            print("train: " + " ".join(map(lambda x: "{:.5f}".format(x), sum_train_counts_ratio)))
            with open(log_file, "a") as f:
                f.write(f"Epoch: {epoch}, Train counts ratio:\n")
                f.write("Sum: " + " ".join(map(lambda x: "{:.5f}".format(x), sum_train_counts_ratio)) + "\n")
                for layer in self.modules():
                    if isinstance(layer, MyLayerNorm) and layer.counts_train is not None:
                        counts_train_ratio = layer.counts_train / layer.counts_train.sum()
                        epoch_train_var_mean = layer.epoch_train_var_mean / layer.epoch_train_var_mean_count
                        epoch_train_var_sum = layer.epoch_train_var_sum / layer.epoch_train_var_mean_count
                        f.write(f"{layer.number} {layer.normalized_shape} ev {epoch_train_var_mean:.2f} evs {epoch_train_var_sum:.2f} rv {layer.running_var_mean.item():.2f}: " + " ".join(map(lambda x: "{:.5f}".format(x), counts_train_ratio)) + "\n")

        if sum_test_counts is not None:
            sum_test_counts_ratio = sum_test_counts / sum_test_counts.sum()
            print("test: " + " ".join(map(lambda x: "{:.5f}".format(x), sum_test_counts_ratio)))
            with open(log_file, "a") as f:
                f.write(f"Epoch: {epoch}, Test counts ratio:\n")
                f.write("Sum: " + " ".join(map(lambda x: "{:.5f}".format(x), sum_test_counts_ratio)) + "\n")
                for layer in self.modules():
                    if isinstance(layer, MyLayerNorm) and layer.counts_test is not None:
                        counts_test_ratio = layer.counts_test / layer.counts_test.sum()
                        epoch_test_var_mean = layer.epoch_test_var_mean / layer.epoch_test_var_mean_count
                        epoch_test_var_sum = layer.epoch_test_var_sum / layer.epoch_test_var_mean_count
                        f.write(f"{layer.number} {layer.normalized_shape} ev {epoch_test_var_mean:.2f} evs {epoch_test_var_sum:.2f} rv {layer.running_var_mean.item():.2f}: " + " ".join(map(lambda x: "{:.5f}".format(x), counts_test_ratio)) + "\n")
        
def ResNet18AvgCustom(custom_settings, if_wide):
    return ResNetAvgCustom(BasicBlockAvgCustom, [2, 2, 2, 2], 1000, custom_settings, if_wide)


