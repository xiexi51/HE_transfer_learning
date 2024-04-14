import torch
import torch.nn as nn
import torch.nn.functional as F
from model import fix_relu_poly, general_relu_poly
from utils import STEFunction
    
class custom_relu(nn.Module):
    def __init__(self, relu_type, poly_weight_inits, poly_factors, num_channels):
        super().__init__()
        if relu_type == "channel":
            self.relu = general_relu_poly(if_channel=True, if_pixel=True, weight_inits=poly_weight_inits, factors=poly_factors, num_channels=num_channels)
        elif relu_type == "fix":
            self.relu = fix_relu_poly(if_pixel=True, factors=poly_factors)
        else:
            self.relu = nn.ReLU()
    
    def forward(self, x, mask):
        if isinstance(self.relu, nn.ReLU):
            x = self.relu(x)
        else:
            x = self.relu(x, mask)
        return x

    def get_relu_density(self, mask):
        return self.relu.get_relu_density(mask)

class Conv2dPruned(nn.Conv2d):
    def __init__(self, prune_type, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.prune_type = prune_type
        if self.prune_type == "pixel":
            self.weight_aux = nn.Parameter(torch.rand_like(self.weight))
        elif self.prune_type == "channel":
            self.weight_aux = nn.Parameter(torch.rand(out_channels))
        elif self.prune_type == "fixed_channel":
            self.weight_aux = nn.Parameter(torch.rand(out_channels), requires_grad=False)

    def forward(self, x, threshold):
        if self.prune_type == "pixel":
            mask = STEFunction.apply(self.weight_aux)
        elif self.prune_type == "channel":
            mask = STEFunction.apply(self.weight_aux)
            mask = mask.view(-1, 1, 1, 1).expand_as(self.weight)
        elif self.prune_type == "fixed_channel":
            mask = (threshold > self.weight_aux).float() 
            mask = mask.view(-1, 1, 1, 1).expand_as(self.weight) 
        else:
            mask = 1
        pruned_weight = self.weight * mask
        return F.conv2d(x, pruned_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def get_conv_density(self):
        mask = STEFunction.apply(self.weight_aux)
        total = mask.numel()
        active = torch.sum(mask)
        return total, active

class BasicBlockAvgCustom(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, relu_type, poly_weight_inits, poly_factors, prune_type):
        super().__init__()
        self.conv1 = Conv2dPruned(prune_type, in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dPruned(prune_type, planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2dPruned(prune_type, in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.relu1 = custom_relu(relu_type, poly_weight_inits, poly_factors, planes)
        self.relu2 = custom_relu(relu_type, poly_weight_inits, poly_factors, planes)

    def forward(self, x, mask, threshold):
        fms = []
        out = self.conv1(x, threshold)
        if len(self.shortcut._modules) > 0:
            shortcut = self.shortcut[0](x, threshold)
            shortcut = self.shortcut[1](shortcut)
        else:
            shortcut = self.shortcut(x)

        out = self.bn1(out)
        out = self.relu1(out, mask)
        fms.append(out)
        
        out = self.conv2(out, threshold)
        out = self.bn2(out)
        out += shortcut
        out = self.relu2(out, mask)
        fms.append(out)
        return out, fms

    def get_relu_density(self, mask):
        total1, relu1 = self.relu1.get_relu_density(mask)
        total2, relu2 = self.relu2.get_relu_density(mask)
        return total1 + total2, relu1 + relu2
    
    def get_conv_density(self):
        total1, active1 = self.conv1.get_conv_density()
        total2, active2 = self.conv2.get_conv_density()
        if len(self.shortcut._modules) > 0:
            total3, active3 = self.shortcut[0].get_conv_density()
        else:
            total3, active3 = 0, 0
            
        return total1 + total2 + total3, active1 + active2 + active3

class ResNetAvgCustom(nn.Module):
    def __init__(self, block, num_blocks, num_classes, relu_type, poly_weight_inits, poly_factors, if_wide, prune_type):
        super().__init__()
        self.relu_type = relu_type
        self.poly_weight_inits = poly_weight_inits
        self.poly_factors = poly_factors
        self.prune_type = prune_type

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.in_planes = 64
        self.conv1 = Conv2dPruned(prune_type, 3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        if not if_wide:
            self.layer1 = self._create_blocks(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._create_blocks(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._create_blocks(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._create_blocks(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512*block.expansion, num_classes)
        else:
            self.layer1 = self._create_blocks(block, 128, num_blocks[0], stride=1)
            self.layer2 = self._create_blocks(block, 256, num_blocks[1], stride=2)
            self.layer3 = self._create_blocks(block, 512, num_blocks[2], stride=2)
            self.layer4 = self._create_blocks(block, 1024, num_blocks[3], stride=2)
            self.linear = nn.Linear(1024*block.expansion, num_classes)

        self.relu1 = custom_relu(relu_type, poly_weight_inits, poly_factors, 64)

        self.if_forward_with_fms = False

    def _create_blocks(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(block(self.in_planes, planes, stride, self.relu_type, self.poly_weight_inits, self.poly_factors, self.prune_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*blocks)
        
    def forward(self, x_mask_threshold):
        x, mask, threshold = x_mask_threshold

        fms = []
        
        out = self.conv1(x, threshold)
        out = self.bn1(out)
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
        total, active = self.conv1.get_conv_density()
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers:
            for block in layer:
                _total, _active = block.get_conv_density()
                total += _total
                active += _active
        return total, active
        
def ResNet18AvgCustom(relu_type, poly_weight_inits, poly_factors, if_wide, prune_type):
    return ResNetAvgCustom(BasicBlockAvgCustom, [2, 2, 2, 2], 1000, relu_type, poly_weight_inits, poly_factors, if_wide, prune_type)


