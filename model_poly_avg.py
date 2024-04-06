import torch
import torch.nn as nn
import torch.nn.functional as F
from model import fix_relu_poly, general_relu_poly
    
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


class BasicBlockAvgCustom(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, relu_type, poly_weight_inits, poly_factors):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.relu1 = custom_relu(relu_type, poly_weight_inits, poly_factors, planes)
        self.relu2 = custom_relu(relu_type, poly_weight_inits, poly_factors, planes)

    def forward(self, x, mask):
        fms = []
        out = self.relu1(self.bn1(self.conv1(x)), mask)
        fms.append(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out, mask)
        fms.append(out)
        return out, fms

    def get_relu_density(self, mask):
        total1, relu1 = self.relu1.get_relu_density(mask)
        total2, relu2 = self.relu2.get_relu_density(mask)
        return total1 + total2, relu1 + relu2


class ResNetAvgCustom(nn.Module):
    def __init__(self, block, num_blocks, num_classes, relu_type, poly_weight_inits, poly_factors):
        super().__init__()
        self.in_planes = 64

        self.relu_type = relu_type
        self.poly_weight_inits = poly_weight_inits
        self.poly_factors = poly_factors

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        

        self.layer1 = self._create_blocks(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._create_blocks(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._create_blocks(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._create_blocks(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.relu1 = custom_relu(relu_type, poly_weight_inits, poly_factors, 64)

        self.if_forward_with_fms = False

    def _create_blocks(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(block(self.in_planes, planes, stride, self.relu_type, self.poly_weight_inits, self.poly_factors))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*blocks)
        
    def forward(self, x_and_mask):
        x, mask = x_and_mask

        fms = []
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out, mask)
        fms.append(out)

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers:
            for block in layer:
                out, _fms = block(out, mask)
                fms += _fms

        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        if self.if_forward_with_fms:
            return (out, fms)
        else:
            return out

    def get_relu_density(self, mask):
        total, relu = self.relu1.get_relu_density(mask)
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers:
            for block in layer:
                _total, _relu = block.get_relu_density(mask)
                total += _total
                relu += _relu
        return total, relu
        
def ResNet18AvgCustom(relu_type, poly_weight_inits, poly_factors):
    return ResNetAvgCustom(BasicBlockAvgCustom, [2, 2, 2, 2], 1000, relu_type, poly_weight_inits, poly_factors)

