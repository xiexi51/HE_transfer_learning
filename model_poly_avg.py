import torch
import torch.nn as nn
import torch.nn.functional as F

class relu_fullpoly(nn.Module):
    def __init__(self, weight_inits, factors, num_channels):
        super(relu_fullpoly, self).__init__()
        self.factors = factors
        self.num_channels = num_channels
        if len(weight_inits) != 3:
            raise ValueError("weight_inits must be of length 3")
        if len(factors) != 3:
            raise ValueError("factors must be of length 3")
        initial_weights = torch.zeros(num_channels, 3)
        for i, weight_init in enumerate(weight_inits):
            initial_weights[:, i] = weight_init  
        self.weight = nn.Parameter(initial_weights, requires_grad=True)
        
    def forward(self, input):
        weights = self.weight.unsqueeze(-1).unsqueeze(-1)
        weights = weights.expand(-1, -1, input.size(2), input.size(3))
        y = (weights[:, 0, :, :] * self.factors[0] * input + weights[:, 1, :, :] * self.factors[1]) * input + weights[:, 2, :, :] * self.factors[2]
        return y


class BasicBlockFullPoly(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, factors, relu2_extra_factor=1):
        super(BasicBlockFullPoly, self).__init__()
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

        self.relu1 = relu_fullpoly([0, 1, 0], factors, planes)
        relu2_factors = factors
        relu2_factors[0] = factors[0] * relu2_extra_factor
        self.relu2 = relu_fullpoly([0, 1, 0], relu2_factors, planes)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out

    def forward_with_fms(self, x):
        fms = []
        out = self.relu1(self.bn1(self.conv1(x)))
        fms.append(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        fms.append(out)
        return out, fms


class ResNetFullPoly(nn.Module):
    def __init__(self, block, num_blocks, num_classes, poly_factors, relu2_extra_factor):
        super(ResNetFullPoly, self).__init__()
        self.in_planes = 64

        self.poly_factors = poly_factors
        self.relu2_extra_factor = relu2_extra_factor

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_0, self.layer1_1 = self._create_blocks(block, 64, num_blocks[0], stride=1)
        self.layer2_0, self.layer2_1 = self._create_blocks(block, 128, num_blocks[1], stride=2)
        self.layer3_0, self.layer3_1 = self._create_blocks(block, 256, num_blocks[2], stride=2)
        self.layer4_0, self.layer4_1 = self._create_blocks(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.relu1 = relu_fullpoly([0, 1, 0], poly_factors, 64)

        self.if_forward_with_fms = False

    def _create_blocks(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            if planes == 512 and stride == 1:
                blocks.append(block(self.in_planes, planes, stride, self.poly_factors, self.relu2_extra_factor))
            else:
                blocks.append(block(self.in_planes, planes, stride, self.poly_factors))

            self.in_planes = planes * block.expansion
        return blocks
        
    def forward(self, x_and_avgmask):
        x, avgmask = x_and_avgmask
        if self.if_forward_with_fms:
            out, fms = self.forward_with_fms(x, avgmask)
            return (out, fms)
        else:
            return self.forward_without_fms(x, avgmask)

    def forward_without_fms(self, x, avgmask):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = avgmask * self.maxpool1(out) + (1-avgmask) * self.avgpool1(out)
        out = self.layer1_0(out)
        out = self.layer1_1(out)
        out = self.layer2_0(out)
        out = self.layer2_1(out)
        out = self.layer3_0(out)
        out = self.layer3_1(out)
        out = self.layer4_0(out)
        out = self.layer4_1(out)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
    
    def forward_with_fms(self, x, avgmask):
        fms = []
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        fms.append(out)
        out = avgmask * self.maxpool1(out) + (1-avgmask) * self.avgpool1(out)
        out, _fms = self.layer1_0.forward_with_fms(out)
        fms += _fms
        out, _fms = self.layer1_1.forward_with_fms(out)
        fms += _fms
        
        out, _fms = self.layer2_0.forward_with_fms(out)
        fms += _fms
        out, _fms = self.layer2_1.forward_with_fms(out)
        fms += _fms

        out, _fms = self.layer3_0.forward_with_fms(out)
        fms += _fms
        out, _fms = self.layer3_1.forward_with_fms(out)
        fms += _fms

        out, _fms = self.layer4_0.forward_with_fms(out)
        fms += _fms
        out, _fms = self.layer4_1.forward_with_fms(out)
        fms += _fms

        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, fms
        
def ResNet18FullPoly(poly_factors, relu2_extra_factor=1):
    return ResNetFullPoly(BasicBlockFullPoly, [2, 2, 2, 2], 1000, poly_factors, relu2_extra_factor)

def find_submodule(module, submodule_name):
    names = submodule_name.split('.')
    for name in names:
        module = getattr(module, name, None)
        if module is None:
            return None
    return module

def copy_parameters(model1, model2):
    for name1 in model1.state_dict():
        param1 = model1.state_dict()[name1]
        if not isinstance(param1, torch.Tensor):
            continue
        if name1.startswith("layer"):
            name2 = name1[:6] + "_" + name1[7:]
        elif name1.startswith("fc"):
            name2 = name1.replace("fc", "linear", 1)
        else:
            name2 = name1
        
        name2 = name2.replace("downsample", "shortcut", 1)

        assert(name2 in model2.state_dict())    
        model2.state_dict()[name2].copy_(param1.data)