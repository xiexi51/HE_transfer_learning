import torch
import torch.nn as nn
import torch.nn.functional as F

class relu_poly(nn.Module):
    def __init__(self, factor1 = 0.01, factor2 = 1, factor3 = 0.01):
        super(relu_poly, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor([0.00, 1, 0.00]), requires_grad=True)
        self.factor1 = factor1
        self.factor2 = factor2
        self.factor3 = factor3

    def forward(self, input, mask):
        y = self.weight[0]*torch.mul(input, input)*self.factor1 + self.weight[1]*input*self.factor2 + self.weight[2]*self.factor3
        y = F.relu(input) * mask + y * (1 - mask)
        return y

class channelwise_relu_poly(nn.Module):
    def __init__(self, num_channels, factor1=0.1, factor2=1, factor3=0.1):
        super(channelwise_relu_poly, self).__init__()
        initial_weights = torch.zeros(num_channels, 3)
        initial_weights[:, 1] = 1  
        self.weight = nn.Parameter(initial_weights, requires_grad=True)
        self.factor1 = factor1
        self.factor2 = factor2
        self.factor3 = factor3
        self.num_channels = num_channels

    def forward(self, input, mask):
        weights = self.weight.unsqueeze(-1).unsqueeze(-1)
        weights = weights.expand(-1, -1, input.size(2), input.size(3))

        square_term = weights[:, 0, :, :] * torch.mul(input, input) * self.factor1
        linear_term = weights[:, 1, :, :] * input * self.factor2
        constant_term = weights[:, 2, :, :] * self.factor3

        y = square_term + linear_term + constant_term
        y = F.relu(input) * mask + y * (1 - mask)
        return y

class BasicBlockPoly(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockPoly, self).__init__()
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
        
        self.relu1 = channelwise_relu_poly(planes)
        self.relu2 = channelwise_relu_poly(planes)

    def forward(self, x, mask):
        out = self.relu1(self.bn1(self.conv1(x)), mask)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out, mask)
        return out


class ResNetPoly(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetPoly, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_0, self.layer1_1 = self._create_blocks(block, 64, num_blocks[0], stride=1)
        self.layer2_0, self.layer2_1 = self._create_blocks(block, 128, num_blocks[1], stride=2)
        self.layer3_0, self.layer3_1 = self._create_blocks(block, 256, num_blocks[2], stride=2)
        self.layer4_0, self.layer4_1 = self._create_blocks(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.relu1 = channelwise_relu_poly(64)

    def _create_blocks(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return blocks

    def forward(self, x, mask):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out, mask)
        out = self.maxpool1(out)
        out = self.layer1_0(out, mask)
        out = self.layer1_1(out, mask)
        out = self.layer2_0(out, mask)
        out = self.layer2_1(out, mask)
        out = self.layer3_0(out, mask)
        out = self.layer3_1(out, mask)
        out = self.layer4_0(out, mask)
        out = self.layer4_1(out, mask)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def forward_save(self, x, mask, save_dir):
        torch.save(x, f"{save_dir}/x_before_conv1.pt")
        torch.save(self.conv1.weight, f"{save_dir}/conv1_weight.pt")

        out = self.conv1(x)
        torch.save(out, f"{save_dir}/x_after_conv1.pt")

        out = self.bn1(out)
        torch.save(out, f"{save_dir}/bn1_activation.pt")

        out = self.relu1(out, mask)
        torch.save(out, f"{save_dir}/relu1.pt")
        out = self.maxpool1(out)
        torch.save(out, f"{save_dir}/maxpool1.pt")

        out = self.layer1_0(out, mask)
        torch.save(out, f"{save_dir}/layer1_0.pt")
        out = self.layer1_1(out, mask)
        torch.save(out, f"{save_dir}/layer1_1.pt")

        out = self.layer2_0(out, mask)
        torch.save(out, f"{save_dir}/layer2_0.pt")
        out = self.layer2_1(out, mask)
        torch.save(out, f"{save_dir}/layer2_1.pt")

        out = self.layer3_0(out, mask)
        torch.save(out, f"{save_dir}/layer3_0.pt")
        out = self.layer3_1(out, mask)
        torch.save(out, f"{save_dir}/layer3_1.pt")

        out = self.layer4_0(out, mask)
        torch.save(out, f"{save_dir}/layer4_0.pt")
        out = self.layer4_1(out, mask)
        torch.save(out, f"{save_dir}/layer4_1.pt")

        out = F.adaptive_avg_pool2d(out, (1, 1))
        torch.save(out, f"{save_dir}/avg_pool2d.pt")
        out = out.view(out.size(0), -1)
        torch.save(out, f"{save_dir}/out_view.pt")
        out = self.linear(out)
        torch.save(out, f"{save_dir}/linear.pt")
        return out
    
    def forward_with_fm(self, x, mask):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out, mask)
        out = self.maxpool1(out)
        out = self.layer1_0(out, mask)
        out = self.layer1_1(out, mask)
        #export
        fm1 = out
        
        out = self.layer2_0(out, mask)
        out = self.layer2_1(out, mask)
        #export
        fm2 = out

        out = self.layer3_0(out, mask)
        out = self.layer3_1(out, mask)
        #export
        fm3 = out

        out = self.layer4_0(out, mask)
        out = self.layer4_1(out, mask)
        #export
        fm4 = out

        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, fm1, fm2, fm3, fm4

    def train_fz_bn(self, freeze_bn=True, freeze_bn_affine=True, mode=True):
        """
            Override the default train() to freeze the BN parameters
        """
        self.train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                if (freeze_bn_affine and m.affine == True):
                    m.weight.requires_grad = not freeze_bn
                    m.bias.requires_grad = not freeze_bn

def ResNet18Poly():
    return ResNetPoly(BasicBlockPoly, [2, 2, 2, 2], 1000)
