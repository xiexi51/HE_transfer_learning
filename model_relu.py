import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlockRelu(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockRelu, self).__init__()
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
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out
    
    def forward_with_fms(self, x):
        fms = []
        out = self.relu1(self.bn1(self.conv1(x)))
        fms.append(out.detach())
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        fms.append(out.detach())
        return out, fms
    
    def forward_with_fms_and_pre(self, x):
        fms = []
        fms_pre = []
        out = self.bn1(self.conv1(x))
        
        fms_pre.append(out.detach())
        out = self.relu1(out)
        fms.append(out.detach())
        
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        
        fms_pre.append(out.detach())
        out = self.relu2(out)
        fms.append(out.detach())
        return out, fms_pre, fms


class ResNetRelu(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetRelu, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_0, self.layer1_1 = self._create_blocks(block, 64, num_blocks[0], stride=1)
        self.layer2_0, self.layer2_1 = self._create_blocks(block, 128, num_blocks[1], stride=2)
        self.layer3_0, self.layer3_1 = self._create_blocks(block, 256, num_blocks[2], stride=2)
        self.layer4_0, self.layer4_1 = self._create_blocks(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.relu1 = nn.ReLU()

    def _create_blocks(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return blocks

    def forward_save(self, x, mask, save_dir):
        torch.save(x, f"{save_dir}/x_before_conv1.pt")
        torch.save(self.conv1.weight, f"{save_dir}/conv1_weight.pt")

        out = self.conv1(x)
        torch.save(out, f"{save_dir}/x_after_conv1.pt")

        out = self.bn1(out)
        torch.save(out, f"{save_dir}/bn1_activation.pt")

        out = self.relu1(out)
        torch.save(out, f"{save_dir}/relu1.pt")
        out = self.maxpool1(out)
        torch.save(out, f"{save_dir}/maxpool1.pt")

        out = self.layer1_0(out)
        torch.save(out, f"{save_dir}/layer1_0.pt")
        out = self.layer1_1(out)
        torch.save(out, f"{save_dir}/layer1_1.pt")

        out = self.layer2_0(out)
        torch.save(out, f"{save_dir}/layer2_0.pt")
        out = self.layer2_1(out)
        torch.save(out, f"{save_dir}/layer2_1.pt")

        out = self.layer3_0(out)
        torch.save(out, f"{save_dir}/layer3_0.pt")
        out = self.layer3_1(out)
        torch.save(out, f"{save_dir}/layer3_1.pt")

        out = self.layer4_0(out)
        torch.save(out, f"{save_dir}/layer4_0.pt")
        out = self.layer4_1(out)
        torch.save(out, f"{save_dir}/layer4_1.pt")

        out = F.adaptive_avg_pool2d(out, (1, 1))
        torch.save(out, f"{save_dir}/avg_pool2d.pt")
        out = out.view(out.size(0), -1)
        torch.save(out, f"{save_dir}/out_view.pt")
        out = self.linear(out)
        torch.save(out, f"{save_dir}/linear.pt")
        return out    

    def forward(self, x, mask):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool1(out)
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
    
    def forward_with_fms(self, x, mask):
        fms = []
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out, mask)
        fms.append(out)
        out = self.maxpool1(out)
        out, _fms = self.layer1_0.forward_with_fms(out, mask)
        fms += _fms
        out, _fms = self.layer1_1.forward_with_fms(out, mask)
        fms += _fms
        
        out, _fms = self.layer2_0.forward_with_fms(out, mask)
        fms += _fms
        out, _fms = self.layer2_1.forward_with_fms(out, mask)
        fms += _fms

        out, _fms = self.layer3_0.forward_with_fms(out, mask)
        fms += _fms
        out, _fms = self.layer3_1.forward_with_fms(out, mask)
        fms += _fms

        out, _fms = self.layer4_0.forward_with_fms(out, mask)
        fms += _fms
        out, _fms = self.layer4_1.forward_with_fms(out, mask)
        fms += _fms

        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, fms
    
    def forward_with_fms_and_pre(self, x, mask):
        fms_pre = []
        fms = []
        
        out = self.conv1(x)
        out = self.bn1(out)
        
        fms_pre.append(out.detach())
        out = self.relu1(out)
        fms.append(out.detach())
        
        out = self.maxpool1(out)
        
        out, _fms_pre, _fms = self.layer1_0.forward_with_fms_and_pre(out)
        fms_pre += _fms_pre
        fms += _fms
        
        
        out, _fms_pre, _fms = self.layer1_1.forward_with_fms_and_pre(out)
        fms_pre += _fms_pre
        fms += _fms
        
        fms_pre += _fms_pre
        out, _fms_pre, _fms = self.layer2_0.forward_with_fms_and_pre(out)
        fms += _fms
        
        
        out, _fms_pre, _fms = self.layer2_1.forward_with_fms_and_pre(out)
        fms_pre += _fms_pre
        fms += _fms

        
        out, _fms_pre, _fms = self.layer3_0.forward_with_fms_and_pre(out)
        fms_pre += _fms_pre
        fms += _fms
        
        
        out, _fms_pre, _fms = self.layer3_1.forward_with_fms_and_pre(out)
        fms_pre += _fms_pre
        fms += _fms

        
        out, _fms_pre, _fms = self.layer4_0.forward_with_fms_and_pre(out)
        fms_pre += _fms_pre
        fms += _fms
        
        
        out, _fms_pre, _fms = self.layer4_1.forward_with_fms_and_pre(out)
        fms_pre += _fms_pre
        fms += _fms

        out = F.adaptive_avg_pool2d(out, (1,1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out, fms_pre, fms

def ResNet18Relu():
    return ResNetRelu(BasicBlockRelu, [2, 2, 2, 2], 1000)
