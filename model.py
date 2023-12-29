import torch
import torch.nn as nn
import torch.nn.functional as F

class general_relu_poly(nn.Module):
    def __init__(self, if_channel, if_pixel, weight_inits, factors, num_channels):
        super(general_relu_poly, self).__init__()
        self.if_channel = if_channel
        self.if_pixel = if_pixel
        self.factors = factors
        self.num_channels = num_channels
        self.rand_mask = None

        if len(weight_inits) != 3:
            raise ValueError("weight_inits must be of length 3")
        if len(factors) != 3:
            raise ValueError("factors must be of length 3")
        if if_channel:
            initial_weights = torch.zeros(num_channels, 3)
            for i, weight_init in enumerate(weight_inits):
                initial_weights[:, i] = weight_init  
            self.weight = nn.Parameter(initial_weights, requires_grad=True)
        else:
            self.weight = nn.Parameter(torch.FloatTensor(weight_inits), requires_grad=True)
    
    def forward(self, input, mask):
        if self.if_channel:
            weights = self.weight.unsqueeze(-1).unsqueeze(-1)
            weights = weights.expand(-1, -1, input.size(2), input.size(3))
            # square_term = weights[:, 0, :, :] * torch.mul(input, input) * self.factors[0]
            # linear_term = weights[:, 1, :, :] * input * self.factors[1]
            # constant_term = weights[:, 2, :, :] * self.factors[2]
            y = (weights[:, 0, :, :] * self.factors[0] * input + weights[:, 1, :, :] * self.factors[1]) * input + weights[:, 2, :, :] * self.factors[2]
            
        else:
            y = (self.weight[0] * self.factors[0] * input + self.weight[1] * self.factors[1]) * input + self.weight[2] * self.factors[2]

        if self.if_pixel:
            if self.rand_mask is None:
                self.rand_mask = nn.Parameter(torch.rand(input.shape[1:], device=input.device), requires_grad=False)
            if_relu = mask > self.rand_mask
            y = F.relu(input) * if_relu.float() + y * (1 - if_relu.float())
        else:
            y = F.relu(input) * mask + y * (1 - mask)

        return y
    
    def get_relu_density(self, mask):
        if not self.if_pixel:
            raise ValueError("get_relu_density can only be called when if_pixel is True")
    
        if_relu = mask > self.rand_mask
        total_elements = self.rand_mask.numel()
        relu_elements = if_relu.sum().item()
        return total_elements, relu_elements


class BasicBlockPoly(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, if_channel, if_pixel, weight_inits, factors, relu2_extra_factor=1):
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

        self.relu1 = general_relu_poly(if_channel, if_pixel, weight_inits, factors, planes)
        relu2_factors = factors
        relu2_factors[0] = factors[0] * relu2_extra_factor
        self.relu2 = general_relu_poly(if_channel, if_pixel, weight_inits, relu2_factors, planes)

    def forward(self, x, mask):
        out = self.relu1(self.bn1(self.conv1(x)), mask)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out, mask)
        return out

    def forward_with_fms(self, x, mask):
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


class ResNetPoly(nn.Module):
    def __init__(self, block, num_blocks, num_classes, if_channel, if_pixel, poly_weight_inits, poly_factors, relu2_extra_factor):
        super(ResNetPoly, self).__init__()
        self.in_planes = 64

        self.if_channel = if_channel
        self.if_pixel = if_pixel
        self.poly_weight_inits = poly_weight_inits
        self.poly_factors = poly_factors
        self.relu2_extra_factor = relu2_extra_factor

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1_0, self.layer1_1 = self._create_blocks(block, 64, num_blocks[0], stride=1)
        self.layer2_0, self.layer2_1 = self._create_blocks(block, 128, num_blocks[1], stride=2)
        self.layer3_0, self.layer3_1 = self._create_blocks(block, 256, num_blocks[2], stride=2)
        self.layer4_0, self.layer4_1 = self._create_blocks(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.relu1 = general_relu_poly(if_channel, if_pixel, poly_weight_inits, poly_factors, 64)

        self.if_forward_with_fms = False

    def _create_blocks(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            if planes == 512 and stride == 1:
                blocks.append(block(self.in_planes, planes, stride, self.if_channel, self.if_pixel, self.poly_weight_inits, self.poly_factors, self.relu2_extra_factor))
            else:
                blocks.append(block(self.in_planes, planes, stride, self.if_channel, self.if_pixel, self.poly_weight_inits, self.poly_factors))

            self.in_planes = planes * block.expansion
        return blocks

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
        
    def forward(self, x_and_mask):
        x, mask = x_and_mask
        if self.if_forward_with_fms:
            out, fms = self.forward_with_fms(x, mask)
            return (out, fms)
        else:
            return self.forward_without_fms(x, mask)

    def forward_without_fms(self, x, mask):

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

    def get_relu_density(self, mask):
        total1, relu1 = self.relu1.get_relu_density(mask)
        total1_0, relu1_0 = self.layer1_0.get_relu_density(mask)
        total1_1, relu1_1 = self.layer1_1.get_relu_density(mask)
        total2_0, relu2_0 = self.layer2_0.get_relu_density(mask)
        total2_1, relu2_1 = self.layer2_1.get_relu_density(mask)
        total3_0, relu3_0 = self.layer3_0.get_relu_density(mask)
        total3_1, relu3_1 = self.layer3_1.get_relu_density(mask)
        total4_0, relu4_0 = self.layer4_0.get_relu_density(mask)
        total4_1, relu4_1 = self.layer4_1.get_relu_density(mask)
        
        total_sum = total1 + total1_0 + total1_1 + total2_0 + total2_1 + total3_0 + total3_1 + total4_0 + total4_1
        relu_sum = relu1 + relu1_0 + relu1_1 + relu2_0 + relu2_1 + relu3_0 + relu3_1 + relu4_0 + relu4_1

        return total_sum, relu_sum
        
def ResNet18Poly(if_channel, if_pixel, poly_weight_inits, poly_factors, relu2_extra_factor=1):
    return ResNetPoly(BasicBlockPoly, [2, 2, 2, 2], 1000, if_channel, if_pixel, poly_weight_inits, poly_factors, relu2_extra_factor)


def initialize_resnet(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def convert_to_bf16_except_bn(model):
    for module in model.modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.to(dtype=torch.bfloat16)
    return model

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