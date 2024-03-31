#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.registry import register_model
import os
from model import fix_relu_poly, general_relu_poly

# Series informed activation function. Implemented by conv.
class activation_full_poly(nn.ReLU):
    def __init__(self, act_relu_type, poly_weight_inits, poly_factors, dim, act_num=3, deploy=False):
        super().__init__()
        self.act_num = act_num
        self.deploy = deploy
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num*2 + 1, act_num*2 + 1))
        if deploy:
            self.bias = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None
            self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        nn.init.trunc_normal_(self.weight, std=.02)

        if act_relu_type == "channel":
            self.relu = general_relu_poly(if_channel=True, if_pixel=True, weight_inits=poly_weight_inits, factors=poly_factors, num_channels=dim)
        elif act_relu_type == "fix":
            self.relu = fix_relu_poly(if_pixel=True, factors=poly_factors)
        else:
            self.relu = nn.ReLU()

    def forward(self, x, mask):
        if isinstance(self.relu, nn.ReLU):
            x = self.relu(x)
        else:
            x = self.relu(x, mask)
        
        fm = x

        if self.deploy:    
            x = F.conv2d(x, self.weight, self.bias, padding=self.act_num, groups=self.dim)
        else:
            x = F.conv2d(x, self.weight, self.bias, padding=self.act_num, groups=self.dim)
            x = self.bn(x)
        return x, fm

    def get_relu_density(self, mask):
        return self.relu.get_relu_density(mask)

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class BlockAvgPoly(nn.Module):
    def __init__(self, act_relu_type, poly_weight_inits, poly_factors, dim, dim_out, act_num=3, stride=2, deploy=False, ada_pool=None, if_shortcut=True, keep_bn = False):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.act_learn = 1
        self.deploy = deploy
        self.stride = stride

        self.keep_bn = keep_bn

        if self.deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
            if self.keep_bn:
                self.bn = nn.BatchNorm2d(dim_out, eps=1e-6)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6)
            )

        if not ada_pool:
            self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(stride)
        else:
            self.pool = nn.Identity() if stride == 1 else nn.AdaptiveAvgPool2d((ada_pool, ada_pool))

        self.act = activation_full_poly(act_relu_type, poly_weight_inits, poly_factors, dim_out, act_num, deploy=self.deploy)

        if act_relu_type == "channel":
            self.relu = general_relu_poly(if_channel=True, if_pixel=True, weight_inits=poly_weight_inits, factors=poly_factors, num_channels=dim)
        elif act_relu_type == "fix":
            self.relu = fix_relu_poly(if_pixel=True, factors=poly_factors)
        else:
            self.relu = nn.ReLU()

        self.if_shortcut = if_shortcut

        if self.if_shortcut:
            # Shortcut connection
            self.use_pooling = stride != 1 or dim != dim_out
            if self.use_pooling:
                self.pooling = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0)
            
            self.adjust_channels = dim != dim_out
            if self.adjust_channels:
                self.channel_padding = nn.ConstantPad1d((0, dim_out - dim), 0)  # Only pad the last dim (channels)

 
    def forward(self, x, mask):
        fms = []

        if self.if_shortcut:
            identity = x
            if self.use_pooling:
                identity = self.pooling(identity)
            if self.adjust_channels:
                # Pad the channels without adding any parameters
                identity = F.pad(identity, (0, 0, 0, 0, 0, identity.size(1)), "constant", 0)

        if self.deploy:
            out = self.conv(x)
            if self.keep_bn:
                out = self.bn(out)
        else:
            out = self.conv1(x)
            
            if isinstance(self.relu, nn.ReLU):
                out = self.relu(out)
            else:
                out = self.relu(out, mask)

            fms.append(out)
            
            out = self.conv2(out)

        out = self.pool(out)

        # add bn here ?

        if self.if_shortcut:
            out += identity  # Add shortcut connection
        
        out, fm = self.act(out, mask)

        fms.append(fm)

        return out, fms

    def get_relu_density(self, mask):
        total1, relu1 = self.relu.get_relu_density(mask)
        total2, relu2 = self.act.get_relu_density(mask)
        return total1 + total2, relu1 + relu2

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        self.conv1[0].weight.data = kernel
        self.conv1[0].bias.data = bias

        if self.keep_bn:
            kernel, bias = self.conv2[0].weight.data, self.conv2[0].bias.data
            self.bn = self.conv2[1]
        else:
            kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight.data = torch.matmul(kernel.transpose(1,3), self.conv1[0].weight.data.squeeze(3).squeeze(2)).transpose(1,3)
        self.conv.bias.data = bias + (self.conv1[0].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)

        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True
    

class VanillaNetAvgPoly(nn.Module):
    def __init__(self, act_relu_type, poly_weight_inits, poly_factors, in_chans=3, num_classes=1000, dims=[96, 192, 384, 768], 
                 drop_rate=0, act_num=3, strides=[2,2,2,1], deploy=False, ada_pool=None, if_shortcut=True, keep_bn=False, **kwargs):
        super().__init__()

        self.deploy = deploy
        self.keep_bn = keep_bn
        self.if_forward_with_fms = False

        stride, padding = (4, 0) if not ada_pool else (3, 1)
        if self.deploy:
            self.stem_conv = nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=stride, padding=padding)
            if self.keep_bn:
                self.stem_bn = nn.BatchNorm2d(dims[0], eps=1e-6)
            self.stem_act = activation_full_poly(act_relu_type, poly_weight_inits, poly_factors, dims[0], act_num, deploy=self.deploy)
        else:
            self.stem1 = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=stride, padding=padding),
                nn.BatchNorm2d(dims[0], eps=1e-6),
            )
            self.stem2 = nn.Sequential(
                nn.Conv2d(dims[0], dims[0], kernel_size=1, stride=1),
                nn.BatchNorm2d(dims[0], eps=1e-6),
                activation_full_poly(act_relu_type, poly_weight_inits, poly_factors, dims[0], act_num)
            )

        self.act_learn = 0

        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            if not ada_pool:
                stage = BlockAvgPoly(act_relu_type, poly_weight_inits, poly_factors, dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], deploy=deploy, if_shortcut=if_shortcut, keep_bn=keep_bn)
            else:
                stage = BlockAvgPoly(act_relu_type, poly_weight_inits, poly_factors, dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], deploy=deploy, ada_pool=ada_pool[i], if_shortcut=if_shortcut, keep_bn=keep_bn)
            self.stages.append(stage)
        self.depth = len(strides)

        if self.deploy:
            self.cls = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Dropout(drop_rate),
                nn.Conv2d(dims[-1], num_classes, 1),
            )
        else:
            self.cls1 = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Dropout(drop_rate),
                nn.Conv2d(dims[-1], num_classes, 1),
                nn.BatchNorm2d(num_classes, eps=1e-6),
            )
            self.cls2 = nn.Sequential(
                nn.Conv2d(num_classes, num_classes, 1)
            )

        if act_relu_type == "channel":
            self.stem_relu = general_relu_poly(if_channel=True, if_pixel=True, weight_inits=poly_weight_inits, factors=poly_factors, num_channels=dims[0])
            self.cls_relu = general_relu_poly(if_channel=True, if_pixel=True, weight_inits=poly_weight_inits, factors=poly_factors, num_channels=num_classes)
        elif act_relu_type == "fix":
            self.stem_relu = fix_relu_poly(if_pixel=True, factors=poly_factors)
            self.cls_relu = fix_relu_poly(if_pixel=True, factors=poly_factors)
        else:
            self.stem_relu = nn.ReLU()
            self.cls_relu = nn.ReLU()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def change_act(self, m):
        raise ValueError("no need to change act here")
        for i in range(self.depth):
            self.stages[i].act_learn = m
        self.act_learn = m

    def forward(self, x_and_mask):
        x, mask = x_and_mask
        fms = []

        if self.deploy:
            x = self.stem_conv(x)
            if self.keep_bn:
                x = self.stem_bn(x)
            x, _fm = self.stem_act(x, mask)
            fms.append(_fm)
        else:
            x = self.stem1(x)
            if isinstance(self.stem_relu, nn.ReLU):
                x = self.stem_relu(x)
            else:
                x = self.stem_relu(x, mask)
            fms.append(x)
            x = self.stem2[0](x)
            x = self.stem2[1](x)
            x, _fm = self.stem2[2](x, mask)
            fms.append(_fm)

        for i in range(self.depth):
            x, _fms = self.stages[i](x, mask)
            fms += _fms

        if self.deploy:
            x = self.cls(x)
        else:
            x = self.cls1(x)
            if isinstance(self.cls_relu, nn.ReLU):
                x = self.cls_relu(x)
            else:
                x = self.cls_relu(x, mask)
            fms.append(x)
            x = self.cls2(x)
        
        x = x.view(x.size(0),-1)

        if self.if_forward_with_fms:
            return (x, fms)
        else:
            return x
    
    def get_relu_density(self, mask):
        total, relu = self.stem_relu.get_relu_density(mask)
        _total, _relu = self.stem2[2].get_relu_density(mask)
        total += _total
        relu += _relu

        for i in range(self.depth):
            _total, _relu = self.stages[i].get_relu_density(mask)
            total += _total
            relu += _relu

        _total, _relu = self.cls_relu.get_relu_density(mask)
        total += _total
        relu += _relu
        
        return total, relu

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        self.stem2[2].switch_to_deploy()
        kernel, bias = self._fuse_bn_tensor(self.stem1[0], self.stem1[1])
        self.stem1[0].weight.data = kernel
        self.stem1[0].bias.data = bias
        if self.keep_bn:
            kernel, bias = self.stem2[0].weight.data, self.stem2[0].bias.data
        else:
            kernel, bias = self._fuse_bn_tensor(self.stem2[0], self.stem2[1])
        self.stem1[0].weight.data = torch.einsum('oi,icjk->ocjk', kernel.squeeze(3).squeeze(2), self.stem1[0].weight.data)
        self.stem1[0].bias.data = bias + (self.stem1[0].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        # self.stem = torch.nn.Sequential(*[self.stem1[0], self.stem2[2]])
        self.stem_conv = self.stem1[0]
        if self.keep_bn:
            self.stem_bn = self.stem2[1]
        self.stem_act = self.stem2[2]

        self.__delattr__('stem1')
        self.__delattr__('stem2')

        for i in range(self.depth):
            self.stages[i].switch_to_deploy()

        kernel, bias = self._fuse_bn_tensor(self.cls1[2], self.cls1[3])
        self.cls1[2].weight.data = kernel
        self.cls1[2].bias.data = bias
        kernel, bias = self.cls2[0].weight.data, self.cls2[0].bias.data
        self.cls1[2].weight.data = torch.matmul(kernel.transpose(1,3), self.cls1[2].weight.data.squeeze(3).squeeze(2)).transpose(1,3)
        self.cls1[2].bias.data = bias + (self.cls1[2].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.cls = torch.nn.Sequential(*self.cls1[0:3])
        self.__delattr__('cls1')
        self.__delattr__('cls2')
        self.deploy = True



@register_model
def vanillanet_5_avg_full_poly(act_relu_type, poly_weight_inits, poly_factors, if_shortcut, keep_bn, **kwargs):
    model = VanillaNetAvgPoly(act_relu_type, poly_weight_inits, poly_factors, dims=[128*4, 256*4, 512*4, 1024*4], strides=[2,2,2], if_shortcut=if_shortcut, keep_bn=keep_bn, **kwargs)
    return model

@register_model
def vanillanet_6_avg_full_poly(act_relu_type, poly_weight_inits, poly_factors, if_shortcut, keep_bn, **kwargs):
    model = VanillaNetAvgPoly(act_relu_type, poly_weight_inits, poly_factors, dims=[128*4, 256*4, 512*4, 1024*4, 1024*4], strides=[2,2,2,1], if_shortcut=if_shortcut, keep_bn=keep_bn, **kwargs)
    return model

# @register_model
# def vanillanet_7(pretrained=False,in_22k=False, **kwargs):
#     model = VanillaNet(dims=[128*4, 128*4, 256*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,2,1], **kwargs)
#     return model

# @register_model
# def vanillanet_8(pretrained=False, in_22k=False, **kwargs):
#     model = VanillaNet(dims=[128*4, 128*4, 256*4, 512*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,1,2,1], **kwargs)
#     return model

# @register_model
# def vanillanet_9(pretrained=False, in_22k=False, **kwargs):
#     model = VanillaNet(dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,1,1,2,1], **kwargs)
#     return model

# @register_model
# def vanillanet_10(pretrained=False, in_22k=False, **kwargs):
#     model = VanillaNet(
#         dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
#         strides=[1,2,2,1,1,1,2,1],
#         **kwargs)
#     return model

# @register_model
# def vanillanet_11(pretrained=False, in_22k=False, **kwargs):
#     model = VanillaNet(
#         dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
#         strides=[1,2,2,1,1,1,1,2,1],
#         **kwargs)
#     return model

# @register_model
# def vanillanet_12(pretrained=False, in_22k=False, **kwargs):
#     model = VanillaNet(
#         dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
#         strides=[1,2,2,1,1,1,1,1,2,1],
#         **kwargs)
#     return model

# @register_model
# def vanillanet_13(pretrained=False, in_22k=False, **kwargs):
#     model = VanillaNet(
#         dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
#         strides=[1,2,2,1,1,1,1,1,1,2,1],
#         **kwargs)
#     return model

# @register_model
# def vanillanet_13_x1_5(pretrained=False, in_22k=False, **kwargs):
#     model = VanillaNet(
#         dims=[128*6, 128*6, 256*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 1024*6, 1024*6],
#         strides=[1,2,2,1,1,1,1,1,1,2,1],
#         **kwargs)
#     return model

# @register_model
# def vanillanet_13_x1_5_ada_pool(pretrained=False, in_22k=False, **kwargs):
#     model = VanillaNet(
#         dims=[128*6, 128*6, 256*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 1024*6, 1024*6],
#         strides=[1,2,2,1,1,1,1,1,1,2,1],
#         ada_pool=[0,38,19,0,0,0,0,0,0,10,0],
#         **kwargs)
#     return model
