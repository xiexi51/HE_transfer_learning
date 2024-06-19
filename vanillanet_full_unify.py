#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.registry import register_model
import os
from model_poly_avg import Conv2dPruned, custom_relu
from my_layer_norm import MyLayerNorm

# Series informed activation function. Implemented by conv.
class activation_unify(nn.ReLU):
    def __init__(self, custom_settings, dim, act_num=3):
        super().__init__()
        self.act_num = act_num

        self.bn = MyLayerNorm()
        # self.bn = nn.BatchNorm2d(dim)

        self.conv = Conv2dPruned(custom_settings, in_channels=dim, out_channels=dim, kernel_size=(act_num * 2 + 1, act_num * 2 + 1), padding=act_num, groups=dim, bias=False)

        nn.init.trunc_normal_(self.conv.weight, std=0.02)

        self.relu = custom_relu(custom_settings)

    def forward(self, x, mask, threshold):
        self.relu.mask = mask
        x = self.relu(x)
        fm = x
        x = self.conv(x, threshold)
        x = self.bn(x)
        return x, fm

    # def get_relu_density(self, mask):
    #     return self.relu.get_relu_density(mask)
    
    # def get_conv_density(self):
    #     return self.conv.get_conv_density()


class BlockAvgPoly(nn.Module):
    def __init__(self, custom_settings, dim, dim_out, act_num=3, stride=2, ada_pool=None, if_shortcut=True, keep_bn = False):
        super().__init__()
        # self.dim = dim
        # self.dim_out = dim_out
        # self.act_learn = 1
        # self.stride = stride

        self.keep_bn = keep_bn
        
        self.conv1 = nn.Sequential(
            Conv2dPruned(custom_settings, dim, dim, kernel_size=3, padding=1),
            MyLayerNorm()
        )
        self.conv2 = nn.Sequential(
            Conv2dPruned(custom_settings, dim, dim_out, kernel_size=3, padding=1),
            MyLayerNorm()
        )

        if not ada_pool:
            self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(stride)
        else:
            self.pool = nn.Identity() if stride == 1 else nn.AdaptiveAvgPool2d((ada_pool, ada_pool))

        self.act = activation_unify(custom_settings, dim_out, act_num)

        self.relu = custom_relu(custom_settings)

        self.if_shortcut = if_shortcut

        if self.if_shortcut:
            # Shortcut connection
            self.use_pooling = stride != 1 or dim != dim_out
            if self.use_pooling:
                self.pooling = nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0)
            
            self.adjust_channels = dim != dim_out
            if self.adjust_channels:
                self.channel_padding = nn.ConstantPad1d((0, dim_out - dim), 0)  # Only pad the last dim (channels)

 
    def forward(self, x, mask, threshold):
        fms = []

        if self.if_shortcut:
            identity = x
            if self.use_pooling:
                identity = self.pooling(identity)
            if self.adjust_channels:
                # Pad the channels without adding any parameters
                identity = F.pad(identity, (0, 0, 0, 0, 0, identity.size(1)), "constant", 0)

        out = self.conv1[0](x, threshold)
        out = self.conv1[1](out)

        self.relu.mask = mask
        
        out = self.relu(out)
        

        fms.append(out)
        
        out = self.conv2[0](out, threshold)
        out = self.conv2[1](out)

        out = self.pool(out)

        if self.if_shortcut:
            out += identity  # Add shortcut connection
        
        out, fm = self.act(out, mask, threshold)

        fms.append(fm)

        return out, fms

class VanillaNetFullUnify(nn.Module):
    def __init__(self, custom_settings, num_classes, in_chans=3, dims=[96, 192, 384, 768], 
                 drop_rate=0, act_num=3, strides=[2,2,2,1], ada_pool=None, if_shortcut=True, keep_bn=False, **kwargs):
        super().__init__()

        self.keep_bn = keep_bn
        self.if_forward_with_fms = False
        # self.drop_rate = drop_rate

        stride, padding = (4, 0) if not ada_pool else (3, 1)
        
        self.stem1 = nn.Sequential(
            Conv2dPruned(custom_settings, in_chans, dims[0], kernel_size=4, stride=stride, padding=padding),
            MyLayerNorm()
        )
        self.stem2 = nn.Sequential(
            Conv2dPruned(custom_settings, dims[0], dims[0], kernel_size=3, padding=1, stride=1),
            MyLayerNorm(),
            activation_unify(custom_settings, dims[0], act_num)
        )

        # self.act_learn = 0

        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            if not ada_pool:
                stage = BlockAvgPoly(custom_settings, dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], if_shortcut=if_shortcut, keep_bn=keep_bn)
            else:
                stage = BlockAvgPoly(custom_settings, dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], ada_pool=ada_pool[i], if_shortcut=if_shortcut, keep_bn=keep_bn)
            self.stages.append(stage)
        self.depth = len(strides)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(custom_settings.drop_rate)

        self.linear = nn.Linear(dims[-1], num_classes)

        self.stem_relu = custom_relu(custom_settings)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_mask_threshold):
        x, mask, threshold = x_mask_threshold
        fms = []

        x = self.stem1[0](x, threshold)
        x = self.stem1[1](x)

        self.stem_relu.mask = mask
        x = self.stem_relu(x)
        
        fms.append(x)

        x = self.stem2[0](x, threshold)
        x = self.stem2[1](x)
        x, _fm = self.stem2[2](x, mask, threshold)
        fms.append(_fm)

        for i in range(self.depth):
            x, _fms = self.stages[i](x, mask, threshold)
            fms += _fms

        featuremap = x

        x = self.avgpool(x)

        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)

        # fms.append(x)

        if self.if_forward_with_fms:
            return (x, fms, featuremap)
        else:
            return (x, featuremap)
    
    def get_relu_density(self, mask):
        # total, relu = self.stem_relu.get_relu_density(mask)
        # _total, _relu = self.stem2[2].get_relu_density(mask)
        # total += _total
        # relu += _relu

        # for i in range(self.depth):
        #     _total, _relu = self.stages[i].get_relu_density(mask)
        #     total += _total
        #     relu += _relu

        # _total, _relu = self.cls_relu.get_relu_density(mask)
        # total += _total
        # relu += _relu

        total = 0
        relu = 0
        for layer in self.modules():
            if isinstance(layer, custom_relu):
                _total, _relu = layer.get_relu_density()
                total += _total
                relu += _relu

        if total == 0:
            return -1, -1
        else:
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



@register_model
def vanillanet_5_full_unify(custom_settings, num_classes, if_shortcut, keep_bn, **kwargs):
    model = VanillaNetFullUnify(custom_settings, num_classes, dims=[128*4, 256*4, 512*4, 1024*4], strides=[2,2,2], if_shortcut=if_shortcut, keep_bn=keep_bn, **kwargs)
    return model

@register_model
def vanillanet_6_full_unify(custom_settings, num_classes, if_shortcut, keep_bn, **kwargs):
    model = VanillaNetFullUnify(custom_settings, num_classes, dims=[128*4, 256*4, 512*4, 1024*4, 1024*4], strides=[2,2,2,1], if_shortcut=if_shortcut, keep_bn=keep_bn, **kwargs)
    return model

@register_model
def vanillanet_7_full_unify(custom_settings, num_classes, if_shortcut, keep_bn, **kwargs):
    model = VanillaNetFullUnify(custom_settings, num_classes, dims=[128*4, 128*4, 256*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,2,1], if_shortcut=if_shortcut, keep_bn=keep_bn, **kwargs)
    return model

# @register_model
# def vanillanet_8(pretrained=False, in_22k=False, **kwargs):
#     model = VanillaNet(dims=[128*4, 128*4, 256*4, 512*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,1,2,1], **kwargs)
#     return model

