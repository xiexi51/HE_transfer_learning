#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import os
from model import general_relu_poly


class activation_poly(nn.Module):
    def __init__(self, dim, poly_weight_inits, poly_weight_factors, act_num=3):
        super().__init__()
        self.act_num = act_num
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num*2 + 1, act_num*2 + 1))
        self.bias = torch.nn.Parameter(torch.zeros(dim))

        self.relu = general_relu_poly(if_channel=True, if_pixel=True, weight_inits=poly_weight_inits, factors=poly_weight_factors, num_channels=dim)
        
        nn.init.trunc_normal_(self.weight, std=.02)

    def forward(self, x, mask):
        x = self.relu(x, mask)
        fm = x
        x = F.conv2d(x, self.weight, self.bias, padding=self.act_num, groups=self.dim)
        return x, fm
    
    def get_relu_density(self, mask):
        return self.relu.get_relu_density(mask)

class Block_poly(nn.Module):
    def __init__(self, dim, dim_out, poly_weight_inits, poly_weight_factors, act_num=3, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(stride)
        self.act = activation_poly(dim_out, poly_weight_inits, poly_weight_factors, act_num)
 
    def forward(self, x, mask):
        x = self.conv(x)
        x = self.pool(x)
        x, fm = self.act(x, mask)
        return x, fm
    
    def get_relu_density(self, mask):
        return self.act.get_relu_density(mask)
    

class VanillaNet_poly(nn.Module):
    def __init__(self, poly_weight_inits, poly_weight_factors, in_chans=3, num_classes=1000, dims=[96, 192, 384, 768], 
                 drop_rate=0, act_num=3, strides=[2,2,2,1]):
        super().__init__()
        stride, padding = 4, 0
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=stride, padding=padding),
            activation_poly(dims[0], poly_weight_inits, poly_weight_factors, act_num)
        )
        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            stage = Block_poly(dims[i], dims[i+1], poly_weight_inits, poly_weight_factors, act_num=act_num, stride=strides[i])
            self.stages.append(stage)
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(drop_rate),
            nn.Conv2d(dims[-1], num_classes, 1),
        )
        self.apply(self._init_weights)

        self.if_forward_with_fms = False

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x_and_mask):
        if self.if_forward_with_fms:
            fms = []
        x, mask = x_and_mask
        x = self.stem[0](x)
        x, fm = self.stem[1](x, mask)
        if self.if_forward_with_fms:
            fms.append(fm)

        for i in range(len(self.stages)):
            x, fm = self.stages[i](x, mask)
            if self.if_forward_with_fms:
                fms.append(fm)
        x = self.cls(x)
        x = x.view(x.size(0),-1)
        
        if self.if_forward_with_fms:
            return (x, fms)
        else:
            return x

    def get_relu_density(self, mask):
        total_elements, relu_elements = self.stem[1].get_relu_density(mask)
        for stage in self.stages:
            stage_total, stage_relu = stage.get_relu_density(mask)
            total_elements += stage_total
            relu_elements += stage_relu
        return total_elements, relu_elements


def vanillanet_5_poly(poly_weight_inits, poly_weight_factors):
    model = VanillaNet_poly(poly_weight_inits, poly_weight_factors, dims=[128*4, 256*4, 512*4, 1024*4], strides=[2,2,2])
    return model

def vanillanet_6_poly(poly_weight_inits, poly_weight_factors):
    model = VanillaNet_poly(poly_weight_inits, poly_weight_factors, dims=[128*4, 256*4, 512*4, 1024*4, 1024*4], strides=[2,2,2,1])
    return model
