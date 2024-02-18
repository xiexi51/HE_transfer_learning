#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import os

class activation(nn.Module):
    def __init__(self, dim, act_num=3):
        super(activation, self).__init__()
        self.act_num = act_num
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num*2 + 1, act_num*2 + 1))
        self.bias = torch.nn.Parameter(torch.zeros(dim))
        
        nn.init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        x = F.relu(x)
        x = F.conv2d(x, self.weight, self.bias, padding=self.act_num, groups=self.dim)
        return x    

class Block(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(stride)
        self.act = activation(dim_out, act_num)
 
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.act(x)
        return x
    

class VanillaNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, dims=[96, 192, 384, 768], 
                 drop_rate=0, act_num=3, strides=[2,2,2,1], **kwargs):
        super().__init__()
        stride, padding = (4, 0)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=stride, padding=padding),
            activation(dims[0], act_num)
        )
        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            stage = Block(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i])
            self.stages.append(stage)
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(drop_rate),
            nn.Conv2d(dims[-1], num_classes, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
        x = self.cls(x)        
        return x.view(x.size(0),-1)


def vanillanet_5(**kwargs):
    model = VanillaNet(dims=[128*4, 256*4, 512*4, 1024*4], strides=[2,2,2], **kwargs)
    return model

def vanillanet_6(**kwargs):
    model = VanillaNet(dims=[128*4, 256*4, 512*4, 1024*4, 1024*4], strides=[2,2,2,1], **kwargs)
    return model

