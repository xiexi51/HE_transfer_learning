import torch
import torch.nn as nn
import torch.nn.functional as F

image_a6000 = torch.load(f'./dump_a6000/input_image.pt')

stem1_conv = nn.Conv2d(3, 512, kernel_size=4, stride=4)

state_dict1 = torch.load(f'./dump_a6000/stem1_state_dict.pt', map_location='cpu')

stem1_conv.weight.data = state_dict1['0.weight']
stem1_conv.bias.data = state_dict1['0.bias']

stem1_conv = stem1_conv.cuda()

stem1_conv_out = stem1_conv(image_a6000)

stem1_conv_out_a6000 = torch.load(f'./dump_a6000/stem1_conv_out.pt')

mse = F.mse_loss(stem1_conv_out, stem1_conv_out_a6000)

print(mse.item())
