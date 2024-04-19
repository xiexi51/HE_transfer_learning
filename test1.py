import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设 x 和 conv1 已经被加载
x = torch.load("x.pth")
conv1 = torch.load("conv1.pth")

# 应用卷积层
out2 = F.conv2d(x, conv1.weight, conv1.bias, conv1.stride, conv1.padding, conv1.dilation, conv1.groups)
print("Output shape:", out2.shape)

accumulated_output = torch.zeros_like(out2)

max_max = -9999
min_min = 9999
max_mean = -9999
min_mean = 9999
max_std = -9999

for i in range(x.shape[1]):
    input_channel = x[:, i:i+1, :, :] 
    output_from_channel = F.conv2d(input_channel, conv1.weight[:, i:i+1, :, :], conv1.bias, conv1.stride, conv1.padding, conv1.dilation, conv1.groups)
    max = output_from_channel.max().item()
    min = output_from_channel.min().item()
    mean = output_from_channel.mean().item()
    std = output_from_channel.std().item()
    if max > max_max:
        max_max = max
    if min < min_min:
        min_min = min
    if mean > max_mean:
        max_mean = mean
    if mean < min_mean:
        min_mean = mean
    if std > max_std:
        max_std = std

print(f"{max_max} {min_min} {max_mean} {min_mean} {max_std}")

diff = accumulated_output - out2

if torch.allclose(accumulated_output, out2, atol=1e-1):
    print("The accumulated output matches the direct output.")
else:
    print("There is a difference between the accumulated and direct outputs.")