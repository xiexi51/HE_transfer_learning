import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init

# 设置文件存放路径
dump_path = './dump_a6000'

# 定义 activation 类
class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.act_num = act_num
        self.deploy = deploy
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num*2 + 1, act_num*2 + 1))
        if deploy:
            self.bias = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        weight_init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        x = super(activation, self).forward(x)
        x = F.conv2d(x, self.weight, self.bias, padding=self.act_num, groups=self.dim)
        return self.bn(x)


# self.stem1 = nn.Sequential(
#                 nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=stride, padding=padding),
#                 nn.BatchNorm2d(dims[0], eps=1e-6),
#             )
class Stem1(nn.Module):
    def __init__(self, dim):
        super(Stem1, self).__init__()
        self.conv = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x    


# 定义模型的某个部分
class Stem2(nn.Module):
    def __init__(self, dim, act_num=3):
        super(Stem2, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        self.activation = activation(dim, act_num)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# 加载状态字典
state_dict1 = torch.load(f'./dump_a6000/stem1_state_dict.pt', map_location='cpu')

state_dict2 = torch.load(f'{dump_path}/stem2_state_dict.pt', map_location='cpu')



# 提取dims[0]的值
dims = [state_dict2['0.weight'].size(0)]


stem1 = Stem1(512)

stem1.conv.weight.data = state_dict1['0.weight']
stem1.conv.bias.data = state_dict1['0.bias']
stem1.bn.weight.data = state_dict1['1.weight']
stem1.bn.bias.data = state_dict1['1.bias']
stem1.bn.running_mean.data = state_dict1['1.running_mean']
stem1.bn.running_var.data = state_dict1['1.running_var']


stem1 = stem1.cuda()

stem1.bn.eval()

# 实例化模型部分
stem2 = Stem2(dims[0], act_num=3)

# 更新模型权重
stem2.conv.weight.data = state_dict2['0.weight']
stem2.conv.bias.data = state_dict2['0.bias']
stem2.bn.weight.data = state_dict2['1.weight']
stem2.bn.bias.data = state_dict2['1.bias']
stem2.bn.running_mean.data = state_dict2['1.running_mean']
stem2.bn.running_var.data = state_dict2['1.running_var']
# 对于自定义激活层，我们同样更新权重和BN参数
stem2.activation.weight.data = state_dict2['2.weight']
stem2.activation.bn.weight.data = state_dict2['2.bn.weight']
stem2.activation.bn.bias.data = state_dict2['2.bn.bias']
stem2.activation.bn.running_mean.data = state_dict2['2.bn.running_mean']
stem2.activation.bn.running_var.data = state_dict2['2.bn.running_var']

stem2 = stem2.cuda()

stem2.bn.eval()
stem2.activation.bn.eval()

input_image = torch.load(f'{dump_path}/input_image.pt')

stem1_out = stem1(input_image)

# save_stem1_out = torch.load(f'./dump/stem1_out.pt')

save_stem1_out_a6000 = torch.load(f'./dump_a6000/stem1_out.pt')


stem1_conv_out = stem1.conv(input_image)

save_stem1_conv_out_a6000 = torch.load(f'./dump_a6000/stem1_conv_out.pt')


mse_stem1_conv_out = F.mse_loss(stem1_conv_out, save_stem1_conv_out_a6000)

print("mse_stem1_conv_out", mse_stem1_conv_out.item())

exit()

mse_stem1_out = F.mse_loss(stem1_out, save_stem1_out_a6000)

print("mse_stem1_out", mse_stem1_out.item())


# 加载输入
input_to_stem2 = torch.load(f'./dump/input_to_stem2.pt')

input_to_stem2_a6000 = torch.load(f'./dump_a6000/input_to_stem2.pt')

mse_stem2_input = F.mse_loss(stem1_out, input_to_stem2_a6000)

print("mse_stem2_input", mse_stem2_input.item())

mse = F.mse_loss(save_stem1_out_a6000, input_to_stem2_a6000)
print("mse", mse.item())

exit()

# 前向传播以获取输出
output_from_stem2 = stem2(input_to_stem2)

# 加载保存的输出
saved_output = torch.load(f'{dump_path}/output_from_stem2.pt')

# 计算输出和保存的输出之间的MSE
mse = F.mse_loss(output_from_stem2, saved_output)

print(mse.item())

saved_output_a6000 = torch.load('./dump_a6000/output_from_stem2.pt')

mse = F.mse_loss(output_from_stem2, saved_output_a6000)

print(mse.item())
