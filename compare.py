import torch

def compare_tensors(file1, file2):
    tensor1 = torch.load(file1)
    tensor2 = torch.load(file2)
    difference = torch.abs(tensor1 - tensor2)
    max_difference = torch.max(difference)
    mean_difference = torch.mean(difference)
    return max_difference.item(), mean_difference.item()

files_to_compare = [
    ("pre_x_before_conv1.pt", "x_before_conv1.pt"),
    ("pre_conv1_weight.pt", "conv1_weight.pt"),
    ("pre_x_after_conv1.pt", "x_after_conv1.pt"),
    ("pre_bn1.pt", "bn1_activation.pt"),
    ("pre_relu.pt", "relu1.pt"),
    ("pre_maxpool.pt", "maxpool1.pt"),
    ("pre_layer1.pt", "layer1_1.pt"),
    ("pre_layer2.pt", "layer2_1.pt"),
    ("pre_layer3.pt", "layer3_1.pt"),
    ("pre_layer4.pt", "layer4_1.pt"),
    ("pre_avgpool.pt", "avg_pool2d.pt"),
    ("pre_flatten.pt", "out_view.pt"),
    ("pre_fc.pt", "linear.pt")
]

for file1, file2 in files_to_compare:
    max_diff, mean_diff = compare_tensors(file1, file2)
    print(f"Comparing {file1} and {file2}:")
    print(f"  Max difference: {max_diff}")
    print(f"  Mean difference: {mean_diff}")
