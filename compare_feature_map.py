import os
import torch
from torch.nn.functional import mse_loss
import math

def compare_files(dir1, dir2, filenames):
    comparisons = {}
    
    for filename in filenames:
        file1 = torch.load(os.path.join(dir1, filename))
        file2 = torch.load(os.path.join(dir2, filename))
        
        if file1.shape != file2.shape:
            print(f"Warning: {filename} has different shapes in {dir1} and {dir2}")
            continue

        diff = file1 - file2
        euclidean_distance = torch.norm(diff).item()
        mse = mse_loss(file1, file2).item()
        
        comparisons[filename] = {
            "difference": diff,
            "euclidean_distance": euclidean_distance,
            "mse": mse
        }
    
    return comparisons

dir1 = "./feature_map_70"
dir2 = "./feature_map_70_mask_0"
filenames = [
    "x_before_conv1.pt", "conv1_weight.pt", "x_after_conv1.pt", "bn1_activation.pt",
    "relu1.pt", "maxpool1.pt", "layer1_0.pt", "layer1_1.pt", "layer2_0.pt", "layer2_1.pt",
    "layer3_0.pt", "layer3_1.pt", "layer4_0.pt", "layer4_1.pt", "avg_pool2d.pt", "out_view.pt", "linear.pt"
]

comparisons = compare_files(dir1, dir2, filenames)

for filename, comparison in comparisons.items():
    print(f"{filename}:")
    print(f"  Euclidean Distance: {comparison['euclidean_distance']:.6f}")
    print(f"  MSE: {comparison['mse']:.6f}")
    print()
