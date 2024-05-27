import torch
model = torch.load("group_norm.pth")['model_state_dict']

print(model.keys())