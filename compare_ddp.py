import torch

def calculate_mse(tensor1, tensor2):
    # 确保两个张量在同一设备上
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
    mse = torch.nn.functional.mse_loss(tensor1, tensor2)
    return mse.item()

def calculate_average_mse(list1, list2):
    # 确保列表中的所有张量都在同一设备上
    mse_values = [calculate_mse(t1, t2.to(t1.device)) for t1, t2 in zip(list1, list2)]
    return sum(mse_values) / len(mse_values)

# 对于每种文件类型，计算origin与0和1之间的MSE
mse_results = {}
# files = ['fms_s', 'fms_t', 'out_s', 'out_t', 'gradients', 'weights']
files = ['gradients']
for file in files:
    for i in [0, 1]:
        file1 = f'origin_2_{file}.pt'
        file2 = f'{i}_2_{file}.pt'

        tensor1 = torch.load(file1)
        tensor2 = torch.load(file2)

        if file in ['out_s', 'out_t']:
            mse = calculate_mse(tensor1, tensor2)
        elif file in ['fms_s', 'fms_t']:
            mse = calculate_average_mse(tensor1, tensor2)
        else: # 对于gradients
            mse_values = [calculate_mse(tensor1[key], tensor2[key].to(tensor1[key].device)) for key in tensor1 if key in tensor2]
            mse = sum(mse_values) / len(mse_values)

        mse_results[f'origin vs {i} {file}'] = mse

# 打印结果
for key, value in mse_results.items():
    print(f'{key}: {value}')
