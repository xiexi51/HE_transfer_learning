import os
import shutil

# Define a function to get folder size
def get_folder_size(folder):
    total = 0
    for path, dirs, files in os.walk(folder):
        for f in files:
            fp = os.path.join(path, f)
            total += os.path.getsize(fp)
    return total / 1e6  # return size in MB

# Define a function to find and read the acc.txt file in a specific folder
def find_max_test_acc_in_folder(folder):
    acc_file_path = os.path.join(folder, 'acc.txt')
    if os.path.exists(acc_file_path):
        with open(acc_file_path, 'r') as file:
            max_test_acc = 0
            for line in file:
                parts = line.split()
                if len(parts) > 5:
                    try:
                        test_acc = float(parts[4])
                        if test_acc > max_test_acc:
                            max_test_acc = test_acc
                    except ValueError:
                        continue
            return max_test_acc
    return None

# Define a function to parse args.txt and extract the needed fields
def parse_args_file(folder):
    args_path = os.path.join(folder, 'args.txt')
    args_dict = {}
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    args_dict[key] = value
    return (
        args_dict.get("v_type", "N/A"),
        args_dict.get("ln_k", "N/A"),
        args_dict.get("ln_mu", "N/A"),
        args_dict.get("relu_dropout", "N/A")
    )

folders_info = []
# Traverse all folders in the current directory, find folders starting with 'runs'
for folder in os.listdir('.'):
    if os.path.isdir(folder) and folder.startswith('runs'):
        max_test_acc = find_max_test_acc_in_folder(folder)
        if max_test_acc is not None:
            size = get_folder_size(folder)
            v_type, ln_k, ln_mu, relu_dropout = parse_args_file(folder)
            folders_info.append((folder, max_test_acc, size, v_type, ln_k, ln_mu, relu_dropout))

# Sort the folders by name
folders_info.sort()

# Print the folders info
for folder, max_test_acc, size, v_type, ln_k, ln_mu, relu_dropout in folders_info:
    print(f"{folder}: acc {max_test_acc}, size {size:.2f}MB, v_type={v_type}, ln_k={ln_k}, ln_mu={ln_mu}, relu_dropout={relu_dropout}")

# Ask the user if they want to remove folders
remove_size = 2
remove_acc = 50

response = input(f"Do you want to remove folders with max_test_acc < {remove_acc} and size < {remove_size}MB? (yes/no) ")
if response.lower() == 'yes':
    for folder, max_test_acc, size, *_ in folders_info:
        if max_test_acc < remove_acc and size < remove_size:
            shutil.rmtree(folder)
            print(f"Folder {folder} removed.")
