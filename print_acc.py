import os
import shutil

base_dir = "../from_azure"

# Define a function to find and read the acc.txt file in a specific folder
def find_max_test_acc_in_folder(folder):
    acc_file_path = os.path.join(folder, 'acc.txt')
    if os.path.exists(acc_file_path):
        with open(acc_file_path, 'r') as file:
            max_test_acc = 0
            # Read the file content line by line, find the maximum test accuracy
            for line in file:
                parts = line.split()
                if len(parts) > 5:  # Ensure there is enough data in the line
                    try:
                        # The 5th token is test accuracy
                        test_acc = float(parts[4])
                        if test_acc > max_test_acc:
                            max_test_acc = test_acc
                    except ValueError:
                        continue  # If the conversion fails, skip this line
            return max_test_acc
    return None  # If the file does not exist or no valid test accuracy is found

results = []

# Traverse all folders in the base directory, find folders starting with 'runs'
for folder in os.listdir(base_dir):
    if os.path.isdir(os.path.join(base_dir, folder)) and folder.startswith('runs'):
        max_test_acc = find_max_test_acc_in_folder(os.path.join(base_dir, folder))
        if max_test_acc is not None:
            results.append((folder, max_test_acc))

# Sort the results by max_test_acc from high to low
results.sort(key=lambda x: x[1], reverse=True)

# Print the sorted results
for folder, max_test_acc in results:
    print(f"{folder}: {max_test_acc}")

# Ask the user if they want to delete folders with max_test_acc < 10
delete = input("Do you want to delete folders with max_test_acc < 10? (yes/no): ")
if delete.lower() == "yes":
    for folder, max_test_acc in results:
        if max_test_acc < 10:
            shutil.rmtree(os.path.join(base_dir, folder))
            print(f"Deleted {folder}")