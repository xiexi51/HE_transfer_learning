import os
import shutil

def get_folder_size(folder):
    """Calculate the total size of a folder in bytes."""
    total_size = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(root, file)
            total_size += os.path.getsize(filepath)
    return total_size

def delete_small_folders(n):
    """Delete folders starting with './runs' if their size is less than n kilobytes."""
    base_path = "."  # Base directory to search for folders
    threshold_size = n * 1024  # Convert kilobytes to bytes

    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if folder_name.startswith("runs") and os.path.isdir(folder_path):
            folder_size = get_folder_size(folder_path)
            if folder_size < threshold_size:
                print(f"Deleting {folder_path} as its size ({folder_size} bytes) is less than {threshold_size} bytes")
                shutil.rmtree(folder_path)
                print(f"Deleted {folder_path}")
            else:
                print(f"Keeping {folder_path} as its size ({folder_size} bytes) is greater than or equal to {threshold_size} bytes")

delete_small_folders(100)
