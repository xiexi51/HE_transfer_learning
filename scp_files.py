import glob
import os
import subprocess

folder_pattern = './runs*'
file_patterns = ['acc.txt', 'args.txt', 'events.*']
target_dir = '/home/xix22010/py_projects/from_azure_0311'

folders = glob.glob(folder_pattern)

for folder in folders:
    print(f"Processing folder: {folder}")
    # Add options to suppress the warning
    subprocess.run(['ssh', '-o', 'ProxyJump=hop20001@137.99.0.102', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', 'xix22010@192.168.10.16', f'mkdir -p {target_dir}/{folder}'], check=True)
    
    for file_pattern in file_patterns:
        for file_path in glob.glob(os.path.join(folder, file_pattern)):
            # Add options to suppress the warning in the scp command as well
            scp_command = f'scp -o ProxyJump=hop20001@137.99.0.102 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {file_path} xix22010@192.168.10.16:{target_dir}/{folder}/'
            try:
                subprocess.run(scp_command, check=True, shell=True)
                print(f"Transferred {file_path} to {target_dir}/{folder}")
            except subprocess.CalledProcessError:
                print(f"Failed to transfer {file_path}.")
        if not glob.glob(os.path.join(folder, file_pattern)):
            print(f"No {file_pattern} in {folder}")
