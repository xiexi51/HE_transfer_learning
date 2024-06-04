#!/bin/bash

# Read ip_list file
ip_list=$(cat ip_list)
master_ip=$(echo "$ip_list" | head -n 1)
slave_ips=$(echo "$ip_list" | tail -n +2)

# Get local IP address
local_ip=$(hostname -I | awk '{print $1}')

# Directory array
directories=(
    "runs20240604101603"
)

# Check if local IP is equal to master IP
if [ "$local_ip" == "$master_ip" ]; then
    # Execute commands on master
    for dir in "${directories[@]}"; do
        mkdir -p "/home/aiscuser/HE_transfer_learning/$dir"

        scp -o ProxyJump=xix22010@137.99.0.102 xix22010@192.168.10.16:/home/xix22010/py_projects/from_azure_0604/$dir/* /home/aiscuser/HE_transfer_learning/$dir
    done

    # Transfer files from master to each slave
    for slave_ip in $slave_ips; do
        for dir in "${directories[@]}"; do
            ssh aiscuser@$slave_ip "mkdir -p /home/aiscuser/HE_transfer_learning/$dir"
            scp -r /home/aiscuser/HE_transfer_learning/$dir/* aiscuser@$slave_ip:/home/aiscuser/HE_transfer_learning/$dir
        done
    done
else
    echo "The local IP is not the master IP ($master_ip)."
fi
