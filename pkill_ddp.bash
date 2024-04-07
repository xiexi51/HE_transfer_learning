#!/bin/bash

# Read the ip_list file into an array
mapfile -t ip_list < ./ip_list

# Get the local machine's IP address; this uses `hostname -I` and takes the first IP
# Note: You may need to adjust this command based on your network setup
my_ip=$(hostname -I | awk '{print $1}')

# Check if the local IP matches the master IP
master_ip=${ip_list[0]}
if [ "$my_ip" != "$master_ip" ]; then
    echo "Local IP does not match the master IP!"
    exit 1
else
    echo "Killing master ($master_ip)"
    pkill ddp > /dev/null 2>&1
fi

# Use SSH to execute `pkill ddp` on each of the slave nodes starting from the second IP
for ((i = 1; i < ${#ip_list[@]}; i++)); do
    slave_ip=${ip_list[$i]}
    echo "Killing slave ($slave_ip)"
    ssh $slave_ip "pkill ddp" > /dev/null 2>&1 &
done
