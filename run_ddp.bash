#!/bin/bash

# Receives a Python script as the first argument, and passes the remaining arguments to the Python script
python_script="$1"
shift  # Remove the first argument, the rest are arguments for the Python script
args="$@"

# Read IPs from ip_list file
ips=($(cat ip_list))  # Assuming ip_list is in the same directory as the script

# Define node information
master_ip="${ips[0]}"  # First IP is the master
slave_ips=("${ips[@]:1}")  # Remaining IPs are slaves
node_rank_begin=($(seq 0 8 $(((${#ips[@]} - 1) * 8))))  # Generate node rank begin array

# Calculate world size
world_size=$((${#ips[@]} * 8))

# Define project root directory
proj_root="/home/aiscuser/HE_transfer_learning"

# Run on the master node
echo "Running on master node (IP: $master_ip)..."
cd $proj_root
echo "python $python_script $args --master_ip $master_ip --world_size $world_size --node_rank_begin ${node_rank_begin[0]}"
python $python_script $args --master_ip $master_ip --world_size $world_size --node_rank_begin ${node_rank_begin[0]}

# Run on each slave node
for i in "${!slave_ips[@]}"; do
  ip=${slave_ips[$i]}
  rank_begin=${node_rank_begin[$(($i + 1))]}  # $i+1 because the master node is already 0
  echo "Running on slave node $((i+1)) (IP: $ip)..."
  ssh aiscuser@$ip "cd $proj_root; python $python_script $args --master_ip $master_ip --world_size $world_size --node_rank_begin $rank_begin"
done
