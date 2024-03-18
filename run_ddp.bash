#!/bin/bash

# Receive the Python script as the first argument, with the remaining arguments passed to the Python script
python_script="$1"
shift  # Remove the first argument, leaving the rest for the Python script
args="$@"

# Define node information
master_ip="10.9.32.40"
slave_ips=("10.9.32.236")  # Add more slave node IPs here if necessary
node_rank_begin=(0 8)  # Starting node rank for each node

# Calculate world size
num_nodes=$((${#slave_ips[@]} + 1))  # Add 1 to include the master node
world_size=$(($num_nodes * 8))

# Define the project root directory and screen session name
proj_root="/home/aiscuser/HE_transfer_learning"
screen_name="my_screen_session"

# Execute on the master node
echo "Running on master node (IP: $master_ip)..."
cd $proj_root
screen -S $screen_name -dm bash -c "python $python_script $args --master_ip $master_ip --world_size $world_size --node_rank_begin ${node_rank_begin[0]}"

# Execute on each slave node
for i in "${!slave_ips[@]}"; do
  ip=${slave_ips[$i]}
  rank_begin=${node_rank_begin[$(($i + 1))]}  # $i+1 because the master node is already 0
  echo "Running on slave node $((i+1)) (IP: $ip)..."
  ssh aiscuser@$ip "cd $proj_root; screen -S $screen_name -dm bash -c 'python $python_script $args --master_ip $master_ip --world_size $world_size --node_rank_begin $rank_begin'"
done
