#!/bin/bash

# Define project root directory
proj_root="/home/aiscuser/HE_transfer_learning"
cd $proj_root

# Check if local git repo is behind the remote and ask to pull changes if necessary
git fetch origin
local_status=$(git status -uno)  # Use -uno to ignore untracked files

if [[ $local_status == *"Your branch is behind"* ]]; then
  echo "Local repository is behind the remote."
  read -p "Do you want to perform 'git pull'? [y/N]: " response
  case "$response" in
    [yY][eE][sS]|[yY])
      git pull
      ;;
    *)
      echo "Skipping 'git pull'."
      ;;
  esac
else
  echo "Local repository is up-to-date."
fi

# Receives a Python script as the first argument, and passes the remaining arguments to the Python script
python_script="$1"
shift  # Remove the first argument, the rest are arguments for the Python script
args="$@"

# Default master port
master_port=6105

# Check if --master_port is in the arguments and update master_port if present
custom_port_set=false
for arg in $args; do
  if [[ "$arg" == "--master_port" ]]; then
    custom_port_set=true
    continue
  fi
  if $custom_port_set; then
    master_port=$arg  # Set the master port to the argument following --master_port
    break
  fi
done

# If --master_port was not set, add default master port to arguments
if [ -z "$custom_port_set" ]; then
  args="$args --master_port $master_port"
fi

log_root="runs$(date '+%Y%m%d%H%M%S')"

# Read IPs from ip_list file
ips=($(cat ip_list))  # Assuming ip_list is in the same directory as the script

# Define node information
master_ip="${ips[0]}"  # First IP is the master
slave_ips=("${ips[@]:1}")  # Remaining IPs are slaves
node_rank_begin=($(seq 0 8 $(((${#ips[@]} - 1) * 8))))  # Generate node rank begin array

# Calculate world size
world_size=$((${#ips[@]} * 8))

# Function to check if port is available
function is_port_available() {
    (echo "" >/dev/tcp/localhost/$1) &>/dev/null
    if [ $? -eq 0 ]; then
        return 1  # Port is in use, return false
    else
        return 0  # Port is not in use, return true
    fi
}

echo "testing master port from $master_port"
# Find an available master port
while ! is_port_available $master_port; do
  master_port=$((master_port+1))
done
echo "Master port found: $master_port"

# Update args with the found master port
args=$(echo $args | sed -E "s/--master_port [0-9]+/--master_port $master_port/")

# Run on the master node
echo "Running on master node (IP: $master_ip)..."
cd $proj_root
master_command="python $python_script $args --log_root $log_root --master_ip $master_ip --world_size $world_size --node_rank_begin ${node_rank_begin[0]}"
echo $master_command  # Echo the command for the master node
$master_command &  # Execute the Python command in the background

# Run on each slave node
for i in "${!slave_ips[@]}"; do
  ip=${slave_ips[$i]}
  rank_begin=${node_rank_begin[$(($i + 1))]}  # $i+1 because the master node is already 0
  echo "Running on slave node $((i+1)) (IP: $ip)..."

  # Prepare commands for updating project directory and copying Python files
  update_cmd="cd $proj_root && git pull"
  copy_cmd="echo 'Copying Python files and ip_list from master...' && scp $master_ip:$proj_root/*.py $proj_root/ && scp $master_ip:$proj_root/ip_list $proj_root/"

  # Command to run Python script on slave node, executed in the background
  slave_command="$update_cmd && $copy_cmd && python $python_script $args --log_root $log_root --master_ip $master_ip --world_size $world_size --node_rank_begin $rank_begin &"

  # Execute the commands on the slave node
  ssh aiscuser@$ip "$slave_command"  # Execute the commands on the slave node
done

# Optionally, wait for all background processes to finish
wait
