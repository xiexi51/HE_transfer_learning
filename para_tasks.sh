#!/bin/bash

visible_gpus=("0" "1" "2" "3" "4" "5" "6" "7" )
parallelism_per_gpu=3
total_parallelism=$(( parallelism_per_gpu * ${#visible_gpus[@]} ))

run_task_script="./run_tasks.sh"

# Call the run_tasks.sh script with only_get_max_id=True to get the total number of tasks (max_id)
total_tasks=$(bash $run_task_script "0" "0" "0" "True" "0" "runs_onlygetmaxid")

echo "Total tasks: $total_tasks"

if [ $(( total_tasks % total_parallelism )) -eq 0 ]; then
  tasks_per_parallel=$(( total_tasks / total_parallelism ))
else
  tasks_per_parallel=$(( total_tasks / total_parallelism + 1 ))
fi

# Create the log_root directory with the format 'runs_YYYY-MM-DD_HH:MM:SS'
current_time=$(date +"%Y-%m-%d_%H:%M:%S")
log_root="runs_$current_time"
mkdir -p $log_root

parallel_args=""
for i in $(seq 0 $((total_parallelism - 1))); do
  gpu_id=${visible_gpus[$(( i % ${#visible_gpus[@]} ))]}
  start_task_id=$(( i * tasks_per_parallel ))
  end_task_id=$(( start_task_id + tasks_per_parallel ))
  parallel_args+="$gpu_id $start_task_id $end_task_id"$'\n'
done

# Add the log_root parameter to the parallel command
echo -e "$parallel_args" | parallel --ungroup --colsep ' ' bash $run_task_script {1} {2} {3} "False" $total_tasks $log_root
