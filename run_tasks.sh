#!/bin/bash

visible_gpu=$1
start_id=$2
end_id=$3
only_get_max_id=$4
max_id=$5
log_root=$6

counter=0
script_name="transfer_new.py"
batch_size_train=500
batch_size_test=500
hidden_dims=("0" "300")
dropout_rates=("0" "0.5")
add_bias1_vals=("True" "False")
add_bias2_vals=("True" "False")
add_relu_vals=("True" "False")
unfreeze_modes=("none" "b2_conv2" "b2")
data_augment_vals=("True" "False")
num_workers=1
pbar="False"
epoch=5

if [[ $only_get_max_id != "True" && ($start_id -ge $max_id) ]]; then
  echo "No valid task ID"
  exit 1
fi

if [[ $only_get_max_id != "True" && ($end_id -gt $max_id) ]]; then
  end_id=$max_id
fi

execute_task() {
  echo "Task ID: $counter GPU: $visible_gpu Parameters: --batch_size_train $batch_size_train --batch_size_test $batch_size_test --hidden_dim $hidden_dim --drop $drop --add_bias1 $add_bias1 --add_bias2 $add_bias2 --add_relu $add_relu --unfreeze_mode $unfreeze_mode --data_augment $data_augment --num_workers $num_workers --pbar $pbar --epoch $epoch --log_root $log_root"
  CUDA_VISIBLE_DEVICES=$visible_gpu python $script_name --id $counter --batch_size_train $batch_size_train --batch_size_test $batch_size_test --hidden_dim $hidden_dim --drop $drop --add_bias1 $add_bias1 --add_bias2 $add_bias2 --add_relu $add_relu --unfreeze_mode $unfreeze_mode --data_augment $data_augment --num_workers $num_workers --pbar $pbar --epoch $epoch --log_root $log_root
}

for hidden_dim in "${hidden_dims[@]}"; do
  for drop in "${dropout_rates[@]}"; do
    for add_bias1 in "${add_bias1_vals[@]}"; do
      for add_bias2 in "${add_bias2_vals[@]}"; do
        for add_relu in "${add_relu_vals[@]}"; do
          for unfreeze_mode in "${unfreeze_modes[@]}"; do
            for data_augment in "${data_augment_vals[@]}"; do
              if [[ $unfreeze_mode == "none" && ($add_bias1 == "True" || $add_bias2 == "True") ]]; then
                continue
              fi
              if [[ $unfreeze_mode == "b2_conv2" && ($add_bias1 == "True") ]]; then
                continue
              fi
              if [[ $only_get_max_id != "True" && ($counter -ge $start_id && $counter -lt $end_id) ]]; then
                execute_task
              fi
              ((counter++))
            done
          done
        done
      done
    done
  done
done

if [[ $only_get_max_id == "True" ]]; then
  echo "$((counter))"
fi
