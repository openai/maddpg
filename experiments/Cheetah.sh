#!/bin/bash
# Function to train a model with a given configuration
commit_changes () {
    learning_curve_path=$1
    policy_path=$2
    git add $learning_curve_path $policy_path
    git commit -m "Updated models on $(date)"
}


train_model () {
    config_path=$1
    script=$2
    data=$3
    model=$4
    agent=$5

    for i in {1..5}
    do
      echo "Training run $i for configuration: $config_path"
      /home/pearl0/miniconda3/envs/MMJC-maddpg/bin/python $script --config $config_path --train True --mal_agent $agent
    done
    commit_changes $data $model
    git push origin UNITYxMaMuJuCo
}

# Training models with different configurations
train_model ./configs/cheetah_config_6.yaml ./Training/train_mujuco.py ./learning_curves/HalfCheetah.6x1.0.001.350.0.99/ ./tmp/policy/HalfCheetah.6x1.0.001.350.0.99/ 0

# Train malfunction
train_model ./configs/cheetah_config_6.yaml ./Training/train_mujuco_malfunction.py ./learning_curves/HalfCheetah.6x1.0.001.350.0.99/malfunction/ ./tmp/policy/HalfCheetah.6x1.0.001.350.0.99malfunction/ 0
train_model ./configs/cheetah_config_6.yaml ./Training/train_mujuco_malfunction.py ./learning_curves/HalfCheetah.6x1.0.001.350.0.99/malfunction/ ./tmp/policy/HalfCheetah.6x1.0.001.350.0.99malfunction/ 1
train_model ./configs/cheetah_config_6.yaml ./Training/train_mujuco_malfunction.py ./learning_curves/HalfCheetah.6x1.0.001.350.0.99/malfunction/ ./tmp/policy/HalfCheetah.6x1.0.001.350.0.99malfunction/ 2
train_model ./configs/cheetah_config_6.yaml ./Training/train_mujuco_malfunction.py ./learning_curves/HalfCheetah.6x1.0.001.350.0.99/malfunction/ ./tmp/policy/HalfCheetah.6x1.0.001.350.0.99malfunction/ 3
train_model ./configs/cheetah_config_6.yaml ./Training/train_mujuco_malfunction.py ./learning_curves/HalfCheetah.6x1.0.001.350.0.99/malfunction/ ./tmp/policy/HalfCheetah.6x1.0.001.350.0.99malfunction/ 4
train_model ./configs/cheetah_config_6.yaml ./Training/train_mujuco_malfunction.py ./learning_curves/HalfCheetah.6x1.0.001.350.0.99/malfunction/ ./tmp/policy/HalfCheetah.6x1.0.001.350.0.99malfunction/ 5

