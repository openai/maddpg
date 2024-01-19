#!/bin/bash

# Step 1: Train your models
# Replace this with your actual model training commands
/home/pearl0/miniconda3/envs/MMJC-maddpg/bin/python ./train_mujuco.py --config ./configs/ant_config_2.yaml --train True
/home/pearl0/miniconda3/envs/MMJC-maddpg/bin/python ./train_mujuco.py --config ./configs/ant_config_4.yaml --train True
/home/pearl0/miniconda3/envs/MMJC-maddpg/bin/python ./train_mujuco.py --config ./configs/cheetah_config_3.yaml --train True
/home/pearl0/miniconda3/envs/MMJC-maddpg/bin/python ./train_mujuco.py --config ./configs/cheetah_config_6.yaml --train True
/home/pearl0/miniconda3/envs/MMJC-maddpg/bin/python ./train_mujuco.py --config ./configs/humanoid_config.yaml --train True
/home/pearl0/miniconda3/envs/MMJC-maddpg/bin/python ./train_mujuco.py --config ./configs/humanoidstandup_config.yaml --train True

# Step 2: Add changes to Git
git add ./learning_curves/Ant.4x2.0.001.350.0.99/ ./tmp/policy/Ant.4x2.0.001.350.0.99/
git add ./learning_curves/Ant.2x4.0.001.350.0.99/ ./tmp/policy/Ant.2x4.0.001.350.0.99/
git add ./learning_curves/HalfCheetah.2x3.0.001.350.0.99/ ./tmp/policy/HalfCheetah.2x3.0.001.350.0.99/
git add ./learning_curves/HalfCheetah.6x1.0.001.350.0.99/ ./tmp/policy/HalfCheetah.6x1.0.001.350.0.99/
git add ./learning_curves/HumanoidStandup.9x8.0.001.350.0.99/ ./tmp/policy/HumanoidStandup.9x8.0.001.350.0.99/
git add ./learning_curves/Humanoid.9x8.0.001.350.0.99/ ./tmp/policy/Humanoid.9x8.0.001.350.0.99/

# Step 3: Commit the changes
# The date and time are included in the commit message for tracking
git commit -m "Updated models on $(date)"

# Step 4: Push to GitHub
git push origin UNITYxMaMuJuCo

# Replace 'main' with your branch name if different
sudo shutdown now
