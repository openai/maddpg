#!/bin/bash


# Function to train a model with a given configuration
bash ./Ant_mal.sh
bash ./Cheetah.sh

# Training models with different configurations
#train_model ./configs/ant_config_2.yaml ./train_mujuco.py ./learning_curves/Ant.4x2.0.001.350.0.99/ ./tmp/policy/Ant.4x2.0.001.350.0.99/
#train_model ./configs/ant_config_4.yaml ./train_mujuco.py ./learning_curves/Ant.2x4.0.001.350.0.99/ ./tmp/policy/Ant.2x4.0.001.350.0.99/
#
## Train malfunction
#train_model ./configs/ant_config_2.yaml ./train_mujuco_malfunction.py ./learning_curves/Ant.4x2.0.001.350.0.99/malfunction/ ./tmp/policy/Ant.4x2.0.001.350.0.99malfunction/
#train_model ./configs/ant_config_4.yaml ./train_mujuco_malfunction.py ./learning_curves/Ant.2x4.0.001.350.0.99/malfunction/ ./tmp/policy/Ant.2x4.0.001.350.0.99malfunction/
#
#
#train_model ./configs/cheetah_config_3.yaml ./train_mujuco.py ./learning_curves/HalfCheetah.2x3.0.001.350.0.99/ ./tmp/policy/HalfCheetah.2x3.0.001.350.0.99/
#train_model ./configs/cheetah_config_6.yaml ./train_mujuco.py ./learning_curves/HalfCheetah.6x1.0.001.350.0.99/ ./tmp/policy/HalfCheetah.6x1.0.001.350.0.99/
#train_model ./configs/humanoid_config.yaml ./train_mujuco.py ./learning_curves/HumanoidStandup.9x8.0.001.350.0.99/ ./tmp/policy/HumanoidStandup.9x8.0.001.350.0.99/
#train_model ./configs/humanoidstandup_config.yaml ./train_mujuco.py ./learning_curves/Humanoid.9x8.0.001.350.0.99/ ./tmp/policy/Humanoid.9x8.0.001.350.0.99/

# Shutdown the system
#sudo shutdown -h now