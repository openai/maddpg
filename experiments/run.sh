#!/bin/bash

# Step 1: Train your models
# Replace this with your actual model training commands
/Users/Hunter/opt/anaconda3/envs/MaMJC-maddpg/bin/python /Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/train_mujuco.py --config ant_config_2.yaml --train True

# Step 2: Add changes to Git
git add ./experiments/learning_curves/Ant.4x2.0.001.350.0.99/ ./experiments/tmp/policy/Ant.4x2.0.001.350.0.99/

# Step 3: Commit the changes
# The date and time are included in the commit message for tracking
git commit -m "Updated models on $(date)"

# Step 4: Push to GitHub
git push origin main

# Replace 'main' with your branch name if different
