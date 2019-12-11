log="logs/log."$(date +%s)

python experiments/train.py --scenario simple_passrush --max-episode-len \
600 --display --num-adversaries 7 --num-episodes 60000 --exp-name experiment_one \
--save-dir ./tmp/policy_9/17000 --save-rate 250
