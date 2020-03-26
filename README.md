**Status:** Archive (code is provided as-is, no updates expected)

# Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

This is the code for implementing the MADDPG algorithm presented in the paper:
[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).
It is configured to be run in conjunction with environments from the
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).
Note: this codebase has been restructured since the original paper, and the results may
vary from those reported in the paper.

**Update:** the original implementation for policy ensemble and policy estimation can be found [here](https://www.dropbox.com/s/jlc6dtxo580lpl2/maddpg_ensemble_and_approx_code.zip?dl=0). The code is provided as-is. 

## Installation

- To install, `cd` into the root directory and type `pip install -e .`

- Known dependencies: Python (3.5.4), OpenAI gym (0.10.5), tensorflow (1.8.0), numpy (1.14.5)

## Case study: Multi-Agent Particle Environments

We demonstrate here how the code can be used in conjunction with the
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).

- Download and install the MPE code [here](https://github.com/openai/multiagent-particle-envs)
by following the `README`.

- Ensure that `multiagent-particle-envs` has been added to your `PYTHONPATH` (e.g. in `~/.bashrc` or `~/.bash_profile`).

- To run the code, `cd` into the `experiments` directory and run `train.py`:

``python train.py --scenario simple``

- You can replace `simple` with any environment in the MPE you'd like to run.

## Command-line options

### Environment options

- `--scenario`: defines which environment in the MPE is to be used (default: `"simple"`)

- `--max-episode-len` maximum length of each episode for the environment (default: `25`)

- `--num-episodes` total number of training episodes (default: `60000`)

- `--num-adversaries`: number of adversaries in the environment (default: `0`)

- `--good-policy`: algorithm used for the 'good' (non adversary) policies in the environment
(default: `"maddpg"`; options: {`"maddpg"`, `"ddpg"`})

- `--adv-policy`: algorithm used for the adversary policies in the environment
(default: `"maddpg"`; options: {`"maddpg"`, `"ddpg"`})

### Core training parameters

- `--lr`: learning rate (default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

### Checkpointing

- `--exp-name`: name of the experiment, used as the file name to save all results (default: `None`)

- `--save-dir`: directory where intermediate training results and model will be saved (default: `"/tmp/policy/"`)

- `--save-rate`: model is saved every time this number of episodes has been completed (default: `1000`)

- `--load-dir`: directory where training state and model are loaded from (default: `""`)

### Evaluation

- `--restore`: restores previous training state stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), and continues training (default: `False`)

- `--display`: displays to the screen the trained policy stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), but does not continue training (default: `False`)

- `--benchmark`: runs benchmarking evaluations on saved policy, saves results to `benchmark-dir` folder (default: `False`)

- `--benchmark-iters`: number of iterations to run benchmarking for (default: `100000`)

- `--benchmark-dir`: directory where benchmarking data is saved (default: `"./benchmark_files/"`)

- `--plots-dir`: directory where training curves are saved (default: `"./learning_curves/"`)

## Code structure

- `./experiments/train.py`: contains code for training MADDPG on the MPE

- `./maddpg/trainer/maddpg.py`: core code for the MADDPG algorithm

- `./maddpg/trainer/replay_buffer.py`: replay buffer code for MADDPG

- `./maddpg/common/distributions.py`: useful distributions used in `maddpg.py`

- `./maddpg/common/tf_util.py`: useful tensorflow functions used in `maddpg.py`



## Paper citation

If you used this code for your experiments or found it helpful, consider citing the following paper:

<pre>
@article{lowe2017multi,
  title={Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Neural Information Processing Systems (NIPS)},
  year={2017}
}
</pre>
