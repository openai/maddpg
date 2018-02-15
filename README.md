# Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

This is the code for implementing the MADDPG algorithm presented in the paper:
[Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/pdf/1706.02275.pdf).
Note: this codebase has been restructured since the original paper, and the results may
vary slightly from those reported in the paper.

## Installation

- To install, `cd` into the root directory and type `pip install -e .`

- Known dependencies: OpenAI gym, tensorflow, numpy

## Case Study: Multi-Agent Particle Environments

We demonstrate here how the code can be used in conjunction with the
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).

- Download and install the MPE code [here](https://github.com/openai/multiagent-particle-envs)
by following the `README`.`

- Ensure that `multiagent-particle-envs` has been added to your `PYTHONPATH` (e.g. in `~/.bashrc` or `~/.bash_profile`).

- To run the code, `cd` into the `experiments` directory and run `train.py`:

``python train.py --scenario simple.py``

- You can replace `simple.py` with any environment in the MPE you'd like to run.
