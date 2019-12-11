#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios
import maddpg.common.tf_util as U
from experiments.train import get_trainers

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    parser.add_argument('-l', '--load_dir', help='Session load directory')
    parser.add_argument('-ap', '--adv_policy', default='ddpg')
    parser.add_argument('-gp', '--good_policy', default='ddpg')
    parser.add_argument('--lr', default=0.0000000001)
    # parser.add_argument('')
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")


    args = parser.parse_args()

    with U.single_threaded_session():

        # load scenario from script
        scenario = scenarios.load(args.scenario).Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = True)
        # render call to create viewer window (necessary only for interactive policies)
        env.render_whole_field()
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = 7
        trainers = get_trainers(env, num_adversaries, obs_shape_n, args)
        # create interactive policies for each agent
        # trainers = []
        # policies = [InteractivePolicy(env,i) for i in range(env.n)]
        # execution loop

        # load session
        saver = U.load_state(args.load_dir)
        # TODO: Pick the latest?
        # TODO: Do I need to make agents???
        # So now the session hosted by U.single_threaded_session SHOULD be loaded?

        obs_n = env.reset()
        while True:
            # query for action from each agent's policy
            # act_n = []
            # for i, policy in enumerate(policies):
            #     act_n.append(policy.action(obs_n[i]))

            act_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            # new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            # step environment
            obs_n, reward_n, done_n, _ = env.step(act_n)
            # render all agent views
            env.render_whole_field()
            # display rewards
            #for agent in env.world.agents:
            #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
