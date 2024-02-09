import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import pickle
import os
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from datetime import datetime

current_time = datetime.now()

# Format the date and time in the format you prefer, e.g., 'YYYYMMDD-HHMMSS'
directory_name_with_time = current_time.strftime('%Y%m%d-%H%M%S')


import os
from datetime import datetime

# Replace 'base_directory_path' with the base path where your directories are located

# Define the date and time format that your directories are using
# This should match the format used when creating the directories
date_time_format = '%Y%m%d-%H%M%S'
def get_directories(base_path):
    """Get a list of all directories in the base path."""
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
def filter_directories_by_date(directories, date_format):
    """Filter out directories that match the date and time pattern."""
    filtered_directories = []
    for directory in directories:
        try:
            # If the directory name can be parsed into a datetime object, it matches the pattern
            datetime.strptime(directory, date_format)
            filtered_directories.append(directory)
        except ValueError:
            # If a ValueError is raised, it means the directory name doesn't match the pattern
            continue
    return filtered_directories
def find_most_recent_directory(base_path,directories, date_format):
    """Find the most recent directory based on the date and time pattern."""
    if not directories:
        return None
    # Parse the directory names to get the corresponding datetime objects
    dates = [datetime.strptime(directory, date_format) for directory in directories]
    # Get the most recent date
    most_recent_date = max(dates)
    # Find the directory that corresponds to the most recent date
    most_recent_directory = directories[dates.index(most_recent_date)]
    return os.path.join(base_path, most_recent_directory, most_recent_directory)
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", default = 'simple', type=str, help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--fixed-agent", default=False)
    parser.add_argument("--fixed-landmark", default=False)
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--location", type=float, default=0.95, help="discount factor")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    known_args, _ = parser.parse_known_args()

    # Now we can use the scenario in setting default values
    scenario = known_args.scenario if known_args.scenario else "default_scenario"
    maxep = known_args.max_episode_len if known_args.max_episode_len else "25"
    lr = known_args.lr if known_args.lr else "1e-2"
    FA = "FA" if known_args.fixed_agent == 'True' else "NFA"
    FL = "FL" if known_args.fixed_landmark == 'True' else "NFL"
    numunits = known_args.num_units if known_args.num_units else "128"
    gamma = known_args.gamma if known_args.gamma else "0.95"

    base_directory_path = f"./tmp/policy/{scenario}.{maxep}.{lr}.{numunits}.{gamma}.{FA}.{FL}"
    if not os.path.exists(base_directory_path):
        os.makedirs(base_directory_path)
    directories = get_directories(base_directory_path)
    date_directories = filter_directories_by_date(directories, date_time_format)
    most_recent_directory = find_most_recent_directory(base_directory_path, date_directories, date_time_format)
    if most_recent_directory is None:
        print("No previous directories found")
        most_recent_directory = ""

    plot_directory_path = f"./learning_curves/{scenario}.{maxep}.{lr}.{numunits}.{gamma}.{FA}.{FL}"
    if not os.path.exists(plot_directory_path):
        os.makedirs(plot_directory_path)
    parser.add_argument("--num-episodes", type=int, default=15000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='test', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default=base_directory_path, help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=500, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default=most_recent_directory, help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)


    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark_files", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default=plot_directory_path, help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        validation_success = []
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()
        print('obs_n', obs_n)
        print('obs_shape_n', obs_shape_n)
        print('env.action_space', env.action_space)
        print('n_agents', env.n)
        print('Starting iterations...')
        while True:
            # get action
            # print(obs_n[0].shape, obs_n.shape)
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                if len(episode_rewards) > arglist.num_episodes - 1000:
                    if np.sum(np.square(env.world.agents[0].state.p_pos - env.world.landmarks[0].state.p_pos)) < 0.1:
                        validation_success.append(1)
                    else:
                        validation_success.append(0)

                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                full_directory_path = os.path.join(arglist.save_dir, directory_name_with_time)
                if not os.path.exists(full_directory_path):
                    os.makedirs(full_directory_path)  # Create the directory since it does not exist
                U.save_state(os.path.join(full_directory_path, directory_name_with_time), saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                full_directory_path = os.path.join(arglist.plots_dir, directory_name_with_time)
                if not os.path.exists(full_directory_path):
                    os.makedirs(full_directory_path)  # Create the directory since it does not exist

                rew_file_name = os.path.join(full_directory_path, arglist.exp_name + '_rewards.pkl')
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = os.path.join(full_directory_path, arglist.exp_name + '_agrewards.pkl')
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                validation_success_file_name = os.path.join(full_directory_path, arglist.exp_name + '_validation_success.pkl')
                with open(validation_success_file_name, 'wb') as fp:
                    pickle.dump(validation_success, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
