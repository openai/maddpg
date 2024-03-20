import argparse
import numpy as np
import tensorflow.compat.v1 as tf
# import tensorflow as tf
tf.disable_v2_behavior()
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
# import tensorflow.contrib.layers as layers
import tensorflow.keras.layers as layers
from datetime import datetime

import yaml
import os
import shutil
import math
import time
import pickle
import random
import gymnasium_robotics
current_time = datetime.now()

# Format the date and time in the format you prefer, e.g., 'YYYYMMDD-HHMMSS'
directory_name_with_time = current_time.strftime('%Y%m%d-%H%M%S')


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
    # return most_recent_directory
    return os.path.join(base_path, most_recent_directory, most_recent_directory)


def parse_args_n_config():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--config", default='ant_config_2.yaml')
    parser.add_argument("--starting_run", default=0, type=int)
    parser.add_argument("--final_run", default=int(1e6), type=int)
    parser.add_argument("--train", default=True, type=bool)
    # Environment
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--num-units", type=int, default=350, help="number of units in the mlp")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--partition", type=str, default="2x4", help="agent configuration file")

    parser.add_argument("--num-episodes", type=int, default=30000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")

    # Core training parameters
    parser.add_argument("--batch-size", type=int, default=100, help="number of episodes to optimize at the same time")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="buffer size")
    parser.add_argument("--mal_agent", type=int, default=0, help="malfunctioning agent")
    #Checkpointing
    # parser.add_argument("--save-rate", type=int, default=1000,
    #                     help="save model once every time this many episodes are completed")
    # parser.add_argument("--benchmark", action="store_true", default=False)
    # parser.add_argument("--benchmark-iters", type=int, default=100000)
    # # Evaluation
    # parser.add_argument("--restore", action="store_true", default=False)
    # parser.add_argument("--display", action="store_true", default=False)



    known_args, _ = parser.parse_known_args()
    config = yaml.safe_load(open(known_args.config, 'r'))

    # Now we can use the scenario in setting default values
    scenario = config['domain']['name']
    if config['domain']['factorization'] == '9|8':
        adjugate = '9x8'
    else:
        adjugate = config['domain']['factorization']
    lr = known_args.lr if known_args.lr else "1e-2"
    numunits = known_args.num_units if known_args.num_units else "128"
    gamma = known_args.gamma if known_args.gamma else "0.95"

    base_directory_path = f"./tmp/policy/{scenario}.{adjugate}.{lr}.{numunits}.{gamma}"
    if not os.path.exists(base_directory_path):
        os.makedirs(base_directory_path)
    directories = get_directories(base_directory_path)
    date_directories = filter_directories_by_date(directories, date_time_format)
    most_recent_directory = find_most_recent_directory(base_directory_path, date_directories, date_time_format)

    if most_recent_directory is None:
        print("No previous directories found")
        most_recent_directory = ""

    plot_directory_path = f"./learning_curves/{scenario}.{adjugate}.{lr}.{numunits}.{gamma}/"
    load_dir = f"./tmp/policy/{scenario}.{adjugate}.{lr}.{numunits}.{gamma}/"

    # print("Base directory path: ", base_directory_path)
    # print("Most recent directory: ", most_recent_directory)
    # print("Plot directory path: ", plot_directory_path)
    # print("Load directory: ", load_dir)
    # print(config['maddpg']['save_dir'])
    # print(config['maddpg']['load_dir'])
    if not os.path.exists(plot_directory_path):
        os.makedirs(plot_directory_path)
    # Checkpointing
    # parser.add_argument("--exp-name", type=str, default='test', help="name of the experiment")
    # parser.add_argument("--save-dir", type=str, default=base_directory_path,
    #                     help="directory in which training state and model should be saved")
    # parser.add_argument("--load-dir", type=str, default=most_recent_directory,
    #                     help="directory in which training state and model are loaded")
    # parser.add_argument("--benchmark_files", type=str, default="./benchmark_files/",
    #                     help="directory where benchmark data is saved")
    # parser.add_argument("--plots-dir", type=str, default=plot_directory_path,
    #                     help="directory where plot data is saved")
    args = parser.parse_args()
    # if (args.restore or args.display or args.benchmark) and args.load_dir == "":
    #     args.load_dir = load_dir

    if (config['maddpg']['restore'] or config['maddpg']['display'] or config['maddpg']['benchmark']) or config['maddpg']['load_dir'] == "":
        config['maddpg']['load_dir'] = most_recent_directory
    config['maddpg']['save_dir'] = base_directory_path
    config['maddpg']['plots_dir'] = plot_directory_path
    return args, config # return both args and config

def mlp_model_actor(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        # Use tf.compat.v1.layers or tf.layers for fully_connected layers
        out = tf.compat.v1.layers.dense(out, units=num_units, activation=tf.nn.tanh)
        out = tf.compat.v1.layers.dense(out, units=num_units, activation=tf.nn.tanh)
        out = tf.compat.v1.layers.dense(out, units=num_outputs, activation=tf.nn.tanh)
        return out
def mlp_model_critic(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        # Use tf.compat.v1.layers or tf.layers for fully_connected layers
        out = tf.compat.v1.layers.dense(out, units=num_units, activation=tf.nn.relu)
        out = tf.compat.v1.layers.dense(out, units=num_units, activation=tf.nn.relu)
        out = tf.compat.v1.layers.dense(out, units=num_outputs, activation=None)
        return out

def make_env(arglist, config, show=False):
    if config['domain']['name'] == 'Ant':
        env = gymnasium_robotics.mamujoco_v0.parallel_env(scenario=config['domain']['name'], agent_conf=config['domain']['factorization'],healthy_reward=0.1,
                                 max_episode_steps=config['domain']['max_episode_len'],
                                       agent_obsk=config['domain']['obsk'], terminate_when_unhealthy=False, use_contact_forces = False)
    else:
        env = gymnasium_robotics.mamujoco_v0.parallel_env(scenario=config['domain']['name'], agent_conf=config['domain']['factorization'],
                                       agent_obsk=config['domain']['obsk'])
    return env

def get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, config, arglist):
    trainers = []
    actor_model = mlp_model_actor
    critic_model = mlp_model_critic
    trainer = MADDPGAgentTrainer
    # for i in range(num_adversaries):
    #     trainers.append(trainer(
    #         "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
    #         local_q_func=(config['maddpg']['adv_policy']=='ddpg')))
    for i in range(num_adversaries, len(env.possible_agents)):
        trainers.append(trainer(
            "agent_%d" % i, actor_model, critic_model, obs_shape_n, action_shape_n, i, arglist,
            local_q_func=(config['maddpg']['good_policy']=='ddpg')))
    return trainers


def train(arglist, config):
    all_ep_runs = []
    all_ag_runs = []
    all_time_steps = []
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist, config, show=False)
        # Create agent trainers
        n_agents = len(env.possible_agents)
        actions_spaces = [env.action_space(agent) for agent in env.possible_agents]

        observations_spaces = [env.observation_space(agent).shape for agent in env.possible_agents]
        # print("Observation space: ", observations_spaces)
        # print("Action space: ",         actions_spaces)
        trainers = get_trainers(env, 0, observations_spaces, actions_spaces, config, arglist)
        print('Using good policy {} and adv policy {}'.format(config['maddpg']['good_policy'], config['maddpg']['adv_policy']))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if config['maddpg']['load_dir'] == "":
            config['maddpg']['load_dir'] = config['maddpg']['save_dir']
        if config['maddpg']['display'] or config['maddpg']['restore'] or config['maddpg']['benchmark']:
            print('Loading previous state...')
            print(config['maddpg']['load_dir'])
            U.load_state(config['maddpg']['load_dir'])

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(n_agents)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        time_steps = []
        validation_success = []
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()

        cur_state_dict, xypos = env.reset()
        cur_state = [np.array(state, dtype=np.float32) for state in cur_state_dict.values()]
        episode_step = 0
        train_step = 0
        t_start = time.time()
        t_total = time.time()
        tot_steps = 0
        malfunction = False

        print(str(config['domain']['name']))
        print('Starting iterations...')

        while True:
            # cur_state_full = torch.tensor(env.state(), dtype=torch.float32, device=TORCH_DEVICE)
            # cur_state_full = np.array(env.state(), dtype=np.float32)
            # get action
            # print("N-agents: ", n_agents)
            # print("cur_state: ", cur_state[0])
            # print("cur_state_dict: ", cur_state_dict)
            # print("cur_state_keys: ", cur_state_dict.keys())
            # print("cur_position: ", xypos)
            # print("cur_state_values: ", cur_state_dict.values())
            # print(len(cur_state), len(cur_state_dict.values()))
            # print(cur_state[0].shape, cur_state_full.shape, env.state().shape)

            actions = [agent.action(obs) for agent, obs in zip(trainers,cur_state)]

            if malfunction:
                actions[mal_agent] = np.zeros_like(actions[mal_agent])

            # environment step
            actions_dict = {env.possible_agents[agent_id]: actions[agent_id] for agent_id in
                            range(len(env.possible_agents))}
            actions_dict_numpy = {env.possible_agents[agent_id]: actions[agent_id].tolist() for agent_id in
                                  range(len(env.possible_agents))}

            # step
            # new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            new_state_dict, reward_dict, is_terminal_dict, is_truncated_dict, xypos = env.step(actions_dict_numpy)
            next_state = [np.array(state, dtype=np.float32) for state in new_state_dict.values()]

            terminal = (episode_step >= arglist.max_episode_len)

            # store to ERB
            # a = np.array(env.map_local_actions_to_global_action(actions_dict_numpy))
            # print(actions_dict, a)
            for i, agent in enumerate(trainers):
                agent.experience(cur_state[i], actions_dict[agent.name], reward_dict[agent.name],  next_state[i], is_terminal_dict[agent.name], terminal)
            # model.erb.add_experience(old_state=cur_state_full,
            #                          actions=torch.tensor(env.map_local_actions_to_global_action(actions_dict_numpy),
            #                                               dtype=torch.float32, device=TORCH_DEVICE),
            #                          reward=reward_dict[env.possible_agents[0]],
            #                          new_state=torch.tensor(env.state(), dtype=torch.float32, device=TORCH_DEVICE),
            #                          is_terminal=is_terminal_dict[env.possible_agents[0]])

            # update cur_state
            # obs_n = new_obs_n
            # new_state = [torch.tensor(state, dtype=torch.float32, device=TORCH_DEVICE) for state in
            #              new_state_dict.values()]
            new_state = [np.array(state, dtype=np.float32) for state in new_state_dict.values()]
            cur_state = new_state

            done = all(is_terminal_dict.values()) or all(is_truncated_dict.values())

            # collect experience
            for i, rew in enumerate(reward_dict.values()):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            # increment global step counter
            train_step += 1

            if done or terminal:
                final_ep_rewards.append(episode_rewards[-1])
                time_steps.append(train_step)
                cur_state_dict = env.reset()[0]
                cur_state = [np.array(state, dtype=np.float32) for state in cur_state_dict.values()]
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # Malfunction
            if len(episode_rewards) == config['domain']['malfunction_episode']:
                malfunction = True
                mal_agent = arglist.mal_agent

            # for benchmarking learned policies
            # if arglist.benchmark:
            #     for i, info in enumerate(info_n):
            #         agent_info[-1][i].append(info_n['n'])
            #     if train_step > arglist.benchmark_iters and (done or terminal):
            #         file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
            #         print('Finished benchmarking, now saving...')
            #         with open(file_name, 'wb') as fp:
            #             pickle.dump(agent_info[:-1], fp)
            #         break
            #     continue

            # for displaying learned policies
            # if arglist.train and len(episode_rewards) % config['maddpg']['save_rate'] == 0 and config['maddpg']['display']:
            #     env = make_env(arglist, config, True)
            #     cur_state_dict = env.reset()[0]
            #     cur_state = [np.array(state, dtype=np.float32) for state in cur_state_dict.values()]
            # if arglist.train and len(episode_rewards) % config['maddpg']['save_rate'] + 50 == 0 and config['maddpg']['display']:
            #     env = make_env(arglist, config, False)
            #
            # if config['maddpg']['display']:
            #     time.sleep(0.1)
            #     env.render()
            #     continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)


            # save model, display training output
            if (done or terminal) and (len(episode_rewards) % config['maddpg']['save_rate'] == 0):
                full_directory_path = os.path.join(config['maddpg']['save_dir'] + 'malfunction', directory_name_with_time)
                # print(full_directory_path)
                if not os.path.exists(full_directory_path):
                    os.makedirs(full_directory_path)  # Create the directory since it does not exist
                U.save_state(os.path.join(full_directory_path, directory_name_with_time), saver=saver)
                # print statement depends on whether or not there are adversaries
                # if num_adversaries == 0:
                #     print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                #         train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                # else:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-config['maddpg']['save_rate']:]),
                    [np.mean(rew[-config['maddpg']['save_rate']:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                # final_ep_rewards.append(np.mean(episode_rewards[-config['maddpg']['save_rate']:]))
                # for rew in agent_rewards:
                #     final_ep_ag_rewards.append(np.mean(rew[-config['maddpg']['save_rate']:]))
                # time_steps.append(train_step)


            # saves final episode reward for plotting training curve later
            # if config['domain']['total_timesteps'] < train_step:

            if len(episode_rewards) > config['domain']['num_episodes']:
                full_directory_path = os.path.join(config['maddpg']['plots_dir'] + 'malfunction',
                                                   directory_name_with_time)
                # print(full_directory_path)
                if not os.path.exists(full_directory_path):
                    os.makedirs(full_directory_path)  # Create the directory since it does not exist
                rew_file_name = os.path.join(full_directory_path, config['maddpg']['exp_name'] + '_' + str(arglist.mal_agent) + 'rewards.pkl')
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = os.path.join(full_directory_path, config['maddpg']['exp_name'] + '_' + str(arglist.mal_agent) + '_agrewards.pkl')
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                agrew_file_name = os.path.join(full_directory_path, config['maddpg']['exp_name'] + '_' + str(arglist.mal_agent) + '_timesteps.pkl')
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(time_steps, fp)
                # validation_success_file_name = os.path.join(full_directory_path,
                #                                             config['maddpg']['exp_name'] + '_validation_success.pkl')
                # with open(validation_success_file_name, 'wb') as fp:
                #     pickle.dump(validation_success, fp)
                print('...Finished total of {} episodes. Time: {}'.format(len(episode_rewards), time.time() - t_total))
                # tf.reset_default_graph()
                break
            cur_state = next_state


if __name__ == '__main__':
    arglist, config = parse_args_n_config()
    train(arglist, config)
