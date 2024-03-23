from gymnasium_robotics import mamujoco_v1
import torch
import argparse
import yaml
import os
import shutil
import math
import time
import pickle
import random
from icecream import ic
import copy


def m_deepcopy(self, excluded_keys: list[str]):
    """similar to `copy.deepcopy`, but excludes copying the member variables in `excluded_keys`."""
    dct = self.__dict__.copy()
    for key in excluded_keys:
        del dct[key]
    # we avoid the normal init. I *think* unpickling does something like this too?
    other = type(self).__new__(type(self))
    other.__dict__ = copy.deepcopy(dct)
    return other

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Runs policy for X episodes and returns return reward
# A fixed seed is used for the eval environment
def eval_policy(env_name: str, conf: str, obsk: int, seed: int = 256, eval_episodes: int = 10) -> float:
    eval_env = mamujoco_v1.parallel_env(scenario=env_name, agent_conf=conf, agent_obsk=obsk, local_categories=[['qpos', 'qvel'], ['qpos']])

    total_return = 0
    for i in range(eval_episodes):
        cur_state_dict = eval_env.reset(seed=seed + i)[0]
        terminated, truncated = 0, 0
        while not (terminated or truncated):
            cur_state = [torch.tensor(local_state, dtype=torch.float32, device=TORCH_DEVICE) for local_state in cur_state_dict.values()]
            actions = model.query_actor(cur_state, add_noise=False)
            actions_dict_numpy = {eval_env.possible_agents[agent_id]: actions[agent_id].tolist() for agent_id in range(len(eval_env.possible_agents))}
            cur_state_dict, reward_dict, is_terminal_dict, is_truncated_dict, info_dict = eval_env.step(actions_dict_numpy)
            total_return += reward_dict['agent_0']
            terminated = is_terminal_dict['agent_0']
            truncated = is_truncated_dict['agent_0']

    return total_return / eval_episodes


def generate_model(model_name: str, load_erb: str | None = None, load_q: str | None = None, load_pi: str | None = None):
    match model_name:
        case 'TD3':
            model = MATD3.model(num_actions_spaces, num_observations_spaces, num_global_observation_space, min_action, max_action, config, torch_device=TORCH_DEVICE)
        case 'TD3-cc':
            model = MATD3_cc.model(num_actions_spaces, num_observations_spaces, num_global_observation_space, min_action, max_action, config, torch_device=TORCH_DEVICE)
        case _:
            assert False, 'invalid learning algorithm'

    if load_erb is not None:
        model.erb = pickle.load(open(load_erb, 'wb'))
    if load_q is not None:
        model.twin_critic.load_state_dict(torch.load(load_q + "_twin_critic"))
        model.twin_critic_optimizer.load_state_dict(torch.load(load_q + "_twin_critics_optimizer"))
        model.target_twin_critic.load_state_dict(torch.load(load_q + "_target_twin_critic"))
    if load_pi is not None:
        assert False, "load_PI not implemented"
        model.actors.load_state_dict(torch.load(load_pi + "_actor"))
        model.actor_optimizer.load_state_dict(torch.load(load_pi + "_actor_optimizer"))
        model.actors_target.load_state_dict(torch.load(load_pi + "_target_actor"))

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='ant_config_2.yaml')
    parser.add_argument("--starting_run", default=0, type=int)
    parser.add_argument("--final_run", default=int(1e6), type=int)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))

    if config['domain']['name'] == "Ant":
        env = mamujoco_v1.parallel_env(scenario=config['domain']['name'], agent_conf=config['domain']['factorization'], agent_obsk=config['domain']['obsk'], local_categories=[['qpos', 'qvel'], ['qpos']], include_cfrc_ext_in_observation=False)
    else:
        env = mamujoco_v1.parallel_env(scenario=config['domain']['name'], agent_conf=config['domain']['factorization'], agent_obsk=config['domain']['obsk'], local_categories=[['qpos', 'qvel'], ['qpos']])

    num_actions_spaces = [env.action_space(agent).shape[0] for agent in env.possible_agents]
    num_observations_spaces = [env.observation_space(agent).shape[0] for agent in env.possible_agents]
    num_global_observation_space = len(env.state())
    min_action = env.action_space(env.possible_agents[0]).low[0]
    max_action = env.action_space(env.possible_agents[0]).high[0]
    ic(num_actions_spaces)
    ic(num_observations_spaces)
    ic(num_global_observation_space)

    # create evaluate directory
    eval_path = 'results/' + 'MA' + config['domain']['algo'] + '_' + str(config['domain']['factorization']) + '_' + config['domain']['name'] + '_' + str(time.time())
    os.makedirs(eval_path)
    shutil.copyfile(args.config, eval_path + '/ant_config_2.yaml')

    for run in range(args.starting_run, min(config['domain']['runs'], args.final_run + 1)):
        # seed all the things
        torch.manual_seed(config['domain']['seed'] + run)
        [act_space.seed(config['domain']['seed'] + indx + run * 1000) for indx, act_space in enumerate(env.action_spaces.values())]
        random.seed(config['domain']['seed'] + run)

        # create model
        model = generate_model(config['domain']['algo'], config['other']['load_erb'], config['other']['load_Q'], config['other']['load_PI'])
        #model.twin_critics[0].load_state_dict(torch.load('best_run0_twin_critic_inv_d'))
        #model.target_twin_critics[0].load_state_dict(torch.load('best_run0_target_twin_critic_inv_d'))
        #model.target_actors[0].load_state_dict(torch.load('best_run0_target_actor_inv_d'))
        #model.actors[0].load_state_dict(torch.load('best_run0_actor_inv_d'))

        #model.twin_critics[0].load_state_dict(torch.load('best_run0_twin_critic_hopper'))
        #model.target_twin_critics[0].load_state_dict(torch.load('best_run0_twin_critic_hopper'))
        #model.target_actors[0].load_state_dict(torch.load('best_run0_actor_hopper'))
        #model.actors[0].load_state_dict(torch.load('best_run0_actor_hopper'))

        #model.actors[0].load_state_dict(torch.load('best_run0_actor_inv'))
        #model.twin_critics[0].load_state_dict(torch.load('best_run0_critic_inv'))

        # create evaluation file
        eval_file = open(eval_path + '/score' + str(run) + '.csv', 'w+')
        eval_max_return = -math.inf

        cur_state_dict = env.reset(seed=config['domain']['seed'] + run)[0]
        cur_state = [torch.tensor(state, dtype=torch.float32, device=TORCH_DEVICE) for state in cur_state_dict.values()]
        for step in range(config['domain']['total_timesteps']):
            cur_state_full = torch.tensor(env.state(), dtype=torch.float32, device=TORCH_DEVICE)
            # sample actions
            with torch.no_grad():
                if step >= config['domain']['init_learn_timestep']:
                    actions = model.query_actor(cur_state, add_noise=True)
                else:
                    actions = [torch.Tensor(act_space.sample()) for act_space in env.action_spaces.values()]
            actions_dict = {env.possible_agents[agent_id]: actions[agent_id].detach() for agent_id in range(len(env.possible_agents))}
            actions_dict_numpy = {env.possible_agents[agent_id]: actions[agent_id].tolist() for agent_id in range(len(env.possible_agents))}

            # step
            new_state_dict, reward_dict, is_terminal_dict, is_truncated_dict, info_dict = env.step(actions_dict_numpy)

            # store to ERB
            model.erb.add_experience(old_state=cur_state_full, actions=torch.tensor(env.map_local_actions_to_global_action(actions_dict_numpy), dtype=torch.float32, device=TORCH_DEVICE), reward=reward_dict[env.possible_agents[0]], new_state=torch.tensor(env.state(), dtype=torch.float32, device=TORCH_DEVICE), is_terminal=is_terminal_dict[env.possible_agents[0]])

            # update cur_state
            new_state = [torch.tensor(state, dtype=torch.float32, device=TORCH_DEVICE) for state in new_state_dict.values()]
            cur_state = new_state

            if step >= config['domain']['init_learn_timestep']:
                model.train_model_step(env)

            if is_terminal_dict[env.possible_agents[0]] or is_truncated_dict[env.possible_agents[0]]:
                cur_state_dict = env.reset()[0]
                cur_state = [torch.tensor(state, dtype=torch.float32, device=TORCH_DEVICE) for state in cur_state_dict.values()]

            # evaluate
            if step % config['domain']['evaluation_frequency'] == 0 and step >= config['domain']['init_learn_timestep']:  # evaluate episode
                total_evalution_return = eval_policy(config['domain']['name'], config['domain']['factorization'], config['domain']['obsk'])
                print('Run: ' + str(run) + ' Training Step: ' + str(step) + ' return: ' + str(total_evalution_return))
                eval_file.write(str(total_evalution_return) + '\n')
                if (eval_max_return < total_evalution_return):
                    eval_max_return = total_evalution_return
                    best_model = m_deepcopy(model, excluded_keys=['erb'])

        best_model.save(eval_path + '/' + 'best_run' + str(run))
        pickle.dump(model.erb, open(eval_path + '/' + 'best_run' + str(run) + '_erb', 'wb'))
        print('Run: ' + str(run) + ' Max return: ' + str(eval_max_return))
        print('Finished score can be found at: ' + eval_path + '/score' + str(run) + '.csv')

    env.close()