import os
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def get_directories(base_path):
    """Get a list of all directories in the base path."""
    return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

def filter_and_sort_directories_by_date(directories, date_format):
    """Filter out directories that match the date and time pattern and sort them."""
    filtered_and_sorted_directories = []
    for directory in directories:
        try:
            # Parse the directory name into a datetime object
            date = datetime.strptime(directory, date_format)
            filtered_and_sorted_directories.append((directory, date))
        except ValueError:
            continue
    # Sort directories by date
    filtered_and_sorted_directories.sort(key=lambda x: x[1])
    return [directory for directory, date in filtered_and_sorted_directories]

def get_rewards_for_last_n_runs(base_path, n, date_format='%Y%m%d-%H%M%S', aggrew=True, valid=True, time=False):
    """Get the rewards for the last n runs."""
    directories = get_directories(base_path)
    date_directories = filter_and_sort_directories_by_date(directories, date_format)

    # Select the last n directories
    recent_n_directories = date_directories[-n:]

    all_rewards = []
    for directory in recent_n_directories:
        full_path = os.path.join(base_path, directory)
        rewards_data = []

        with open(os.path.join(full_path, 'test_rewards.pkl'), 'rb') as f:
            rewards = pickle.load(f)
            rewards_data.append(rewards)

        if aggrew:
            with open(os.path.join(full_path, 'test_agrewards.pkl'), 'rb') as f:
                agrewards = pickle.load(f)
                rewards_data.append(agrewards)
        else:
            rewards_data.append(None)

        # if valid:
        #     with open(os.path.join(full_path, 'test_validation_success.pkl'), 'rb') as f:
        #         valid = pickle.load(f)
        #         rewards_data.append(valid)
        # else:
        #     rewards_data.append(None)

        if time:
            with open(os.path.join(full_path, 'test_timesteps.pkl'), 'rb') as f:
                time = pickle.load(f)
                rewards_data.append(time)
        else:
            rewards_data.append(None)

        all_rewards.append(tuple(rewards_data))

    return all_rewards

def calculate_mean_and_confidence_interval(data):
    """
    Calculate the mean and 95% confidence interval for each timestep.

    :param data: A list of lists, where each inner list represents a run and contains values for each timestep.
    :return: A tuple of two numpy arrays - one for the mean and one for the 95% confidence interval.
    """
    data = np.array(data)
    mean = np.mean(data, axis=0)
    stderr = stats.sem(data, axis=0, nan_policy='omit')
    confidence_interval = stderr * stats.t.ppf((1 + 0.95) / 2., len(data) - 1)

    return mean, confidence_interval


def average_and_confidence(run_info):
    """
    Calculate the average and 95% confidence interval for rewards at each timestep.

    :param output of get_rewards_for_last_n_runs
    :return: A tuple of four numpy arrays - mean rewards, confidence interval for rewards, mean timesteps, confidence interval for timesteps.
    """
    rewards = []
    timesteps = []
    for rewards_data, agrewards_data, time_data in run_info:
        rewards.append(rewards_data)
        # timesteps.append(time_data)
    mean_rewards, conf_rewards = calculate_mean_and_confidence_interval(rewards)
    # mean_timesteps, conf_timesteps = calculate_mean_and_confidence_interval(timesteps)

    return mean_rewards, conf_rewards #, mean_timesteps, conf_timesteps



def plot_with_confidence_interval(mean_values, confidence_interval, timesteps, title="Plot with Confidence Interval",
                                  xlabel="Timestep", ylabel="Value"):
    """
    Plot mean values with confidence interval using matplotlib.

    :param mean_values: Array of mean values.
    :param confidence_interval: Array of confidence interval values.
    :param timesteps: Array of timesteps.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    """

    upper_bound = mean_values + confidence_interval
    lower_bound = mean_values - confidence_interval

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_values, label="Mean", color="blue")
    plt.fill_between(timesteps, lower_bound, upper_bound, color="blue", alpha=0.2, label="95% Confidence Interval")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_multiple_with_confidence_intervals(mean_values_list, confidence_intervals_list, timesteps, labels, title="Comparison Plot", xlabel="Timestep", ylabel="Value", save=False, save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots'):
    """
    Plot multiple sets of mean values with their confidence intervals.

    :param mean_values_list: List of arrays of mean values for each algorithm.
    :param confidence_intervals_list: List of arrays of confidence intervals for each algorithm.
    :param timesteps: Array of timesteps.
    :param labels: List of labels for each algorithm.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    """
    plt.figure(figsize=(10, 6))

    for mean_values, confidence_interval, label in zip(mean_values_list, confidence_intervals_list, labels):
        upper_bound = mean_values + confidence_interval
        lower_bound = mean_values - confidence_interval

        plt.plot(timesteps, mean_values, label=f"Mean - {label}")
        plt.fill_between(timesteps, lower_bound, upper_bound, alpha=0.2, label=f"95% CI - {label}")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save:
        plt.savefig(os.path.join(save_path, title+'.png'))
    plt.show()

def plot_trajectories(trajectories, title="Agent Trajectories", xlabel="X Position", ylabel="Y Position", save=False, save_path='/Users/Hunter/Development/Academic/UML/RL/Hasenfus-RL/Multi-Agent/maddpg/experiments/plots'):
    """
    Plot a series of x, y pairs as trajectories on an xy-plane.

    :param trajectories: A list of trajectories, where each trajectory is a list of (x, y) pairs.
    """
    plt.figure(figsize=(8, 6))

    for i, traj in enumerate(trajectories):
        # Assuming each trajectory is a list of (x, y) tuples
        x_coords, y_coords = zip(*traj)
        plt.plot(x_coords, y_coords, 'r-')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if save:
        plt.savefig(os.path.join(save_path, title+'.png'))
    plt.show()

def get_trajectories(base_path, date_format='%Y%m%d-%H%M%S'):
    """Get the rewards for the last n runs."""
    directories = get_directories(base_path)
    date_directories = filter_and_sort_directories_by_date(directories, date_format)

    # Select the last n directories
    recent_directory = date_directories[-1]

    full_path = os.path.join(base_path, recent_directory)
    # print(full_path)
    directories2 = get_directories(full_path)
    # print(directories2)
    date_directories2 = filter_and_sort_directories_by_date(directories2, date_format)
    # print(date_directories2)
    recent_directory2 = date_directories2[-1]

    full_path2 = os.path.join(full_path, recent_directory2)

    with open(os.path.join(full_path2, 'test_test_trajectories.pkl'), 'rb') as f:
        trajectories = pickle.load(f)

    return trajectories
