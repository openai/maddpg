import os
from datetime import datetime
import pickle
# Replace 'base_directory_path' with the base path where your directories are located

# Define the date and time format that your directories are using
# This should match the format used when creating the directories

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
    return os.path.join(base_path, most_recent_directory)
def recent_rewards(base_path, aggrew=True, valid=True, time=False):
    date_format = '%Y%m%d-%H%M%S'
    """Get the rewards from the most recent directory."""
    # Get the directories
    directories = get_directories(base_path)
    # Filter out the directories that don't match the date and time pattern
    date_directories = filter_directories_by_date(directories, date_format)
    # Find the most recent directory
    most_recent_directory = find_most_recent_directory(base_path, date_directories, date_format)
    # Load the data from .pkl files
    returnable = []
    with open(os.path.join(most_recent_directory, 'test_rewards.pkl'), 'rb') as f:
        rewards = pickle.load(f)
        returnable.append(rewards)
    if aggrew:
        with open(os.path.join(most_recent_directory, 'test_agrewards.pkl'), 'rb') as f:
            agrewards = pickle.load(f)
            returnable.append(agrewards)
    else:
        returnable.append(None)
    if valid:
        with open(os.path.join(most_recent_directory, 'test_validation_success.pkl'), 'rb') as f:
            valid = pickle.load(f)
            returnable.append(valid)
    else:
        returnable.append(None)
    if time:
        with open(os.path.join(most_recent_directory, 'test_timesteps.pkl'), 'rb') as f:
            time = pickle.load(f)
            returnable.append(time)

    return tuple(returnable)
