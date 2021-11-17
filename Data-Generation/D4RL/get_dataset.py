import gym
import d4rl 

# Create the environment
env = gym.make('halfcheetah-medium-v1') # list of supported envs -> https://github.com/rail-berkeley/d4rl/wiki/Tasks

# Automatically download and return the dataset
dataset = env.get_dataset() # downloaded in hdf5 format to /home/username/.d4rl/datasets/halfcheetah_medium-v1.hdf5
# print(dataset['observations']) # An (N, dim_observation)-dimensional numpy array of observations
# print(dataset['actions']) # An (N, dim_action)-dimensional numpy array of actions
# print(dataset['rewards']) # An (N,)-dimensional numpy array of rewards