import argparse
import gym
import h5py
import numpy as np
import os

from stable_baselines3 import PPO, A2C

"""
example usage:
python3 cartpole_datagen.py --env Acrobot-v1 --episodes 20 --max_steps 100 --timesteps 20_000
"""

# parse args
parser = argparse.ArgumentParser(description="Process input.")
parser.add_argument("--env", help="Environment name", default="CartPole-v1")
parser.add_argument(
    "--timesteps", help="Number of steps for model to learn", type=int, default=25_000
)
parser.add_argument("--episodes", help="Number of episodes", type=int, default=100)
parser.add_argument(
    "--max_steps", help="Number of steps in each episode", type=int, default=1000
)
args = parser.parse_args()

# TODO:
# 1. setup environment and dataloader in SPIRL to verify if the generated data works
# 2. add parameters to modify model.predict to have noise (random action with some probability) and adversarial actions (tbd)
# 3. modularize code to work with other openAI Gym environments and use other stable_baselines3 RL algorithms


def save_data(actions, states, images, dones, episode_num):
    print("Saving data")
    # following how authors of SPIRL save their data
    # https://github.com/kpertsch/d4rl/blob/master/scripts/generate_randMaze2d_datasets.py#L108
    # hardcode path for now
    env_name = (args.env).split("-")[0].lower()
    if not os.path.exists(f"./data/{env_name}"):
        os.makedirs(f"./data/{env_name}")
    data_path = f"./data/{env_name}/{env_name}_a2c_data_{episode_num}.h5"

    f = h5py.File(data_path, "w")
    f.create_dataset("traj_per_file", data=1)

    # cast everything to an np array
    actions = np.array(actions, dtype=np.float32)
    states = np.array(states, dtype=np.float32)
    images = np.array(images, dtype=np.float32)
    dones = np.array(dones, dtype=np.bool_)

    # save the data into a data group
    traj_data = f.create_group("traj0")
    traj_data.create_dataset("states", data=states)
    traj_data.create_dataset("images", data=images, dtype=np.uint8)
    traj_data.create_dataset("actions", data=actions)

    if np.sum(dones) == 0:
        dones[-1] = True

    # build pad-mask that indicates how long sequence is
    is_terminal_idxs = np.nonzero(dones)[0]
    pad_mask = np.zeros((len(dones),))
    pad_mask[: is_terminal_idxs[0]] = 1.0
    traj_data.create_dataset("pad_mask", data=pad_mask)

    f.close()

    return


if __name__ == "__main__":
    # create the environment
    env = gym.make(args.env)
    env.reset()

    # instantiate the learning algorithm for the cartpole environment and learn
    # according to this paper: https://arxiv.org/pdf/1810.01940.pdf
    # actor critic models like A2C work well on cartpole
    # value function works better, but is not in stable baselines
    model = A2C("MlpPolicy", env)

    # Use PPO https://arxiv.org/pdf/1707.06347.pdf: better performance than A2C on more complicated Atari games
    # Should work well on basic RL environments
    # model = PPO("MlpPolicy", env)

    # command for the model to train, according to stable baseline3 docs, 25_000 is enough for PPO to learn
    print("Model started learning")
    model.learn(total_timesteps=args.timesteps)

    # model.save("./TrainedModels/cartpole_a2c")

    # assume that the model has been trained
    # model = A2C.load("./TrainedModels/cartpole_a2c")
    # model = PPO.load("./TrainedModels/cartpole_a2c")

    # use trained model to generate data
    # need state, action, and image sequences ie. at each timestep: https://github.com/clvrai/spirl#adding-a-new-dataset-for-model-training
    # will need to format dataloader to take this data as input
    episodes = args.episodes
    max_steps = args.max_steps
    observation = env.reset()
    action_sequence = []
    state_sequence = []
    image_sequence = []
    done_sequence = []
    for cur_episode in range(episodes):
        for step in range(max_steps):
            # policy predicts what action to take and track action + state info
            action, _ = model.predict(observation)
            action_sequence.append(action)
            state_sequence.append(observation)
            image_sequence.append(env.render("rgb_array"))

            # for visualization
            env.render()

            # take the predicted action
            observation, reward, done, info = env.step(action)
            done_sequence.append(done)

            # save data and reset everything if this episode is done
            if done:
                observation = env.reset()
                print("episode: ", cur_episode)
                print("num_steps: ", step)
                print("~~~~~~~~~~~~~~~~~~~~~~~~")
                # save the data
                save_data(
                    action_sequence,
                    state_sequence,
                    image_sequence,
                    done_sequence,
                    cur_episode,
                )

                # reset the tracked data
                action_sequence = []
                state_sequence = []
                image_sequence = []
                # stop iterating on this episode -- move to next
                break
