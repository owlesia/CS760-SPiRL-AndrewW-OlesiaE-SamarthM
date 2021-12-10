from spirl.rl.components.environment import GymEnv
from spirl.utils.pytorch_utils import ten2ar
import numpy as np
import math


class CartPoleEnv(GymEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, action):
        # action is binary for cartpole, but the decoder doesn't learn the sigmoid for classification
        decided_action = (1 / (1 + math.exp(-action[0]))) >= 0.5
        obs, rew, done, info = super().step(decided_action)
        # print(action)
        # print(type(action))
        # print(decided_action)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return obs, np.float64(rew), np.bool_(done), info     # casting reward to float64 is important for getting shape later

    def render(self, mode='rgb_array'):
        img = self._env.render("rgb_array")
        dim = img.shape
        compress_height_condition = [True if i %(dim[0]/32) == 0 else False for i in range(dim[0])]
        img = np.compress(compress_height_condition, img, 0)
        compress_width_condition = [True if i % (math.ceil(dim[1]/32)) == 0 else False for i in range(dim[1])]
        img = np.compress(compress_width_condition, img, 1)
        return img
