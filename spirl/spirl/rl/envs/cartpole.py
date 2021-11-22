from spirl.rl.components.environment import GymEnv


class CartPoleEnv(GymEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    