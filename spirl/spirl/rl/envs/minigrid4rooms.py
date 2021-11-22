from spirl.rl.components.environment import GymEnv


class Minigrid4RoomsEnv(GymEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
