from spirl.rl.components.environment import GymEnv


class CartPoleEnv(GymEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, action):
        print(type(action))
        print(action)
        cur_action = action
        if (cur_action != 0 or cur_action != 1):
            if cur_action <= 0:
                cur_action = 0
            if cur_action > 0:
                cur_action = 1
            print("action changed to ", cur_action)
        return super().step(cur_action)