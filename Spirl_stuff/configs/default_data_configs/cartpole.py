from spirl.utils.general_utils import AttrDict
from spirl.data.cartpole.src.cartpole_data_loader import CartPoleDataset

# https://gym.openai.com/envs/CartPole-v1/: action space is +/- 1 (direction to move cart)
# http://researchers.lille.inria.fr/~munos/variable/cartpole.html state space is position, angle of pole, velocity, change in angle
data_spec = AttrDict(
    dataset_class=CartPoleDataset,
    n_actions=2,
    state_dim=4,
    env_name="CartPole-v1",
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 280