from spirl.utils.general_utils import AttrDict
from spirl.data.cartpole.src.cartpole_data_loader import CartPoleDataset
from spirl.components.data_loader import GlobalSplitVideoDataset

# https://gym.openai.com/envs/CartPole-v1/: action space is +/- 1 (direction to move cart)
# http://researchers.lille.inria.fr/~munos/variable/cartpole.html state space is position, angle of pole, velocity, change in angle
# data_spec = AttrDict(
#     dataset_class=CartPoleDataset,
#     n_actions=1,
#     state_dim=4,
#     res=32,
#     env_name="CartPole-v1",
#     crop_rand_subseq=True,
# )
# data_spec.max_seq_len = 300

# apparently they never use MazeDataLoader either
data_spec = AttrDict(
    dataset_class=GlobalSplitVideoDataset,
    n_actions=1,
    state_dim=4,
    split=AttrDict(train=0.9, val=0.1, test=0.0),
    res=32,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 300
