from spirl.utils.general_utils import AttrDict
from spirl.data.minigrid4rooms.src.minigrid4rooms_data_loader import (
    D4RLSequenceSplitDataset,
)


data_spec = AttrDict(
    dataset_class=D4RLSequenceSplitDataset,
    n_actions=9,
    state_dim=3,
    env_name="MiniGrid-FourRooms-v0",
    res=128,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 280
