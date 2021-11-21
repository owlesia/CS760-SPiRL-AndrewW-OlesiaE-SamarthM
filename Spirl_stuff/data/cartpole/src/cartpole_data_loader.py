import h5py
from spirl.components.data_loader import Dataset

class CartPoleDataset(Dataset):

    # use super for everything already implemented
    def __init__(self, data_dir, data_conf, phase, shuffle=True, dataset_size=-1, resolution=None):
        super().__init__(data_dir, data_conf, phase, shuffle=shuffle, dataset_size=dataset_size)

    def __getitem__(self, index):
        """Load a single sequence from disk according to index."""
        output = {}

        file_path = self.data_dir + self.phase + "/cartpole_a2c_data_0" + ".h5"
        f = h5py.File(file_path, "r")

        output["states"] = f['traj0']['states'][0:]
        output["actions"] = f['traj0']['actions'][0:]
        output["images"] = f['traj0']['images'][0:]

        f.close()
        return output

    def _get_samples_per_file(self, path):
        """Returns number of data samples per data file."""
        f = h5py.File(path, "r")
        samples_per_file = len(f['traj0']['states'][0:])
        f.close()
        return samples_per_file

