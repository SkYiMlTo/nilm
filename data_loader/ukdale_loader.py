import h5py
import torch
import numpy as np

class UKDALELoader:
    """
    Loader for your current UK-DALE HDF5 file (House 1 only)
    """

    def __init__(self, h5_path, window_size=128, stride=64, device=None):
        self.h5_path = h5_path
        self.window_size = window_size
        self.stride = stride
        self.device = device if device else torch.device("cpu")

        self.h5_file = h5py.File(self.h5_path, 'r')

        # Hardcode appliances for House 1
        self.appliance_map = {
            'fridge': 'house_1/elec/fridge',
            'kettle': 'house_1/elec/kettle',
            'dishwasher': 'house_1/elec/dishwasher'
        }

    def get_windows(self):
        """
        Load House 1 mains + appliances and return sliding windows
        """
        mains = np.array(self.h5_file['house_1/elec/mains'][:], dtype=np.float32)

        targets = []
        for appl_name, path in self.appliance_map.items():
            targets.append(np.array(self.h5_file[path][:], dtype=np.float32))
        targets = np.stack(targets, axis=1)

        # Normalize
        mains = (mains - mains.mean()) / (mains.std() + 1e-8)
        targets = (targets - targets.mean(axis=0)) / (targets.std(axis=0) + 1e-8)

        # Sliding windows
        X_windows, Y_windows = [], []
        for start in range(0, len(mains) - self.window_size + 1, self.stride):
            end = start + self.window_size
            X_windows.append(mains[start:end])
            Y_windows.append(targets[start:end])
        X_windows = torch.tensor(np.stack(X_windows), device=self.device)
        Y_windows = torch.tensor(np.stack(Y_windows), device=self.device)

        return X_windows, Y_windows

    def close(self):
        self.h5_file.close()
