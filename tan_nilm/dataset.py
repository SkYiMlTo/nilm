# dataset.py
import h5py
import torch
from torch.utils.data import Dataset

class NILMDataset(Dataset):
    def __init__(self, file_path, seq_len=256, stride=None, houses=['house_1'], app_channels=[1]):
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len  # default to non-overlapping
        self.file_path = file_path
        self.houses = houses
        self.app_channels = app_channels
        self.index = []

        # Build index lazily (just pointers to sequences)
        with h5py.File(file_path, 'r') as f:
            for house_key in self.houses:
                house = f[house_key]
                for app_ch in self.app_channels:
                    app_data = house[f'channel_{app_ch}'][:]
                    mains_data = house['mains'][:]
                    min_len = min(len(app_data), len(mains_data))
                    for i in range(0, min_len - seq_len, self.stride):
                        self.index.append((house_key, app_ch, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        house_key, app_ch, start_idx = self.index[idx]
        with h5py.File(self.file_path, 'r') as f:
            house = f[house_key]
            app_data = house[f'channel_{app_ch}'][start_idx:start_idx+self.seq_len]
            mains_data = house['mains'][start_idx:start_idx+self.seq_len]

        x = torch.tensor(mains_data, dtype=torch.float32).unsqueeze(-1)  # (seq_len, 1)
        y = torch.tensor(app_data, dtype=torch.float32).unsqueeze(-1)   # (seq_len, 1)
        return x, y
