"""PyTorch dataset wrappers for generated ColoredMNIST npz files."""

import numpy as np
import torch
from torch.utils.data import Dataset


class ColoredMNISTNPZDataset(Dataset):
    """Load one generated ColoredMNIST split from a compressed npz file."""

    def __init__(self, path):
        data = np.load(path)
        self.images = data["images"].astype(np.float32)
        self.digits = data["digits"].astype(np.int64)
        self.colors = data["colors"].astype(np.int64)
        self.digit_groups = data["digit_groups"].astype(np.int64)
        self.metadata = str(data["metadata"])
        data.close()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            "image": torch.from_numpy(self.images[index]),
            "digit": torch.tensor(self.digits[index], dtype=torch.long),
            "color": torch.tensor(self.colors[index], dtype=torch.long),
            "digit_group": torch.tensor(self.digit_groups[index], dtype=torch.long),
        }

