"""Utilities for frozen base-autoencoder embedding splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """NPZ-backed dataset for standardized base-autoencoder embeddings."""

    def __init__(self, path: str | Path, mean=None, std=None):
        data = np.load(path)
        embeddings = data["embeddings"].astype(np.float32)
        self.digits = data["digits"].astype(np.int64)
        self.colors = data["colors"].astype(np.int64)
        self.digit_groups = data["digit_groups"].astype(np.int64)
        self.metadata = str(data["metadata"])
        data.close()

        if mean is None:
            mean = embeddings.mean(axis=0, keepdims=True)
        if std is None:
            std = embeddings.std(axis=0, keepdims=True)

        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.embeddings = ((embeddings - self.mean) / self.std).astype(np.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        return {
            "embedding": torch.from_numpy(self.embeddings[index]),
            "digit": torch.tensor(self.digits[index], dtype=torch.long),
            "color": torch.tensor(self.colors[index], dtype=torch.long),
            "digit_group": torch.tensor(self.digit_groups[index], dtype=torch.long),
        }


def load_scaler(path: str | Path):
    """Load an embedding scaler saved by split-SAE training."""
    data = np.load(path)
    mean = data["mean"].astype(np.float32)
    std = np.maximum(data["std"].astype(np.float32), 1e-6)
    data.close()
    return mean, std


def load_embedding_dataset(
    embedding_dir: str | Path,
    split_name: str,
    mean=None,
    std=None,
) -> EmbeddingDataset:
    """Load one named embedding split as an ``EmbeddingDataset``."""
    path = Path(embedding_dir) / f"{split_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding split: {path}")
    return EmbeddingDataset(path, mean=mean, std=std)


def load_embedding_split(embedding_dir: str | Path, split_name: str) -> dict:
    """Load one named embedding split as raw NumPy arrays."""
    path = Path(embedding_dir) / f"{split_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding split: {path}")

    data = np.load(path)
    split = {
        "embeddings": data["embeddings"].astype(np.float32),
        "digits": data["digits"].astype(np.int64),
        "colors": data["colors"].astype(np.int64),
        "digit_groups": data["digit_groups"].astype(np.int64),
        "metadata": str(data["metadata"]),
    }
    data.close()
    return split
