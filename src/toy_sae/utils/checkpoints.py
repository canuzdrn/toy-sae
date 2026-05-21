"""Checkpoint loading and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a PyTorch checkpoint with compatibility across torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_checkpoint_name(name: str) -> str:
    """Add a .pt suffix to a checkpoint name when it is omitted."""
    if name.endswith(".pt"):
        return name
    return f"{name}.pt"


def resolve_checkpoint_path(checkpoint_dir: str | Path, checkpoint_name: str) -> Path:
    """Resolve a checkpoint name inside a run directory.

    ``checkpoint_dir`` may also be a direct path to a .pt checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.is_file():
        return checkpoint_dir

    checkpoint_path = checkpoint_dir / normalize_checkpoint_name(checkpoint_name)
    if checkpoint_path.exists():
        return checkpoint_path

    available = sorted(path.name for path in checkpoint_dir.glob("*.pt"))
    available_text = ", ".join(available) if available else "none"
    raise FileNotFoundError(
        f"Missing checkpoint {checkpoint_path}. Available .pt files in "
        f"{checkpoint_dir}: {available_text}"
    )
