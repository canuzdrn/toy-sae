#!/usr/bin/env python
"""Extract frozen base-autoencoder embeddings for all ColoredMNIST splits."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_sae.datasets.npz_dataset import ColoredMNISTNPZDataset
from toy_sae.models.base_autoencoder import ConvAutoencoder


DEFAULT_SPLITS = [
    "ae_train_balanced",
    "ae_val_balanced",
    "split_train_biased",
    "split_val_biased",
    "test_id_biased",
    "test_balanced",
    "test_reversed",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/colored_mnist"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/base_ae/best.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/base_ae_embeddings"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    embedding_dim = checkpoint["embedding_dim"]
    model = ConvAutoencoder(embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def extract_split(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            batch_embeddings = model.encode(images)
            embeddings.append(batch_embeddings.cpu().numpy())

    return np.concatenate(embeddings, axis=0).astype(np.float32)


def save_embeddings(path, embeddings, dataset, split_name, checkpoint_path, checkpoint):
    metadata = {
        "split": split_name,
        "source_dataset_metadata": str(dataset.metadata),
        "source_checkpoint": str(checkpoint_path),
        "base_ae_epoch": int(checkpoint["epoch"]),
        "base_ae_embedding_dim": int(checkpoint["embedding_dim"]),
        "base_ae_val_weighted_loss": float(checkpoint["val_weighted_loss"]),
        "base_ae_val_mse": float(checkpoint["val_mse"]),
        "embedding_mean": float(embeddings.mean()),
        "embedding_std": float(embeddings.std()),
        "array_layout": "embeddings are float32 with shape (N, embedding_dim)",
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embeddings,
        digits=dataset.digits,
        colors=dataset.colors,
        digit_groups=dataset.digit_groups,
        metadata=json.dumps(metadata, sort_keys=True),
    )


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    device = get_device()
    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.out_dir}")

    model, checkpoint = load_model(args.checkpoint, device)

    for split_name in args.splits:
        split_path = args.data_dir / f"{split_name}.npz"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split: {split_path}")

        dataset = ColoredMNISTNPZDataset(split_path)
        print(f"Extracting {split_name}: {len(dataset)} examples")
        embeddings = extract_split(
            model,
            dataset,
            args.batch_size,
            args.num_workers,
            device,
        )

        out_path = args.out_dir / f"{split_name}.npz"
        save_embeddings(out_path, embeddings, dataset, split_name, args.checkpoint, checkpoint)
        print(
            f"  saved {out_path} with embeddings shape {embeddings.shape} | "
            f"mean={embeddings.mean():.4f} std={embeddings.std():.4f}"
        )


if __name__ == "__main__":
    main()
