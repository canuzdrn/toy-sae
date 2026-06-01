"""Visualize Split-SAE image reconstructions from a checkpoint."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_sae.models.base_autoencoder import ConvAutoencoder
from toy_sae.utils.checkpoints import load_checkpoint, resolve_checkpoint_path
from toy_sae.utils.embeddings import load_scaler
from toy_sae.utils.split_sae_loading import load_split_sae_model
from toy_sae.utils.torch_utils import get_device


COLOR_NAMES = {0: "red", 1: "green"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Split-SAE checkpoint directory, or direct path to a .pt checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best_recon",
        help="Checkpoint name inside --checkpoint-dir. The .pt suffix is optional.",
    )
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/base_ae_embeddings"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/colored_mnist"))
    parser.add_argument("--split", default="test_balanced")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/base_ae/best.pt"))
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Dataset indices to visualize. If omitted, four examples are sampled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for deterministic random sampling when --indices is not provided.",
    )
    return parser.parse_args()


def load_base_autoencoder(path, device):
    checkpoint = load_checkpoint(path)
    model = ConvAutoencoder(embedding_dim=checkpoint.get("embedding_dim", 64))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_embedding_split(embedding_dir, split):
    path = embedding_dir / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding split: {path}")

    data = np.load(path)
    split_data = {
        "embeddings": data["embeddings"].astype(np.float32),
        "digits": data["digits"].astype(np.int64),
        "colors": data["colors"].astype(np.int64),
        "digit_groups": data["digit_groups"].astype(np.int64),
    }
    data.close()
    return split_data


def load_image_split(image_dir, split):
    path = image_dir / f"{split}.npz"
    if not path.exists():
        return None

    data = np.load(path)
    split_data = {
        "images": data["images"].astype(np.float32),
        "digits": data["digits"].astype(np.int64),
        "colors": data["colors"].astype(np.int64),
        "digit_groups": data["digit_groups"].astype(np.int64),
    }
    data.close()
    return split_data


def choose_indices(num_examples, num_samples=4, seed=None):
    if num_examples < num_samples:
        raise ValueError(f"Need at least {num_samples} examples, got {num_examples}")
    rng = np.random.default_rng(seed)
    return rng.choice(num_examples, size=num_samples, replace=False)


def validate_indices(indices, num_examples):
    if len(indices) != 4:
        raise ValueError(f"Expected exactly 4 indices for the 4-row plot, got {len(indices)}")
    for index in indices:
        if index < 0 or index >= num_examples:
            raise ValueError(f"Index {index} is outside split range [0, {num_examples})")


def image_from_tensor(tensor):
    image = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    return np.moveaxis(image, 0, -1)


def image_from_array(array):
    return np.moveaxis(np.clip(array, 0.0, 1.0), 0, -1)


def validate_matching_labels(embedding_split, image_split, indices):
    if image_split is None:
        return

    for index in indices:
        for key in ["digits", "colors", "digit_groups"]:
            embedding_value = int(embedding_split[key][index])
            image_value = int(image_split[key][index])
            if embedding_value != image_value:
                raise ValueError(
                    f"Embedding/image split mismatch at index {index}: "
                    f"{key} is {embedding_value} in embeddings but {image_value} in images."
                )


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint_dir, args.checkpoint_name)
    if args.scaler is None:
        args.scaler = checkpoint_path.parent / "embedding_scaler.npz"

    if not args.base_checkpoint.exists():
        raise FileNotFoundError(f"Missing base autoencoder checkpoint: {args.base_checkpoint}")
    if not args.scaler.exists():
        raise FileNotFoundError(f"Missing embedding scaler: {args.scaler}")

    checkpoint = load_checkpoint(checkpoint_path)
    split_sae = load_split_sae_model(checkpoint, device, model_family="auto")
    base_ae = load_base_autoencoder(args.base_checkpoint, device)

    mean, std = load_scaler(args.scaler)
    embedding_split = load_embedding_split(args.embedding_dir, args.split)
    image_split = load_image_split(args.image_dir, args.split)

    if args.indices is None:
        indices = choose_indices(len(embedding_split["embeddings"]), num_samples=4, seed=args.seed)
    else:
        indices = np.array(args.indices, dtype=np.int64)
        validate_indices(indices, len(embedding_split["embeddings"]))
    validate_matching_labels(embedding_split, image_split, indices)

    raw_embedding = embedding_split["embeddings"][indices]
    standardized_embedding = (raw_embedding - mean) / std

    embedding_tensor = torch.from_numpy(standardized_embedding).to(device)
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)

    with torch.no_grad():
        outputs = split_sae(embedding_tensor, grl_lambda=0.0)
        good_embedding = outputs["good_reconstruction"] * std_tensor + mean_tensor
        bad_embedding = outputs["bad_reconstruction"] * std_tensor + mean_tensor
        reconstructed_embedding = outputs["reconstruction"] * std_tensor + mean_tensor
        good_images = base_ae.decode(good_embedding)
        bad_images = base_ae.decode(bad_embedding)
        reconstructed_images = base_ae.decode(reconstructed_embedding)

    if image_split is not None:
        input_images = [image_from_array(image_split["images"][index]) for index in indices]
        input_title = "input image"
        full_pixel_mses = [
            float(((image_split["images"][index] - reconstructed_images[row].cpu().numpy()) ** 2).mean())
            for row, index in enumerate(indices)
        ]
        good_pixel_mses = [
            float(((image_split["images"][index] - good_images[row].cpu().numpy()) ** 2).mean())
            for row, index in enumerate(indices)
        ]
        bad_pixel_mses = [
            float(((image_split["images"][index] - bad_images[row].cpu().numpy()) ** 2).mean())
            for row, index in enumerate(indices)
        ]
    else:
        raw_embedding_tensor = torch.from_numpy(raw_embedding).to(device)
        with torch.no_grad():
            decoded_input_images = base_ae.decode(raw_embedding_tensor)
        input_images = [image_from_tensor(image) for image in decoded_input_images]
        input_title = "base-AE decode(input embedding)"
        full_pixel_mses = None
        good_pixel_mses = None
        bad_pixel_mses = None

    good_images = [image_from_tensor(image) for image in good_images]
    bad_images = [image_from_tensor(image) for image in bad_images]
    reconstructed_images = [image_from_tensor(image) for image in reconstructed_images]
    good_embedding_mses = ((outputs["good_reconstruction"] - embedding_tensor) ** 2).mean(dim=1)
    bad_embedding_mses = ((outputs["bad_reconstruction"] - embedding_tensor) ** 2).mean(dim=1)
    embedding_mses = ((outputs["reconstruction"] - embedding_tensor) ** 2).mean(dim=1)
    good_embedding_mses = [float(value.item()) for value in good_embedding_mses]
    bad_embedding_mses = [float(value.item()) for value in bad_embedding_mses]
    embedding_mses = [float(value.item()) for value in embedding_mses]

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {args.split}")
    print(
        "index | digit | group | color | good_emb_mse | bad_emb_mse | "
        "full_emb_mse | good_image_mse | bad_image_mse | full_image_mse"
    )
    for row, index in enumerate(indices):
        digit = int(embedding_split["digits"][index])
        color = int(embedding_split["colors"][index])
        group = int(embedding_split["digit_groups"][index])
        color_name = COLOR_NAMES.get(color, str(color))
        good_image_mse_text = f"{good_pixel_mses[row]:.6f}" if good_pixel_mses is not None else "n/a"
        bad_image_mse_text = f"{bad_pixel_mses[row]:.6f}" if bad_pixel_mses is not None else "n/a"
        full_image_mse_text = f"{full_pixel_mses[row]:.6f}" if full_pixel_mses is not None else "n/a"
        print(
            f"{index} | {digit} | {group} | {color_name} | "
            f"{good_embedding_mses[row]:.6f} | {bad_embedding_mses[row]:.6f} | "
            f"{embedding_mses[row]:.6f} | {good_image_mse_text} | "
            f"{bad_image_mse_text} | {full_image_mse_text}"
        )

    fig, axes = plt.subplots(4, 4, figsize=(12.5, 11.0))
    for row, index in enumerate(indices):
        digit = int(embedding_split["digits"][index])
        color = int(embedding_split["colors"][index])
        group = int(embedding_split["digit_groups"][index])
        color_name = COLOR_NAMES.get(color, str(color))
        row_title = f"idx={index}, digit={digit}, group={group}, color={color_name}"

        axes[row, 0].imshow(input_images[row])
        axes[row, 0].set_title(f"{input_title}\n{row_title}", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(good_images[row])
        axes[row, 1].set_title(
            f"good-only reconstruction\nemb MSE={good_embedding_mses[row]:.4f}",
            fontsize=10,
        )
        axes[row, 1].axis("off")

        axes[row, 2].imshow(bad_images[row])
        axes[row, 2].set_title(
            f"bad-only reconstruction\nemb MSE={bad_embedding_mses[row]:.4f}",
            fontsize=10,
        )
        axes[row, 2].axis("off")

        axes[row, 3].imshow(reconstructed_images[row])
        axes[row, 3].set_title(
            f"good + bad reconstruction\nemb MSE={embedding_mses[row]:.4f}",
            fontsize=10,
        )
        axes[row, 3].axis("off")

    fig.suptitle(f"{checkpoint_path.parent.name} | {args.split}", fontsize=12)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
