"""Visualize content and style transforms from a factorized split-SAE checkpoint."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.debug.visualize_reconstructions import (
    COLOR_NAMES,
    choose_indices,
    image_from_array,
    image_from_tensor,
    load_base_autoencoder,
    load_embedding_split,
    load_image_split,
    validate_indices,
    validate_matching_labels,
)
from toy_sae.utils.checkpoints import load_checkpoint, resolve_checkpoint_path
from toy_sae.utils.embeddings import load_scaler
from toy_sae.utils.split_sae_loading import infer_model_family, load_split_sae_model
from toy_sae.utils.torch_utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="best_recon")
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/base_ae_embeddings"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/colored_mnist"))
    parser.add_argument("--split", default="test_balanced")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("checkpoints/base_ae/best.pt"),
    )
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def decode_embeddings(base_ae, standardized, mean, std):
    raw = standardized * std + mean
    return base_ae.decode(raw)


def main():
    args = parse_args()
    device = get_device()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_dir, args.checkpoint_name)
    if args.scaler is None:
        args.scaler = checkpoint_path.parent / "embedding_scaler.npz"

    checkpoint = load_checkpoint(checkpoint_path)
    model_family = infer_model_family(checkpoint)
    if model_family != "factorized_style":
        raise ValueError(
            f"Expected a factorized_style checkpoint, inferred {model_family!r}"
        )
    model = load_split_sae_model(checkpoint, device, model_family=model_family)
    base_ae = load_base_autoencoder(args.base_checkpoint, device)
    mean, std = load_scaler(args.scaler)
    embedding_split = load_embedding_split(args.embedding_dir, args.split)
    image_split = load_image_split(args.image_dir, args.split)

    if args.indices is None:
        indices = choose_indices(
            len(embedding_split["embeddings"]),
            num_samples=4,
            seed=args.seed,
        )
    else:
        indices = np.asarray(args.indices, dtype=np.int64)
        validate_indices(indices, len(embedding_split["embeddings"]))
    validate_matching_labels(embedding_split, image_split, indices)

    standardized = (
        embedding_split["embeddings"][indices] - mean
    ) / std
    embedding_tensor = torch.from_numpy(standardized.astype(np.float32)).to(device)
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)

    with torch.no_grad():
        z_good, z_bad = model.encode(embedding_tensor)
        content = model.decode_good(z_good)
        style_logits, style_weights = model.decode_style(z_bad)
        reconstruction = model.apply_style(content, style_weights)

        style_reconstructions = []
        for style_index in range(model.num_styles):
            one_hot = torch.zeros_like(style_weights)
            one_hot[:, style_index] = 1.0
            style_reconstructions.append(model.apply_style(content, one_hot))

        content_images = decode_embeddings(
            base_ae,
            content,
            mean_tensor,
            std_tensor,
        )
        style_images = [
            decode_embeddings(base_ae, values, mean_tensor, std_tensor)
            for values in style_reconstructions
        ]
        reconstructed_images = decode_embeddings(
            base_ae,
            reconstruction,
            mean_tensor,
            std_tensor,
        )

    if image_split is None:
        with torch.no_grad():
            input_decodes = base_ae.decode(
                torch.from_numpy(
                    embedding_split["embeddings"][indices]
                ).to(device)
            )
        input_images = [image_from_tensor(image) for image in input_decodes]
        input_label = "base-AE input decode"
    else:
        input_images = [
            image_from_array(image_split["images"][index])
            for index in indices
        ]
        input_label = "input image"

    content_images = [image_from_tensor(image) for image in content_images]
    style_images = [
        [image_from_tensor(image) for image in images]
        for images in style_images
    ]
    reconstructed_images = [
        image_from_tensor(image) for image in reconstructed_images
    ]

    columns = 3 + model.num_styles
    fig, axes = plt.subplots(
        len(indices),
        columns,
        figsize=(3.0 * columns, 2.9 * len(indices)),
        squeeze=False,
    )
    for row, index in enumerate(indices):
        digit = int(embedding_split["digits"][index])
        group = int(embedding_split["digit_groups"][index])
        color = int(embedding_split["colors"][index])
        color_name = COLOR_NAMES.get(color, str(color))

        axes[row, 0].imshow(input_images[row])
        axes[row, 0].set_title(
            f"{input_label}\ndigit={digit}, group={group}, color={color_name}",
            fontsize=9,
        )
        axes[row, 1].imshow(content_images[row])
        content_mse = ((content[row] - embedding_tensor[row]) ** 2).mean().item()
        axes[row, 1].set_title(
            f"canonical content\nemb MSE={content_mse:.4f}",
            fontsize=9,
        )

        for style_index in range(model.num_styles):
            axes[row, 2 + style_index].imshow(style_images[style_index][row])
            style_mse = (
                (style_reconstructions[style_index][row] - embedding_tensor[row])
                ** 2
            ).mean().item()
            style_name = COLOR_NAMES.get(style_index, str(style_index))
            axes[row, 2 + style_index].set_title(
                f"forced style {style_index} ({style_name})\nemb MSE={style_mse:.4f}",
                fontsize=9,
            )

        weights = ", ".join(
            f"{value:.2f}" for value in style_weights[row].cpu().tolist()
        )
        full_mse = (
            (reconstruction[row] - embedding_tensor[row]) ** 2
        ).mean().item()
        axes[row, -1].imshow(reconstructed_images[row])
        axes[row, -1].set_title(
            f"selected mixture [{weights}]\nemb MSE={full_mse:.4f}",
            fontsize=9,
        )

        for axis in axes[row]:
            axis.axis("off")

    fig.suptitle(f"{checkpoint_path.parent.name} | {args.split}", fontsize=12)
    fig.tight_layout()

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=180, bbox_inches="tight")
        print(f"Saved visualization to {args.out}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
