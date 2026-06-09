"""Train shared-trunk split SAE with canonical-color good reconstruction.

The usual split-SAE reconstruction target is the original colored base-AE
embedding:

    good_decoder(z_good) + bad_decoder(z_bad) ~= original colored embedding

This script adds a second target for only the good branch:

    good_decoder(z_good) ~= canonical-color embedding

By default, the full reconstruction loss is computed with no gradient into the
good branch, so that the original colored target trains the bad branch to supply
the missing residual without pulling the good branch back toward color. The
good-branch gradient from the full reconstruction loss can be partially restored
with --good-full-recon-grad-scale.

The canonical target is built from the raw ColoredMNIST image by removing the
original foreground color, recoloring the same image into a fixed/canonical
form, and encoding that image through the frozen base autoencoder. This tests
whether z_good can become the main content pathway without being forced to carry
the original color.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.train.train_split_sae_adv_update import (
    backbone_parameters,
    color_head_parameters,
    compute_losses,
    latent_stats,
    make_loader,
    make_model,
    set_backbone_trainable,
    set_color_heads_trainable,
    train_adversary_heads,
)
from toy_sae.datasets.colored_mnist import RGB_PALETTE
from toy_sae.models.base_autoencoder import ConvAutoencoder
from toy_sae.utils.checkpoints import load_checkpoint
from toy_sae.utils.embeddings import EmbeddingDataset
from toy_sae.utils.history_plots import plot_history_metrics
from toy_sae.utils.script_utils import args_to_dict, save_json
from toy_sae.utils.torch_utils import get_device, set_seed


class CanonicalGoodEmbeddingDataset(Dataset):
    """Embedding dataset with a canonical-color target for good reconstruction."""

    def __init__(
        self,
        embedding_path,
        image_path,
        base_ae,
        device,
        *,
        mean=None,
        std=None,
        batch_size=512,
        canonical_target="avg-red-green",
    ):
        self.base_dataset = EmbeddingDataset(embedding_path, mean=mean, std=std)
        image_data = np.load(image_path)
        images = image_data["images"].astype(np.float32)
        digits = image_data["digits"].astype(np.int64)
        colors = image_data["colors"].astype(np.int64)
        digit_groups = image_data["digit_groups"].astype(np.int64)
        image_data.close()

        if len(images) != len(self.base_dataset):
            raise ValueError(
                f"Image/embedding length mismatch: {len(images)} images vs "
                f"{len(self.base_dataset)} embeddings"
            )
        if not np.array_equal(digits, self.base_dataset.digits):
            raise ValueError(f"Digit labels differ between {image_path} and {embedding_path}")
        if not np.array_equal(colors, self.base_dataset.colors):
            raise ValueError(f"Color labels differ between {image_path} and {embedding_path}")
        if not np.array_equal(digit_groups, self.base_dataset.digit_groups):
            raise ValueError(
                f"Digit-group labels differ between {image_path} and {embedding_path}"
            )

        self.canonical_good_embeddings = make_canonical_good_embeddings(
            images,
            base_ae,
            device,
            mean=self.base_dataset.mean,
            std=self.base_dataset.std,
            batch_size=batch_size,
            target=canonical_target,
        )

        self.mean = self.base_dataset.mean
        self.std = self.base_dataset.std
        self.digits = self.base_dataset.digits
        self.colors = self.base_dataset.colors
        self.digit_groups = self.base_dataset.digit_groups
        self.metadata = self.base_dataset.metadata
        self.embeddings = self.base_dataset.embeddings

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        item = self.base_dataset[index]
        item["canonical_good_embedding"] = torch.from_numpy(
            self.canonical_good_embeddings[index]
        )
        return item


def parse_args(default_bad_latent_dim=32, description=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/base_ae_embeddings"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/colored_mnist"))
    parser.add_argument("--base-ae-checkpoint", type=Path, default=Path("checkpoints/base_ae/best.pt"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/split_sae"))
    parser.add_argument("--train-split", default="split_train_biased")
    parser.add_argument("--val-split", default="split_val_biased")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--good-latent-dim", type=int, default=128)
    parser.add_argument("--bad-latent-dim", type=int, default=default_bad_latent_dim)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--canonical-encode-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-good-recon", type=float, default=0.0)
    parser.add_argument("--lambda-badcon", type=float, default=0.1)
    parser.add_argument("--lambda-sparse-good", type=float, default=0.0)
    parser.add_argument("--lambda-sparse-bad", type=float, default=0.0)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-dom", type=float, default=1.0)
    parser.add_argument(
        "--lambda-canonical-good",
        type=float,
        default=0.5,
        help="MSE weight for good_reconstruction against the canonical-color target.",
    )
    parser.add_argument(
        "--lambda-bad-residual",
        type=float,
        default=0.0,
        help=(
            "MSE weight for bad_reconstruction against original minus canonical "
            "embedding. Use this to explicitly make z_bad the residual pathway."
        ),
    )
    parser.add_argument(
        "--good-full-recon-grad-scale",
        type=float,
        default=0.0,
        help=(
            "Scale for the full-reconstruction gradient into good_reconstruction. "
            "0.0 fully detaches the good branch; 1.0 uses the original full gradient."
        ),
    )
    parser.add_argument(
        "--detach-good-in-full-recon",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compatibility shortcut. --detach-good-in-full-recon sets "
            "--good-full-recon-grad-scale 0.0; --no-detach-good-in-full-recon "
            "sets it to 1.0."
        ),
    )
    parser.add_argument(
        "--canonical-good-target",
        choices=["avg-red-green", "gray", "red", "green"],
        default="avg-red-green",
        help=(
            "Canonical target encoded by the frozen base AE. avg-red-green averages "
            "the red and green base-AE embeddings for the same image."
        ),
    )
    parser.add_argument("--grl-lambda", type=float, default=5.0)
    parser.add_argument("--adversary-steps", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def resolve_good_full_recon_grad_scale(args):
    if args.detach_good_in_full_recon is not None:
        shortcut_scale = 0.0 if args.detach_good_in_full_recon else 1.0
        if args.good_full_recon_grad_scale != 0.0:
            raise ValueError(
                "Use either --good-full-recon-grad-scale or "
                "--detach-good-in-full-recon/--no-detach-good-in-full-recon, not both."
            )
        args.good_full_recon_grad_scale = shortcut_scale

    if not 0.0 <= args.good_full_recon_grad_scale <= 1.0:
        raise ValueError("--good-full-recon-grad-scale must be between 0.0 and 1.0")


def load_base_ae(checkpoint_path, device):
    checkpoint = load_checkpoint(checkpoint_path)
    model = ConvAutoencoder(embedding_dim=int(checkpoint["embedding_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def recolor_images(images, color_index):
    grayscale = images.max(axis=1)
    palette = RGB_PALETTE[color_index].astype(np.float32)
    return np.moveaxis(grayscale[..., None] * palette[None, None, None, :], -1, 1).astype(
        np.float32
    )


def make_gray_images(images):
    grayscale = images.max(axis=1)
    return np.repeat(grayscale[:, None, :, :], repeats=3, axis=1).astype(np.float32)


def encode_images(model, images, device, batch_size):
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start : start + batch_size]).to(device)
            embeddings.append(model.encode(batch).cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def make_canonical_good_embeddings(images, base_ae, device, *, mean, std, batch_size, target):
    if target == "avg-red-green":
        red_embeddings = encode_images(
            base_ae,
            recolor_images(images, color_index=0),
            device,
            batch_size,
        )
        green_embeddings = encode_images(
            base_ae,
            recolor_images(images, color_index=1),
            device,
            batch_size,
        )
        canonical_embeddings = 0.5 * (red_embeddings + green_embeddings)
    elif target == "red":
        canonical_embeddings = encode_images(
            base_ae,
            recolor_images(images, color_index=0),
            device,
            batch_size,
        )
    elif target == "green":
        canonical_embeddings = encode_images(
            base_ae,
            recolor_images(images, color_index=1),
            device,
            batch_size,
        )
    elif target == "gray":
        canonical_embeddings = encode_images(base_ae, make_gray_images(images), device, batch_size)
    else:
        raise ValueError(f"Unknown canonical target: {target}")

    return ((canonical_embeddings - mean) / std).astype(np.float32)


def run_epoch(model, loader, main_optimizer, head_optimizer, device, args, training):
    if training:
        model.train()
    else:
        model.eval()

    ce_loss = nn.CrossEntropyLoss()
    totals = {
        "total_loss": 0.0,
        "base_total_loss": 0.0,
        "recon_mse": 0.0,
        "good_recon_mse": 0.0,
        "canonical_good_mse": 0.0,
        "bad_residual_mse": 0.0,
        "badcon_loss": 0.0,
        "good_color_loss": 0.0,
        "bad_color_loss": 0.0,
        "good_sparsity": 0.0,
        "bad_sparsity": 0.0,
        "good_color_acc": 0.0,
        "bad_color_acc": 0.0,
        "z_good_mean_abs": 0.0,
        "z_bad_mean_abs": 0.0,
        "z_good_active_frac": 0.0,
        "z_bad_active_frac": 0.0,
        "z_good_active_count": 0.0,
        "z_bad_active_count": 0.0,
    }
    if training:
        totals["head_update_loss"] = 0.0
    total_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            embeddings = batch["embedding"].to(device)
            colors = batch["color"].to(device)
            canonical_good_embeddings = batch["canonical_good_embedding"].to(device)
            batch_size = embeddings.shape[0]

            head_update_loss = None
            if training:
                head_update_loss = train_adversary_heads(
                    model,
                    embeddings,
                    colors,
                    head_optimizer,
                    ce_loss,
                    args,
                )
                set_backbone_trainable(model, True)
                set_color_heads_trainable(model, False)
                main_optimizer.zero_grad(set_to_none=True)

            outputs = model(embeddings, grl_lambda=args.grl_lambda)
            losses = compute_losses(outputs, embeddings, colors, ce_loss, args)
            base_total_loss = losses["total_loss"]
            if args.good_full_recon_grad_scale < 1.0:
                good_reconstruction_for_recon_loss = (
                    args.good_full_recon_grad_scale * outputs["good_reconstruction"]
                    + (1.0 - args.good_full_recon_grad_scale)
                    * outputs["good_reconstruction"].detach()
                )
                reconstruction_for_loss = (
                    good_reconstruction_for_recon_loss + outputs["bad_reconstruction"]
                )
                recon_mse_for_loss = ((reconstruction_for_loss - embeddings) ** 2).mean()
                base_total_loss = (
                    base_total_loss
                    - args.lambda_recon * losses["recon_mse"]
                    + args.lambda_recon * recon_mse_for_loss
                )

            canonical_good_mse = (
                (outputs["good_reconstruction"] - canonical_good_embeddings) ** 2
            )
            canonical_good_mse = canonical_good_mse.mean()
            bad_residual_target = (embeddings - canonical_good_embeddings).detach()
            bad_residual_mse = (
                (outputs["bad_reconstruction"] - bad_residual_target) ** 2
            )
            bad_residual_mse = bad_residual_mse.mean()
            total_loss = (
                base_total_loss
                + args.lambda_canonical_good * canonical_good_mse
                + args.lambda_bad_residual * bad_residual_mse
            )

            if training:
                total_loss.backward()
                main_optimizer.step()
                set_color_heads_trainable(model, True)

            z_good = outputs["z_good"]
            z_bad = outputs["z_bad"]
            good_stats = latent_stats(z_good)
            bad_stats = latent_stats(z_bad)
            total_examples += batch_size

            totals["total_loss"] += total_loss.item() * batch_size
            totals["base_total_loss"] += base_total_loss.item() * batch_size
            totals["canonical_good_mse"] += canonical_good_mse.item() * batch_size
            totals["bad_residual_mse"] += bad_residual_mse.item() * batch_size
            for key in [
                "recon_mse",
                "good_recon_mse",
                "badcon_loss",
                "good_color_loss",
                "bad_color_loss",
                "good_sparsity",
                "bad_sparsity",
                "good_color_acc",
                "bad_color_acc",
            ]:
                totals[key] += losses[key].item() * batch_size

            totals["z_good_mean_abs"] += good_stats["mean_abs"].item() * batch_size
            totals["z_bad_mean_abs"] += bad_stats["mean_abs"].item() * batch_size
            totals["z_good_active_frac"] += good_stats["active_fraction"].item() * batch_size
            totals["z_bad_active_frac"] += bad_stats["active_fraction"].item() * batch_size
            totals["z_good_active_count"] += good_stats["active_count"].item() * batch_size
            totals["z_bad_active_count"] += bad_stats["active_count"].item() * batch_size
            if training and head_update_loss is not None:
                totals["head_update_loss"] += head_update_loss * batch_size

    set_backbone_trainable(model, True)
    set_color_heads_trainable(model, True)
    return {key: value / total_examples for key, value in totals.items()}


def save_checkpoint(path, model, main_optimizer, head_optimizer, args, epoch, train_metrics, val_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": main_optimizer.state_dict(),
            "main_optimizer_state_dict": main_optimizer.state_dict(),
            "head_optimizer_state_dict": head_optimizer.state_dict(),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "args": args_to_dict(
                args,
                extra={"training_procedure": "canonical_good_reconstruction"},
            ),
        },
        path,
    )


def save_scaler(path, train_dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mean=train_dataset.mean, std=train_dataset.std)


def main():
    args = parse_args()
    resolve_good_full_recon_grad_scale(args)
    if args.head_lr is None:
        args.head_lr = args.lr

    set_seed(args.seed, deterministic=args.deterministic)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_embedding_path = args.embedding_dir / f"{args.train_split}.npz"
    val_embedding_path = args.embedding_dir / f"{args.val_split}.npz"
    train_image_path = args.image_dir / f"{args.train_split}.npz"
    val_image_path = args.image_dir / f"{args.val_split}.npz"
    for path in [train_embedding_path, val_embedding_path, train_image_path, val_image_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    device = get_device()
    print(f"Using device: {device}")
    print(f"Train embeddings: {train_embedding_path}")
    print(f"Val embeddings: {val_embedding_path}")
    print(f"Good full-reconstruction gradient scale: {args.good_full_recon_grad_scale}")
    print(f"Train images: {train_image_path}")
    print(f"Val images: {val_image_path}")
    print(f"Base AE checkpoint: {args.base_ae_checkpoint}")
    print("Model: shared-encoder split SAE")
    print("Training procedure: adversary-head updates + canonical-color good reconstruction")

    base_ae = load_base_ae(args.base_ae_checkpoint, device)
    train_dataset = CanonicalGoodEmbeddingDataset(
        train_embedding_path,
        train_image_path,
        base_ae,
        device,
        batch_size=args.canonical_encode_batch_size,
        canonical_target=args.canonical_good_target,
    )
    val_dataset = CanonicalGoodEmbeddingDataset(
        val_embedding_path,
        val_image_path,
        base_ae,
        device,
        mean=train_dataset.mean,
        std=train_dataset.std,
        batch_size=args.canonical_encode_batch_size,
        canonical_target=args.canonical_good_target,
    )

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")
    print(f"Embedding scaler mean={train_dataset.mean.mean():.4f} std={train_dataset.std.mean():.4f}")
    print(f"canonical_good_target={args.canonical_good_target}")
    print(f"lambda_canonical_good={args.lambda_canonical_good}")

    model = make_model(args).to(device)
    main_optimizer = torch.optim.AdamW(
        backbone_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    head_optimizer = torch.optim.AdamW(
        color_head_parameters(model),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    config_path = args.checkpoint_dir / "config.json"
    history_path = args.checkpoint_dir / "history.json"
    scaler_path = args.checkpoint_dir / "embedding_scaler.npz"
    config = args_to_dict(args, extra={"training_procedure": "canonical_good_reconstruction"})
    save_json(config_path, config)
    save_scaler(scaler_path, train_dataset)

    best_val_recon = float("inf")
    best_val_total = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, main_optimizer, head_optimizer, device, args, True
        )
        val_metrics = run_epoch(model, val_loader, None, None, device, args, False)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        message = (
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics['total_loss']:.6f} "
            f"val_loss={val_metrics['total_loss']:.6f} | "
            f"val_recon={val_metrics['recon_mse']:.6f} "
            f"val_good_original={val_metrics['good_recon_mse']:.6f} "
            f"val_good_canonical={val_metrics['canonical_good_mse']:.6f} "
            f"val_bad_residual={val_metrics['bad_residual_mse']:.6f} "
            f"val_badcon={val_metrics['badcon_loss']:.6f} "
            f"val_good_color_acc={val_metrics['good_color_acc']:.4f} "
            f"val_bad_color_acc={val_metrics['bad_color_acc']:.4f} | "
            f"val_z_good_active={val_metrics['z_good_active_frac']:.4f} "
            f"({val_metrics['z_good_active_count']:.1f}/{args.good_latent_dim}) "
            f"val_z_bad_active={val_metrics['z_bad_active_frac']:.4f} "
            f"({val_metrics['z_bad_active_count']:.1f}/{args.bad_latent_dim})"
        )
        print(message)

        save_checkpoint(
            args.checkpoint_dir / "latest.pt",
            model,
            main_optimizer,
            head_optimizer,
            args,
            epoch,
            train_metrics,
            val_metrics,
        )

        if val_metrics["recon_mse"] < best_val_recon:
            best_val_recon = val_metrics["recon_mse"]
            save_checkpoint(
                args.checkpoint_dir / "best_recon.pt",
                model,
                main_optimizer,
                head_optimizer,
                args,
                epoch,
                train_metrics,
                val_metrics,
            )
            print(f"  saved new best reconstruction checkpoint to {args.checkpoint_dir / 'best_recon.pt'}")

        if val_metrics["total_loss"] < best_val_total:
            best_val_total = val_metrics["total_loss"]
            save_checkpoint(
                args.checkpoint_dir / "best_total.pt",
                model,
                main_optimizer,
                head_optimizer,
                args,
                epoch,
                train_metrics,
                val_metrics,
            )
            print(f"  saved new best total-loss checkpoint to {args.checkpoint_dir / 'best_total.pt'}")

    save_json(history_path, history)
    print(f"Saved config to {config_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved history to {history_path}")

    if not args.no_plots:
        plot_dir = args.checkpoint_dir / "plots"
        plot_result = plot_history_metrics(history_path, out_dir=plot_dir)
        print(f"Saved {len(plot_result['saved_paths'])} history plots to {plot_dir}")


if __name__ == "__main__":
    main()
