"""Train shared-trunk split SAE with paired color-invariance on z_good.

This keeps the existing adversary-head-update training objective intact and adds
one paired diagnostic/training constraint:

    z_good(same image colored red) ~= z_good(same image colored green)

The pairs are built from the raw ColoredMNIST split by removing the foreground
color, recoloring the same digit image red and green, and encoding both through
the frozen base autoencoder.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.train.train_split_sae_adv_update import (
    accuracy_from_logits,
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


class PairedEmbeddingDataset(Dataset):
    """Embedding dataset with red/green paired embeddings for the same image."""

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

        paired_embeddings = make_paired_embeddings(
            images,
            base_ae,
            device,
            mean=self.base_dataset.mean,
            std=self.base_dataset.std,
            batch_size=batch_size,
        )
        self.red_embeddings = paired_embeddings["red"]
        self.green_embeddings = paired_embeddings["green"]

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
        item["red_embedding"] = torch.from_numpy(self.red_embeddings[index])
        item["green_embedding"] = torch.from_numpy(self.green_embeddings[index])
        return item


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/base_ae_embeddings"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/colored_mnist"))
    parser.add_argument("--base-ae-checkpoint", type=Path, default=Path("checkpoints/base_ae/best.pt"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/split_sae"))
    parser.add_argument("--train-split", default="split_train_biased")
    parser.add_argument("--val-split", default="split_val_biased")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--good-latent-dim", type=int, default=128)
    parser.add_argument("--bad-latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pair-encode-batch-size", type=int, default=512)
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
    parser.add_argument("--lambda-pair-good", type=float, default=0.1)
    parser.add_argument("--grl-lambda", type=float, default=5.0)
    parser.add_argument("--adversary-steps", type=int, default=5)
    parser.add_argument(
        "--adversary-on-pairs",
        action="store_true",
        help=(
            "Train the color heads on balanced recolored red/green paired embeddings "
            "during the head-only update step. The main GRL loss is still computed on "
            "the original split embeddings."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


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


def encode_images(model, images, device, batch_size):
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start : start + batch_size]).to(device)
            embeddings.append(model.encode(batch).cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def make_paired_embeddings(images, base_ae, device, *, mean, std, batch_size):
    red_images = recolor_images(images, color_index=0)
    green_images = recolor_images(images, color_index=1)
    red_embeddings = encode_images(base_ae, red_images, device, batch_size)
    green_embeddings = encode_images(base_ae, green_images, device, batch_size)

    red_embeddings = ((red_embeddings - mean) / std).astype(np.float32)
    green_embeddings = ((green_embeddings - mean) / std).astype(np.float32)
    return {"red": red_embeddings, "green": green_embeddings}


def pair_losses(model, red_embeddings, green_embeddings):
    z_good_red, z_bad_red = model.encode(red_embeddings)
    z_good_green, z_bad_green = model.encode(green_embeddings)
    good_pair_mse = ((z_good_red - z_good_green) ** 2).mean()
    bad_pair_mse = ((z_bad_red - z_bad_green) ** 2).mean()
    good_pair_cosine = F.cosine_similarity(z_good_red, z_good_green, dim=1).mean()
    bad_pair_cosine = F.cosine_similarity(z_bad_red, z_bad_green, dim=1).mean()
    return good_pair_mse, bad_pair_mse, good_pair_cosine, bad_pair_cosine


def make_paired_head_batch(red_embeddings, green_embeddings):
    embeddings = torch.cat([red_embeddings, green_embeddings], dim=0)
    colors = torch.cat(
        [
            torch.zeros(red_embeddings.shape[0], dtype=torch.long, device=red_embeddings.device),
            torch.ones(green_embeddings.shape[0], dtype=torch.long, device=green_embeddings.device),
        ],
        dim=0,
    )
    return embeddings, colors


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
        "badcon_loss": 0.0,
        "good_color_loss": 0.0,
        "bad_color_loss": 0.0,
        "good_pair_mse": 0.0,
        "bad_pair_mse": 0.0,
        "good_pair_cosine": 0.0,
        "bad_pair_cosine": 0.0,
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
            red_embeddings = batch["red_embedding"].to(device)
            green_embeddings = batch["green_embedding"].to(device)
            batch_size = embeddings.shape[0]

            head_update_loss = None
            if training:
                head_embeddings = embeddings
                head_colors = colors
                if args.adversary_on_pairs:
                    head_embeddings, head_colors = make_paired_head_batch(
                        red_embeddings,
                        green_embeddings,
                    )
                head_update_loss = train_adversary_heads(
                    model,
                    head_embeddings,
                    head_colors,
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
            good_pair_mse, bad_pair_mse, good_pair_cosine, bad_pair_cosine = pair_losses(
                model,
                red_embeddings,
                green_embeddings,
            )
            total_loss = base_total_loss + args.lambda_pair_good * good_pair_mse

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

            totals["good_pair_mse"] += good_pair_mse.item() * batch_size
            totals["bad_pair_mse"] += bad_pair_mse.item() * batch_size
            totals["good_pair_cosine"] += good_pair_cosine.item() * batch_size
            totals["bad_pair_cosine"] += bad_pair_cosine.item() * batch_size
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
            "args": args_to_dict(args, extra={"training_procedure": "paired_good_invariance"}),
        },
        path,
    )


def save_scaler(path, train_dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mean=train_dataset.mean, std=train_dataset.std)


def main():
    args = parse_args()
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
    print(f"Train paired images: {train_image_path}")
    print(f"Val paired images: {val_image_path}")
    print(f"Base AE checkpoint: {args.base_ae_checkpoint}")
    print("Model: shared-encoder split SAE")
    print("Training procedure: adversary-head updates + paired z_good invariance")

    base_ae = load_base_ae(args.base_ae_checkpoint, device)
    train_dataset = PairedEmbeddingDataset(
        train_embedding_path,
        train_image_path,
        base_ae,
        device,
        batch_size=args.pair_encode_batch_size,
    )
    val_dataset = PairedEmbeddingDataset(
        val_embedding_path,
        val_image_path,
        base_ae,
        device,
        mean=train_dataset.mean,
        std=train_dataset.std,
        batch_size=args.pair_encode_batch_size,
    )

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")
    print(f"Embedding scaler mean={train_dataset.mean.mean():.4f} std={train_dataset.std.mean():.4f}")
    print(f"lambda_pair_good={args.lambda_pair_good}")
    print(f"adversary_on_pairs={args.adversary_on_pairs}")

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
    config = args_to_dict(args, extra={"training_procedure": "paired_good_invariance"})
    save_json(config_path, config)
    save_scaler(scaler_path, train_dataset)

    best_val_recon = float("inf")
    best_val_total = float("inf")
    best_val_pair = float("inf")
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
            f"val_good_branch={val_metrics['good_recon_mse']:.6f} "
            f"val_badcon={val_metrics['badcon_loss']:.6f} "
            f"val_good_pair={val_metrics['good_pair_mse']:.6f} "
            f"val_bad_pair={val_metrics['bad_pair_mse']:.6f} "
            f"val_good_pair_cos={val_metrics['good_pair_cosine']:.4f} "
            f"val_bad_pair_cos={val_metrics['bad_pair_cosine']:.4f} | "
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

        if val_metrics["good_pair_mse"] < best_val_pair:
            best_val_pair = val_metrics["good_pair_mse"]
            save_checkpoint(
                args.checkpoint_dir / "best_pair.pt",
                model,
                main_optimizer,
                head_optimizer,
                args,
                epoch,
                train_metrics,
                val_metrics,
            )
            print(f"  saved new best pair-invariance checkpoint to {args.checkpoint_dir / 'best_pair.pt'}")

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
