"""Train the task-free split SAE on frozen base-AE embeddings."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_sae.models.split_sae import SplitSparseAutoencoder
from toy_sae.utils.embeddings import EmbeddingDataset
from toy_sae.utils.history_plots import plot_history_metrics
from toy_sae.utils.script_utils import args_to_dict, save_json
from toy_sae.utils.torch_utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/base_ae_embeddings"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/split_sae"))
    parser.add_argument("--train-split", default="split_train_biased")
    parser.add_argument("--val-split", default="split_val_biased")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--good-latent-dim", type=int, default=128)
    parser.add_argument("--bad-latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-good-recon", type=float, default=None)
    parser.add_argument("--lambda-badcon", type=float, default=0.1)
    parser.add_argument("--lambda-sparse-good", type=float, default=1e-3)
    parser.add_argument("--lambda-sparse-bad", type=float, default=5e-3)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-dom", type=float, default=1.0)
    parser.add_argument("--grl-lambda", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def accuracy_from_logits(logits, labels):
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean()


def latent_stats(z):
    active_fraction = (z > 0).float().mean()
    return {
        "mean_abs": z.abs().mean(),
        "active_fraction": active_fraction,
        "active_count": active_fraction * z.shape[1],
    }


def make_model(args):
    return SplitSparseAutoencoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        good_latent_dim=args.good_latent_dim,
        bad_latent_dim=args.bad_latent_dim,
    )


def run_epoch(model, loader, optimizer, device, args, training):
    if training:
        model.train()
    else:
        model.eval()

    ce_loss = nn.CrossEntropyLoss()
    totals = {
        "total_loss": 0.0,
        "recon_mse": 0.0,
        "good_recon_mse": 0.0,
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
    total_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, disable=True):
            embeddings = batch["embedding"].to(device)
            colors = batch["color"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(embeddings, grl_lambda=args.grl_lambda)
            reconstruction = outputs["reconstruction"]
            good_reconstruction = outputs["good_reconstruction"]
            bad_reconstruction = outputs["bad_reconstruction"]
            z_good = outputs["z_good"]
            z_bad = outputs["z_bad"]
            good_color_logits = outputs["good_color_logits"]
            bad_color_logits = outputs["bad_color_logits"]

            recon_mse = ((reconstruction - embeddings) ** 2).mean()
            good_recon_mse = ((good_reconstruction - embeddings) ** 2).mean()
            badcon_loss = (bad_reconstruction ** 2).mean()
            good_sparsity = z_good.abs().mean()
            bad_sparsity = z_bad.abs().mean()
            good_color_loss = ce_loss(good_color_logits, colors)
            bad_color_loss = ce_loss(bad_color_logits, colors)
            good_color_acc = accuracy_from_logits(good_color_logits, colors)
            bad_color_acc = accuracy_from_logits(bad_color_logits, colors)

            total_loss = (
                args.lambda_recon * recon_mse
                + args.lambda_good_recon * good_recon_mse
                + args.lambda_badcon * badcon_loss
                + args.lambda_sparse_good * good_sparsity
                + args.lambda_sparse_bad * bad_sparsity
                + args.lambda_adv * good_color_loss
                + args.lambda_dom * bad_color_loss
            )

            if training:
                total_loss.backward()
                optimizer.step()

            good_stats = latent_stats(z_good)
            bad_stats = latent_stats(z_bad)
            batch_size = embeddings.shape[0]
            total_examples += batch_size

            totals["total_loss"] += total_loss.item() * batch_size
            totals["recon_mse"] += recon_mse.item() * batch_size
            totals["good_recon_mse"] += good_recon_mse.item() * batch_size
            totals["badcon_loss"] += badcon_loss.item() * batch_size
            totals["good_color_loss"] += good_color_loss.item() * batch_size
            totals["bad_color_loss"] += bad_color_loss.item() * batch_size
            totals["good_sparsity"] += good_sparsity.item() * batch_size
            totals["bad_sparsity"] += bad_sparsity.item() * batch_size
            totals["good_color_acc"] += good_color_acc.item() * batch_size
            totals["bad_color_acc"] += bad_color_acc.item() * batch_size
            totals["z_good_mean_abs"] += good_stats["mean_abs"].item() * batch_size
            totals["z_bad_mean_abs"] += bad_stats["mean_abs"].item() * batch_size
            totals["z_good_active_frac"] += good_stats["active_fraction"].item() * batch_size
            totals["z_bad_active_frac"] += bad_stats["active_fraction"].item() * batch_size
            totals["z_good_active_count"] += good_stats["active_count"].item() * batch_size
            totals["z_bad_active_count"] += bad_stats["active_count"].item() * batch_size

    return {key: value / total_examples for key, value in totals.items()}


def save_checkpoint(path, model, optimizer, args, epoch, train_metrics, val_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "args": args_to_dict(args),
        },
        path,
    )


def save_scaler(path, train_dataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mean=train_dataset.mean,
        std=train_dataset.std,
    )


def main():
    args = parse_args()
    if args.lambda_good_recon is None:
        args.lambda_good_recon = 0.1

    set_seed(args.seed, deterministic=args.deterministic)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.embedding_dir / f"{args.train_split}.npz"
    val_path = args.embedding_dir / f"{args.val_split}.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train embeddings: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation embeddings: {val_path}")

    train_dataset = EmbeddingDataset(train_path)
    val_dataset = EmbeddingDataset(val_path, mean=train_dataset.mean, std=train_dataset.std)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Train embeddings: {train_path} ({len(train_dataset)} examples)")
    print(f"Val embeddings: {val_path} ({len(val_dataset)} examples)")
    print(f"Embedding scaler mean={train_dataset.mean.mean():.4f} std={train_dataset.std.mean():.4f}")

    print("Model: shared-encoder split SAE")

    model = make_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config_path = args.checkpoint_dir / "config.json"
    history_path = args.checkpoint_dir / "history.json"
    scaler_path = args.checkpoint_dir / "embedding_scaler.npz"
    save_json(config_path, args_to_dict(args))
    save_scaler(scaler_path, train_dataset)

    best_val_recon = float("inf")
    best_val_total = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args, False)
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        message = (
            f"epoch {epoch:03d} | "
            f"train_loss={train_metrics['total_loss']:.6f} "
            f"val_loss={val_metrics['total_loss']:.6f} | "
            f"val_recon={val_metrics['recon_mse']:.6f} "
        )
        message += (
            f"val_good_branch={val_metrics['good_recon_mse']:.6f} "
            f"val_badcon={val_metrics['badcon_loss']:.6f} "
        )
        message += (
            "| "
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
            optimizer,
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
                optimizer,
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
                optimizer,
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
