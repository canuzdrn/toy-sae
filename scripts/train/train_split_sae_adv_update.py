"""Train shared-trunk split SAE with extra adversary-head updates per batch."""

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
    parser.add_argument("--grl-lambda", type=float, default=5.0)
    parser.add_argument("--adversary-steps", type=int, default=3)
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


def set_requires_grad(module, value):
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def set_backbone_trainable(model, value):
    set_requires_grad(model.encoder, value)
    set_requires_grad(model.good_encoder, value)
    set_requires_grad(model.bad_encoder, value)
    set_requires_grad(model.good_decoder, value)
    set_requires_grad(model.bad_decoder, value)


def set_color_heads_trainable(model, value):
    set_requires_grad(model.good_color_head, value)
    set_requires_grad(model.bad_color_head, value)


def make_model(args):
    return SplitSparseAutoencoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        good_latent_dim=args.good_latent_dim,
        bad_latent_dim=args.bad_latent_dim,
    )


def backbone_parameters(model):
    return (
        list(model.encoder.parameters())
        + list(model.good_encoder.parameters())
        + list(model.bad_encoder.parameters())
        + list(model.good_decoder.parameters())
        + list(model.bad_decoder.parameters())
    )


def color_head_parameters(model):
    return list(model.good_color_head.parameters()) + list(model.bad_color_head.parameters())


def train_adversary_heads(model, embeddings, colors, head_optimizer, ce_loss, args):
    if args.adversary_steps <= 0:
        return None

    set_backbone_trainable(model, False)
    set_color_heads_trainable(model, True)

    model.train()
    with torch.no_grad():
        z_good, z_bad = model.encode(embeddings)
        z_good = z_good.detach()
        z_bad = z_bad.detach()

    total_head_loss = 0.0
    for _ in range(args.adversary_steps):
        head_optimizer.zero_grad(set_to_none=True)
        good_color_logits = model.classify_good_color_no_grl(z_good)
        bad_color_logits = model.classify_bad_color(z_bad)
        good_color_loss = ce_loss(good_color_logits, colors)
        bad_color_loss = ce_loss(bad_color_logits, colors)
        head_loss = args.lambda_adv * good_color_loss + args.lambda_dom * bad_color_loss
        head_loss.backward()
        head_optimizer.step()
        total_head_loss += head_loss.item()

    return total_head_loss / args.adversary_steps


def compute_losses(outputs, embeddings, colors, ce_loss, args):
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

    return {
        "total_loss": total_loss,
        "recon_mse": recon_mse,
        "good_recon_mse": good_recon_mse,
        "badcon_loss": badcon_loss,
        "good_color_loss": good_color_loss,
        "bad_color_loss": bad_color_loss,
        "good_sparsity": good_sparsity,
        "bad_sparsity": bad_sparsity,
        "good_color_acc": good_color_acc,
        "bad_color_acc": bad_color_acc,
    }


def run_epoch(model, loader, main_optimizer, head_optimizer, device, args, training):
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
    if training:
        totals["head_update_loss"] = 0.0
    total_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, disable=True):
            embeddings = batch["embedding"].to(device)
            colors = batch["color"].to(device)
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

            if training:
                losses["total_loss"].backward()
                main_optimizer.step()
                set_color_heads_trainable(model, True)

            z_good = outputs["z_good"]
            z_bad = outputs["z_bad"]
            good_stats = latent_stats(z_good)
            bad_stats = latent_stats(z_bad)
            total_examples += batch_size

            for key in [
                "total_loss",
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
            "args": args_to_dict(args, extra={"training_procedure": "adversary_head_updates"}),
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
    if args.head_lr is None:
        args.head_lr = args.lr

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
    print(f"Training procedure: {args.adversary_steps} adversary-head update(s) per batch")

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
    config = args_to_dict(args, extra={"training_procedure": "adversary_head_updates"})
    save_json(config_path, config)
    save_scaler(scaler_path, train_dataset)

    best_val_recon = float("inf")
    best_val_total = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            main_optimizer,
            head_optimizer,
            device,
            args,
            True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            None,
            device,
            args,
            False,
        )
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
            f"val_good_branch={val_metrics['good_recon_mse']:.6f} "
            f"val_badcon={val_metrics['badcon_loss']:.6f} "
            f"| val_good_color_acc={val_metrics['good_color_acc']:.4f} "
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
