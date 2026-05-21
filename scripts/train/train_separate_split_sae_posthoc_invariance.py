#!/usr/bin/env python
"""Train separate split-SAE and checkpoint by fresh post-hoc z_good invariance."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_sae.models.separate_split_sae import SeparateEncoderSAE
from toy_sae.utils.embeddings import EmbeddingDataset
from toy_sae.utils.history_plots import plot_history_metrics
from toy_sae.utils.script_utils import args_to_dict, save_json
from toy_sae.utils.torch_utils import get_device, set_seed


DEFAULT_PROBE_EVAL_SPLITS = [
    "split_val_biased",
    "test_balanced",
    "test_reversed",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/base_ae_embeddings"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/separate_split_sae"))
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
    parser.add_argument("--lambda-good-recon", type=float, default=0.0)
    parser.add_argument("--lambda-badcon", type=float, default=0.0)
    parser.add_argument("--lambda-sparse-good", type=float, default=0.0)
    parser.add_argument("--lambda-sparse-bad", type=float, default=0.0)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-dom", type=float, default=1.0)
    parser.add_argument("--grl-lambda", type=float, default=1.0)
    parser.add_argument("--probe-every", type=int, default=1)
    parser.add_argument("--probe-max-train-examples", type=int, default=10000)
    parser.add_argument("--probe-max-val-examples", type=int, default=10000)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--probe-max-iter", type=int, default=1000)
    parser.add_argument("--probe-eval-splits", nargs="+", default=DEFAULT_PROBE_EVAL_SPLITS)
    parser.add_argument("--probe-selection", default="test_balanced")
    parser.add_argument("--posthoc-tradeoff-alpha", type=float, default=0.25)
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


def metric_suffix(name):
    return name.replace("-", "_").replace("/", "_")


def make_color_balanced_probe_indices(colors, max_examples, seed):
    colors = np.asarray(colors)
    classes = np.unique(colors)
    class_indices = [np.flatnonzero(colors == class_id) for class_id in classes]
    min_class_count = min(len(indices) for indices in class_indices)
    if min_class_count == 0:
        raise ValueError("Cannot build a color-balanced probe subset with an empty color class")

    if max_examples <= 0:
        examples_per_class = min_class_count
    else:
        examples_per_class = min(max_examples // len(classes), min_class_count)
    if examples_per_class < 1:
        raise ValueError(
            f"Probe subset is too small for {len(classes)} color classes: max_examples={max_examples}"
        )

    rng = np.random.default_rng(seed)
    selected = []
    for indices in class_indices:
        selected.append(rng.choice(indices, size=examples_per_class, replace=False))
    return np.sort(np.concatenate(selected))


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
    return SeparateEncoderSAE(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        good_latent_dim=args.good_latent_dim,
        bad_latent_dim=args.bad_latent_dim,
    )


def extract_z_good_for_probe(model, dataset, indices, batch_size, num_workers, device):
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    was_training = model.training
    model.eval()

    z_good_batches = []
    color_batches = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, disable=True):
            embeddings = batch["embedding"].to(device)
            z_good, _ = model.encode(embeddings)
            z_good_batches.append(z_good.cpu().numpy())
            color_batches.append(batch["color"].numpy())

    if was_training:
        model.train()

    return (
        np.concatenate(z_good_batches, axis=0).astype(np.float32),
        np.concatenate(color_batches, axis=0).astype(np.int64),
    )


def make_color_probe(max_iter, seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def run_posthoc_invariance_probe(
    model,
    train_dataset,
    eval_datasets,
    train_indices,
    eval_indices,
    device,
    args,
):
    z_train, y_train = extract_z_good_for_probe(
        model,
        train_dataset,
        train_indices,
        args.probe_batch_size,
        args.num_workers,
        device,
    )
    probe = make_color_probe(args.probe_max_iter, args.seed)
    probe.fit(z_train, y_train)

    train_acc = accuracy_score(y_train, probe.predict(z_train))
    metrics = {
        "posthoc_z_good_color_train_acc": float(train_acc),
        "posthoc_probe_train_examples": int(len(train_indices)),
    }

    split_gaps = []
    split_accs = []
    for split_name, dataset in eval_datasets.items():
        z_eval, y_eval = extract_z_good_for_probe(
            model,
            dataset,
            eval_indices[split_name],
            args.probe_batch_size,
            args.num_workers,
            device,
        )
        color_acc = accuracy_score(y_eval, probe.predict(z_eval))
        color_gap = abs(color_acc - 0.5)
        suffix = metric_suffix(split_name)
        metrics[f"posthoc_z_good_color_acc_{suffix}"] = float(color_acc)
        metrics[f"posthoc_z_good_color_gap_{suffix}"] = float(color_gap)
        metrics[f"posthoc_probe_examples_{suffix}"] = int(len(eval_indices[split_name]))
        split_accs.append(color_acc)
        split_gaps.append(color_gap)

    mean_acc = float(np.mean(split_accs))
    mean_gap = float(np.mean(split_gaps))
    metrics["posthoc_z_good_color_mean_acc"] = mean_acc
    metrics["posthoc_z_good_color_mean_gap"] = mean_gap

    if args.probe_selection == "mean":
        selection_acc = mean_acc
        selection_gap = mean_gap
    else:
        selection_suffix = metric_suffix(args.probe_selection)
        selection_acc = metrics[f"posthoc_z_good_color_acc_{selection_suffix}"]
        selection_gap = metrics[f"posthoc_z_good_color_gap_{selection_suffix}"]

    metrics["posthoc_z_good_color_acc"] = float(selection_acc)
    metrics["posthoc_z_good_color_gap"] = float(selection_gap)
    metrics["posthoc_invariance_score"] = float(selection_gap)
    return metrics


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
            "args": args_to_dict(args, extra={"model_name": "separate_encoder_sae"}),
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
    if args.probe_every <= 0:
        raise ValueError("--probe-every must be positive because this script checkpoints by post-hoc probing")
    if args.probe_selection != "mean" and args.probe_selection not in args.probe_eval_splits:
        raise ValueError(
            "--probe-selection must be 'mean' or one of --probe-eval-splits. "
            f"Got {args.probe_selection!r} with eval splits {args.probe_eval_splits}."
        )

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
    probe_eval_datasets = {}
    for split_name in args.probe_eval_splits:
        split_path = args.embedding_dir / f"{split_name}.npz"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing probe eval embeddings: {split_path}")
        if split_name == args.train_split:
            probe_eval_datasets[split_name] = train_dataset
        elif split_name == args.val_split:
            probe_eval_datasets[split_name] = val_dataset
        else:
            probe_eval_datasets[split_name] = EmbeddingDataset(
                split_path,
                mean=train_dataset.mean,
                std=train_dataset.std,
            )

    probe_train_indices = make_color_balanced_probe_indices(
        train_dataset.colors,
        args.probe_max_train_examples,
        args.seed,
    )
    probe_eval_indices = {}
    for offset, (split_name, dataset) in enumerate(probe_eval_datasets.items(), start=1):
        probe_eval_indices[split_name] = make_color_balanced_probe_indices(
            dataset.colors,
            args.probe_max_val_examples,
            args.seed + offset,
        )

    device = get_device()
    print(f"Using device: {device}")
    print(f"Train embeddings: {train_path} ({len(train_dataset)} examples)")
    print(f"Val embeddings: {val_path} ({len(val_dataset)} examples)")
    print(f"Embedding scaler mean={train_dataset.mean.mean():.4f} std={train_dataset.std.mean():.4f}")
    print("Model: separate-encoder residual split SAE")
    print(
        "Post-hoc invariance probe: "
        f"every {args.probe_every} epoch(s), "
        f"{len(probe_train_indices)} train examples, "
        f"eval splits={args.probe_eval_splits}, "
        f"selection={args.probe_selection}, "
        f"tradeoff_alpha={args.posthoc_tradeoff_alpha}"
    )

    model = make_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config_path = args.checkpoint_dir / "config.json"
    history_path = args.checkpoint_dir / "history.json"
    scaler_path = args.checkpoint_dir / "embedding_scaler.npz"
    config = args_to_dict(args, extra={"model_name": "separate_encoder_sae"})
    save_json(config_path, config)
    save_scaler(scaler_path, train_dataset)

    best_invariance_score = float("inf")
    best_invariance_recon = float("inf")
    best_tradeoff_score = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args, False)
        should_probe = args.probe_every > 0 and epoch % args.probe_every == 0
        if should_probe:
            val_metrics.update(
                run_posthoc_invariance_probe(
                    model,
                    train_dataset,
                    probe_eval_datasets,
                    probe_train_indices,
                    probe_eval_indices,
                    device,
                    args,
                )
            )
            val_metrics["posthoc_tradeoff_score"] = (
                val_metrics["recon_mse"]
                + args.posthoc_tradeoff_alpha * val_metrics["posthoc_z_good_color_gap"]
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
            f"val_badcon={val_metrics['badcon_loss']:.6f} | "
            f"val_good_color_acc={val_metrics['good_color_acc']:.4f} "
            f"val_bad_color_acc={val_metrics['bad_color_acc']:.4f} | "
            f"val_z_good_active={val_metrics['z_good_active_frac']:.4f} "
            f"({val_metrics['z_good_active_count']:.1f}/{args.good_latent_dim}) "
            f"val_z_bad_active={val_metrics['z_bad_active_frac']:.4f} "
            f"({val_metrics['z_bad_active_count']:.1f}/{args.bad_latent_dim})"
        )
        if should_probe:
            message += (
                " | "
                f"posthoc_z_good_color_acc={val_metrics['posthoc_z_good_color_acc']:.4f} "
                f"posthoc_gap={val_metrics['posthoc_z_good_color_gap']:.4f} "
                f"posthoc_tradeoff={val_metrics['posthoc_tradeoff_score']:.6f}"
            )
        print(message)

        if should_probe:
            invariance_score = val_metrics["posthoc_z_good_color_gap"]
            tradeoff_score = val_metrics["posthoc_tradeoff_score"]
            recon_mse = val_metrics["recon_mse"]
            improves_invariance = invariance_score < best_invariance_score
            ties_invariance = np.isclose(invariance_score, best_invariance_score)
            improves_recon_tiebreak = ties_invariance and recon_mse < best_invariance_recon
            if improves_invariance or improves_recon_tiebreak:
                best_invariance_score = invariance_score
                best_invariance_recon = recon_mse
                save_checkpoint(
                    args.checkpoint_dir / "best_linear_posthoc_invariance.pt",
                    model,
                    optimizer,
                    args,
                    epoch,
                    train_metrics,
                    val_metrics,
                )
                print(
                    "  saved new best linear post-hoc invariance checkpoint to "
                    f"{args.checkpoint_dir / 'best_linear_posthoc_invariance.pt'}"
                )

            if tradeoff_score < best_tradeoff_score:
                best_tradeoff_score = tradeoff_score
                save_checkpoint(
                    args.checkpoint_dir / "best_linear_posthoc_tradeoff.pt",
                    model,
                    optimizer,
                    args,
                    epoch,
                    train_metrics,
                    val_metrics,
                )
                print(
                    "  saved new best linear post-hoc tradeoff checkpoint to "
                    f"{args.checkpoint_dir / 'best_linear_posthoc_tradeoff.pt'}"
                )

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
