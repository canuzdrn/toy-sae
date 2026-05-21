#!/usr/bin/env python
"""Visualize color separability in split-SAE latents.

This script is a qualitative debugging tool. It loads a trained split-SAE run
folder, extracts z_good and/or z_bad for one or more embedding splits, projects
the latents to 2D and 3D, and saves scatter plots colored by true labels.

The most important plots are the within-digit-group color plots. If red and
green examples still separate inside one fixed digit group, that is visual
evidence that the latent geometry still contains direct color information.
"""

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

NUMBA_CACHE_DIR = Path(os.environ.get("NUMBA_CACHE_DIR", "/private/tmp/numba_cache"))
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))
import umap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_sae.utils.checkpoints import load_checkpoint, resolve_checkpoint_path
from toy_sae.utils.embeddings import EmbeddingDataset, load_scaler
from toy_sae.utils.script_utils import args_to_dict, command_string, save_json
from toy_sae.utils.split_sae_loading import infer_model_family, load_split_sae_model
from toy_sae.utils.torch_utils import get_device


DEFAULT_SPLITS = ["test_balanced"]
LATENT_NAMES = ["z_good", "z_bad"]
PROJECTION_METHODS = ["pca", "tsne", "umap"]
PROJECTION_DIMS = [2, 3]
COLOR_NAMES = {0: "red", 1: "green"}
COLOR_PALETTE = {0: "#d62728", 1: "#2ca02c"}
DIGIT_GROUP_NAMES = {0: "digits 0-4", 1: "digits 5-9"}
DIGIT_GROUP_PALETTE = {0: "#1f77b4", 1: "#ff7f0e"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/base_ae_embeddings"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Run folder containing the checkpoint and embedding_scaler.npz.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best_recon",
        help=(
            "Checkpoint name inside --checkpoint-dir, such as best_recon, "
            "best_total, or latest. The .pt suffix is optional."
        ),
    )
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/latent_color_separability"))
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--max-examples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--digits", nargs="*", type=int, default=[])
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--no-balanced-sample", action="store_true")
    parser.add_argument("--no-standardize-latents", action="store_true")
    return parser.parse_args()


def load_split(embedding_dir, split_name, mean, std):
    path = embedding_dir / f"{split_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding split: {path}")
    return EmbeddingDataset(path, mean, std)


def sample_indices(colors, max_examples, seed, balanced):
    num_examples = len(colors)
    if max_examples is None or max_examples <= 0 or max_examples >= num_examples:
        return np.arange(num_examples)

    rng = np.random.default_rng(seed)
    if not balanced:
        return np.sort(rng.choice(num_examples, size=max_examples, replace=False))

    class_indices = []
    for color in sorted(np.unique(colors)):
        color_indices = np.flatnonzero(colors == color)
        class_indices.append(color_indices)

    if len(class_indices) < 2 or any(len(indices) == 0 for indices in class_indices):
        return np.sort(rng.choice(num_examples, size=max_examples, replace=False))

    per_class = max_examples // len(class_indices)
    sampled = []
    for indices in class_indices:
        take = min(per_class, len(indices))
        sampled.append(rng.choice(indices, size=take, replace=False))

    sampled = np.concatenate(sampled)
    if len(sampled) < max_examples:
        remaining = np.setdiff1d(np.arange(num_examples), sampled, assume_unique=False)
        extra_count = min(max_examples - len(sampled), len(remaining))
        if extra_count > 0:
            sampled = np.concatenate(
                [sampled, rng.choice(remaining, size=extra_count, replace=False)]
            )
    return np.sort(sampled)


def extract_latents(model, dataset, indices, batch_size, num_workers, device):
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    z_good_batches = []
    z_bad_batches = []
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embedding"].to(device)
            z_good, z_bad = model.encode(embeddings)
            z_good_batches.append(z_good.cpu().numpy())
            z_bad_batches.append(z_bad.cpu().numpy())

    return {
        "z_good": np.concatenate(z_good_batches, axis=0).astype(np.float32),
        "z_bad": np.concatenate(z_bad_batches, axis=0).astype(np.float32),
        "digits": dataset.digits[indices],
        "digit_groups": dataset.digit_groups[indices],
        "colors": dataset.colors[indices],
    }


def make_reducer(method, values, n_components, args):
    if method == "pca":
        return PCA(n_components=n_components, random_state=args.seed)
    if method == "tsne":
        n_samples = values.shape[0]
        if n_samples < 4:
            raise ValueError("t-SNE needs at least 4 samples")
        perplexity = min(args.tsne_perplexity, max(2.0, (n_samples - 1) / 3.0))
        return TSNE(
            n_components=n_components,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=args.seed,
        )
    if method == "umap":
        return umap.UMAP(n_components=n_components, random_state=args.seed)
    raise ValueError(f"Unknown projection method: {method}")


def project_latent(values, method, n_components, args):
    if not args.no_standardize_latents:
        values = StandardScaler().fit_transform(values)
    reducer = make_reducer(method, values, n_components, args)
    coords = reducer.fit_transform(values)
    metadata = {"n_components": n_components}
    if method == "pca":
        explained_variance_ratio = [
            float(value) for value in reducer.explained_variance_ratio_
        ]
        metadata["explained_variance_ratio"] = explained_variance_ratio
        metadata["captured_variance_ratio"] = float(sum(explained_variance_ratio))
    return coords.astype(np.float32), metadata


def safe_silhouette(coords, labels):
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2:
        return None
    if len(labels) <= len(unique):
        return None
    try:
        return float(silhouette_score(coords, labels))
    except ValueError:
        return None


def scatter_by_labels(ax, coords, labels, palette, names, title):
    is_3d = coords.shape[1] == 3
    for value in sorted(np.unique(labels)):
        mask = labels == value
        label = names.get(int(value), str(int(value)))
        color = palette.get(int(value), None)
        if is_3d:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                s=10,
                alpha=0.65,
                linewidths=0,
                label=label,
                color=color,
            )
        else:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=10,
                alpha=0.65,
                linewidths=0,
                label=label,
                color=color,
            )
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    if is_3d:
        ax.set_zlabel("component 3")
    ax.legend(loc="best", markerscale=2, frameon=True)
    ax.grid(alpha=0.2)


def save_plot(coords, labels, palette, names, title, out_path):
    if coords.shape[1] == 3:
        fig = plt.figure(figsize=(7.0, 5.8), dpi=160)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=160)
    scatter_by_labels(ax, coords, labels, palette, names, title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def projection_title_label(method, projection_metadata):
    n_components = projection_metadata.get("n_components", 2)
    dim_label = f"{n_components}D"

    if method != "pca":
        method_name = "t-SNE" if method == "tsne" else method.upper()
        return f"{method_name} {dim_label}"

    captured = projection_metadata.get("captured_variance_ratio")
    if captured is None:
        return f"PCA {dim_label}"
    return f"PCA {dim_label} ({captured * 100:.1f}% variance)"


def plot_projection(
    coords,
    labels,
    split_name,
    latent_name,
    method,
    projection_metadata,
    out_dir,
    digits,
):
    saved_paths = []

    n_components = projection_metadata.get("n_components", coords.shape[1])
    dim_suffix = "" if n_components == 2 else f"_{n_components}d"
    base = f"{split_name}_{latent_name}_{method}{dim_suffix}"
    method_label = projection_title_label(method, projection_metadata)
    title_prefix = f"{split_name} {latent_name} {method_label}"

    path = out_dir / f"{base}_colored_by_color.png"
    save_plot(
        coords,
        labels["colors"],
        COLOR_PALETTE,
        COLOR_NAMES,
        f"{title_prefix}: colored by true color",
        path,
    )
    saved_paths.append(path)

    path = out_dir / f"{base}_colored_by_digit_group.png"
    save_plot(
        coords,
        labels["digit_groups"],
        DIGIT_GROUP_PALETTE,
        DIGIT_GROUP_NAMES,
        f"{title_prefix}: colored by digit group",
        path,
    )
    saved_paths.append(path)

    for digit_group in [0, 1]:
        mask = labels["digit_groups"] == digit_group
        if mask.sum() == 0:
            continue
        path = out_dir / f"{base}_digit_group_{digit_group}_colored_by_color.png"
        save_plot(
            coords[mask],
            labels["colors"][mask],
            COLOR_PALETTE,
            COLOR_NAMES,
            f"{title_prefix}: {DIGIT_GROUP_NAMES[digit_group]}, colored by color",
            path,
        )
        saved_paths.append(path)

    for digit in digits:
        mask = labels["digits"] == digit
        if mask.sum() == 0:
            continue
        path = out_dir / f"{base}_digit_{digit}_colored_by_color.png"
        save_plot(
            coords[mask],
            labels["colors"][mask],
            COLOR_PALETTE,
            COLOR_NAMES,
            f"{title_prefix}: digit {digit}, colored by color",
            path,
        )
        saved_paths.append(path)

    return saved_paths


def projection_metrics(coords, labels):
    metrics = {
        "color_silhouette_all": safe_silhouette(coords, labels["colors"]),
    }
    for digit_group in [0, 1]:
        mask = labels["digit_groups"] == digit_group
        metrics[f"color_silhouette_digit_group_{digit_group}"] = (
            safe_silhouette(coords[mask], labels["colors"][mask]) if mask.sum() else None
        )
    for digit in sorted(np.unique(labels["digits"])):
        mask = labels["digits"] == digit
        metrics[f"color_silhouette_digit_{int(digit)}"] = (
            safe_silhouette(coords[mask], labels["colors"][mask]) if mask.sum() else None
        )
    return metrics


def main():
    args = parse_args()
    device = get_device()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint_dir, args.checkpoint_name)
    run_dir = checkpoint_path.parent

    checkpoint = load_checkpoint(checkpoint_path)
    model_family = infer_model_family(checkpoint)
    model = load_split_sae_model(checkpoint, device, model_family=model_family)

    scaler = args.scaler
    if scaler is None:
        scaler = run_dir / "embedding_scaler.npz"
    mean, std = load_scaler(scaler)

    output_dir = args.out_dir / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "command": command_string(),
        "args": args_to_dict(args),
        "checkpoint_dir": str(run_dir),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_args": checkpoint.get("args", {}),
        "model_family": model_family,
        "scaler": str(scaler),
        "output_root": str(args.out_dir),
        "output_dir": str(output_dir),
        "splits": {},
    }

    print(f"Using device: {device}")
    print(f"Checkpoint directory: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model family: {model_family}")
    print(f"Scaler: {scaler}")
    print(f"Output directory: {output_dir}")

    for split_index, split_name in enumerate(args.splits):
        dataset = load_split(args.embedding_dir, split_name, mean, std)
        indices = sample_indices(
            dataset.colors,
            args.max_examples,
            args.seed + split_index,
            balanced=not args.no_balanced_sample,
        )
        latents = extract_latents(
            model,
            dataset,
            indices,
            args.batch_size,
            args.num_workers,
            device,
        )
        labels = {
            "digits": latents["digits"],
            "digit_groups": latents["digit_groups"],
            "colors": latents["colors"],
        }
        split_summary = {
            "num_examples": int(len(indices)),
            "latents": {},
        }

        for latent_name in LATENT_NAMES:
            latent_summary = {}
            for method in PROJECTION_METHODS:
                method_summary = {}
                for n_components in PROJECTION_DIMS:
                    coords, projection_metadata = project_latent(
                        latents[latent_name],
                        method,
                        n_components,
                        args,
                    )
                    method_output_dir = output_dir / method
                    saved_paths = plot_projection(
                        coords,
                        labels,
                        split_name,
                        latent_name,
                        method,
                        projection_metadata,
                        method_output_dir,
                        args.digits,
                    )
                    dim_key = f"{n_components}d"
                    method_summary[dim_key] = {
                        "projection": projection_metadata,
                        "metrics": projection_metrics(coords, labels),
                        "plots": [str(path) for path in saved_paths],
                    }
                    print(
                        f"{split_name} | {latent_name} | {method} | {dim_key}: "
                        f"saved {len(saved_paths)} plots"
                    )
                latent_summary[method] = method_summary
            split_summary["latents"][latent_name] = latent_summary
        summary["splits"][split_name] = split_summary

    summary_path = output_dir / "projection_summary.json"
    save_json(summary_path, summary)
    print(f"Saved projection summary to {summary_path}")


if __name__ == "__main__":
    main()
