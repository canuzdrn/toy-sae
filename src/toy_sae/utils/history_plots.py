"""Utilities for plotting metrics from training history files."""

import json
import os
from pathlib import Path
import re


def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def load_history(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing history file: {path}")
    return json.loads(path.read_text())


def metric_values(history, split, metric):
    epochs = []
    values = []

    for row in history:
        if split not in row:
            continue
        if metric not in row[split]:
            continue
        epochs.append(row["epoch"])
        values.append(row[split][metric])

    return epochs, values


def available_metrics(history):
    metrics = set()
    for row in history:
        for split in ["train", "val"]:
            if split not in row:
                continue
            metrics.update(row[split].keys())
    return sorted(metrics)


def default_output_path(history_path, metric, split, out_dir=None):
    history_path = Path(history_path)
    if out_dir is None:
        out_dir = history_path.parent / "plots"
    else:
        out_dir = Path(out_dir)

    name = safe_filename(metric)
    if split != "both":
        name = f"{safe_filename(split)}_{name}"
    return out_dir / f"{name}.png"


def get_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_history_metric(history_path, metric, split="both", out_path=None, out_dir=None):
    history_path = Path(history_path)
    history = load_history(history_path)
    if out_path is None:
        out_path = default_output_path(history_path, metric, split, out_dir)
    else:
        out_path = Path(out_path)

    splits = ["train", "val"] if split == "both" else [split]
    plotted_anything = False

    plt = get_pyplot()
    plt.figure(figsize=(8, 5))
    for split_name in splits:
        epochs, values = metric_values(history, split_name, metric)
        if not values:
            continue
        plt.plot(epochs, values, label=split_name)
        plotted_anything = True

    if not plotted_anything:
        plt.close()
        raise ValueError(
            f"Metric '{metric}' was not found for split '{split}'. "
            f"Available metrics: {available_metrics(history)}"
        )

    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(metric)
    plt.grid(True, alpha=0.3)
    if split == "both":
        plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def plot_history_metrics(history_path, metrics=None, split="both", out_dir=None):
    history = load_history(history_path)
    if metrics is None:
        metrics = available_metrics(history)

    saved_paths = []
    skipped_metrics = []
    for metric in metrics:
        try:
            saved_paths.append(plot_history_metric(history_path, metric, split=split, out_dir=out_dir))
        except ValueError:
            skipped_metrics.append(metric)

    return {
        "saved_paths": saved_paths,
        "skipped_metrics": skipped_metrics,
    }
