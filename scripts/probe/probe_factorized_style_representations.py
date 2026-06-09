"""Probe each stage of a factorized split-SAE bad/style pathway.

The factorized decoder exposes three distinct representations:

1. raw z_bad from the bad encoder,
2. style logits produced by the style selector,
3. style probabilities consumed by the decoder.

This script fits the same linear digit, digit-group, and color probes to each
representation. Comparing them shows whether information present in raw z_bad
survives into the decoder-facing style code.
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_sae.utils.checkpoints import load_checkpoint
from toy_sae.utils.embeddings import EmbeddingDataset, load_scaler
from toy_sae.utils.script_utils import args_to_dict, command_string, save_json
from toy_sae.utils.split_sae_loading import infer_model_family, load_split_sae_model
from toy_sae.utils.torch_utils import get_device


DEFAULT_EVAL_SPLITS = [
    "split_val_biased",
    "test_id_biased",
    "test_balanced",
    "test_reversed",
]
REPRESENTATION_NAMES = [
    "z_bad",
    "style_logits",
    "style_probabilities",
]
LABEL_NAMES = [
    "digits",
    "digit_groups",
    "colors",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=Path("data/base_ae_embeddings"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--train-split", default="split_train_biased")
    parser.add_argument("--eval-splits", nargs="+", default=DEFAULT_EVAL_SPLITS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_args(args):
    if args.scaler is None:
        args.scaler = args.checkpoint.parent / "embedding_scaler.npz"
    if args.out_dir is None:
        run_name = f"{args.checkpoint.parent.name}_{args.checkpoint.stem}"
        args.out_dir = Path("outputs/factorized_style_probes") / run_name

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Missing factorized split-SAE checkpoint: {args.checkpoint}"
        )
    if not args.scaler.exists():
        raise FileNotFoundError(f"Missing embedding scaler: {args.scaler}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive")


def load_split(embedding_dir, split_name, mean, std):
    path = embedding_dir / f"{split_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding split: {path}")
    return EmbeddingDataset(path, mean, std)


def representation_summary(values):
    return {
        "dimension": int(values.shape[1]),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "mean_abs": float(np.abs(values).mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def extract_representations(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    batches = {name: [] for name in REPRESENTATION_NAMES}
    total_examples = 0
    style_correct = 0
    style_entropy_sum = 0.0
    style_max_probability_sum = 0.0
    style_usage_sum = torch.zeros(model.num_styles, dtype=torch.float64)

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            embeddings = batch["embedding"].to(device)
            colors = batch["color"].to(device)
            _, z_bad = model.encode(embeddings)
            style_logits, style_probabilities = model.decode_style(z_bad)

            current_batch_size = embeddings.shape[0]
            total_examples += current_batch_size
            style_correct += (
                style_logits.argmax(dim=1) == colors
            ).sum().item()

            entropy = -(
                style_probabilities
                * style_probabilities.clamp_min(1e-8).log()
            ).sum(dim=1)
            style_entropy_sum += entropy.sum().item()
            style_max_probability_sum += (
                style_probabilities.max(dim=1).values.sum().item()
            )
            style_usage_sum += (
                style_probabilities.sum(dim=0).detach().cpu().double()
            )

            batches["z_bad"].append(z_bad.cpu().numpy())
            batches["style_logits"].append(style_logits.cpu().numpy())
            batches["style_probabilities"].append(
                style_probabilities.cpu().numpy()
            )

    representations = {
        name: np.concatenate(batches[name], axis=0).astype(np.float32)
        for name in REPRESENTATION_NAMES
    }
    summaries = {
        name: representation_summary(values)
        for name, values in representations.items()
    }
    z_bad = representations["z_bad"]
    summaries["z_bad"].update(
        {
            "active_fraction": float((z_bad > 0).mean()),
            "active_count": float((z_bad > 0).mean() * z_bad.shape[1]),
        }
    )

    return {
        **representations,
        "digits": dataset.digits,
        "digit_groups": dataset.digit_groups,
        "colors": dataset.colors,
        "diagnostics": {
            "style_selection_accuracy": float(style_correct / total_examples),
            "style_entropy": float(style_entropy_sum / total_examples),
            "style_max_probability": float(
                style_max_probability_sum / total_examples
            ),
            "style_usage": [
                float(value / total_examples)
                for value in style_usage_sum.tolist()
            ],
        },
        "representation_summary": summaries,
    }


def make_probe(max_iter, seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def fit_probes(train_data, max_iter, seed):
    probes = {}
    for representation_name in REPRESENTATION_NAMES:
        probes[representation_name] = {}
        for label_name in LABEL_NAMES:
            probe = make_probe(max_iter, seed)
            probe.fit(
                train_data[representation_name],
                train_data[label_name],
            )
            probes[representation_name][label_name] = probe
    return probes


def evaluate_probes(probes, data):
    results = {}
    for representation_name in REPRESENTATION_NAMES:
        results[representation_name] = {}
        for label_name in LABEL_NAMES:
            predictions = probes[representation_name][label_name].predict(
                data[representation_name]
            )
            labels = data[label_name]
            results[representation_name][label_name] = {
                "accuracy": float(accuracy_score(labels, predictions)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, predictions)
                ),
            }
    return results


def print_results(results):
    print("Style pathway diagnostics")
    print(
        "split | style_selection_acc | style_entropy | "
        "style_max_probability | style_usage"
    )
    print("--- | ---: | ---: | ---: | ---")
    for split_name, split_results in results.items():
        diagnostics = split_results["diagnostics"]
        usage = ", ".join(
            f"{value:.4f}" for value in diagnostics["style_usage"]
        )
        print(
            f"{split_name} | "
            f"{diagnostics['style_selection_accuracy']:.4f} | "
            f"{diagnostics['style_entropy']:.6g} | "
            f"{diagnostics['style_max_probability']:.6f} | "
            f"[{usage}]"
        )

    print("")
    print("Factorized style representation probes")
    print(
        "split | representation | digit_acc | digit_bal_acc | "
        "group_acc | group_bal_acc | color_acc | color_bal_acc"
    )
    print("--- | --- | ---: | ---: | ---: | ---: | ---: | ---:")
    for split_name, split_results in results.items():
        probe_results = split_results["probe_accuracy"]
        for representation_name in REPRESENTATION_NAMES:
            row = probe_results[representation_name]
            print(
                f"{split_name} | {representation_name} | "
                f"{row['digits']['accuracy']:.4f} | "
                f"{row['digits']['balanced_accuracy']:.4f} | "
                f"{row['digit_groups']['accuracy']:.4f} | "
                f"{row['digit_groups']['balanced_accuracy']:.4f} | "
                f"{row['colors']['accuracy']:.4f} | "
                f"{row['colors']['balanced_accuracy']:.4f}"
            )

    print("")
    print(
        "Interpret digit/group results on biased splits cautiously: style "
        "representations can predict those labels indirectly through the "
        "training color correlation. test_balanced is the primary global "
        "check for information beyond color."
    )


def save_csv(path, results):
    fieldnames = [
        "split",
        "representation",
        "digit_accuracy",
        "digit_balanced_accuracy",
        "digit_group_accuracy",
        "digit_group_balanced_accuracy",
        "color_accuracy",
        "color_balanced_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, split_results in results.items():
            for representation_name in REPRESENTATION_NAMES:
                row = split_results["probe_accuracy"][representation_name]
                writer.writerow(
                    {
                        "split": split_name,
                        "representation": representation_name,
                        "digit_accuracy": row["digits"]["accuracy"],
                        "digit_balanced_accuracy": row["digits"][
                            "balanced_accuracy"
                        ],
                        "digit_group_accuracy": row["digit_groups"]["accuracy"],
                        "digit_group_balanced_accuracy": row["digit_groups"][
                            "balanced_accuracy"
                        ],
                        "color_accuracy": row["colors"]["accuracy"],
                        "color_balanced_accuracy": row["colors"][
                            "balanced_accuracy"
                        ],
                    }
                )


def main():
    args = parse_args()
    resolve_args(args)

    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint)
    model_family = infer_model_family(checkpoint)
    if model_family != "factorized_style":
        raise ValueError(
            "This probe requires a factorized_style checkpoint; "
            f"inferred {model_family!r}"
        )

    model = load_split_sae_model(
        checkpoint,
        device,
        model_family="factorized_style",
    )
    mean, std = load_scaler(args.scaler)

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Scaler: {args.scaler}")
    print(f"Probe train split: {args.train_split}")
    print(
        "Representations: raw z_bad, style logits, decoder-facing "
        "style probabilities"
    )

    train_dataset = load_split(
        args.embedding_dir,
        args.train_split,
        mean,
        std,
    )
    train_data = extract_representations(
        model,
        train_dataset,
        args.batch_size,
        args.num_workers,
        device,
    )
    probes = fit_probes(train_data, args.max_iter, args.seed)

    all_splits = [args.train_split]
    for split_name in args.eval_splits:
        if split_name not in all_splits:
            all_splits.append(split_name)

    results = {}
    for split_name in all_splits:
        if split_name == args.train_split:
            data = train_data
        else:
            dataset = load_split(
                args.embedding_dir,
                split_name,
                mean,
                std,
            )
            data = extract_representations(
                model,
                dataset,
                args.batch_size,
                args.num_workers,
                device,
            )

        results[split_name] = {
            "diagnostics": data["diagnostics"],
            "representation_summary": data["representation_summary"],
            "probe_accuracy": evaluate_probes(probes, data),
        }

    print("")
    print_results(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "style_representation_probe_results.json"
    csv_path = args.out_dir / "style_representation_probe_accuracy.csv"
    payload = {
        "command": command_string(),
        "checkpoint": str(args.checkpoint),
        "model": model_family,
        "checkpoint_args": checkpoint["args"],
        "probe_args": args_to_dict(args),
        "representations": REPRESENTATION_NAMES,
        "labels": LABEL_NAMES,
        "results": results,
    }
    save_json(json_path, payload)
    save_csv(csv_path, results)

    print("")
    print(f"Saved detailed results to {json_path}")
    print(f"Saved probe accuracy table to {csv_path}")


if __name__ == "__main__":
    main()
