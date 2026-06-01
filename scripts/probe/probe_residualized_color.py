"""Probe color after removing linear digit/group subspaces from split-SAE latents.

This diagnostic asks whether color is recoverable from a latent only because the
latent contains digit or digit-group information. It fits a linear nuisance
classifier on the training split, removes the row-space of that classifier from
the latent, and then trains fresh post-hoc probes on the residualized features.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_sae.utils.checkpoints import load_checkpoint, resolve_checkpoint_path
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
LATENT_NAMES = ["z_good", "z_bad"]
RESIDUALIZATIONS = ["none", "digit_group", "digit"]
LABELS = {
    "colors": "color",
    "digit_groups": "digit_group",
    "digits": "digit",
}
NUISANCE_LABEL_KEYS = {
    "digit_group": "digit_groups",
    "digit": "digits",
}


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
        help="Checkpoint name inside --checkpoint-dir. The .pt suffix is optional.",
    )
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/residualized_color_probes"),
    )
    parser.add_argument("--train-split", default="split_train_biased")
    parser.add_argument("--eval-splits", nargs="+", default=DEFAULT_EVAL_SPLITS)
    parser.add_argument("--latents", nargs="+", choices=LATENT_NAMES, default=LATENT_NAMES)
    parser.add_argument(
        "--residualizations",
        nargs="+",
        choices=RESIDUALIZATIONS,
        default=RESIDUALIZATIONS,
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--svd-tol",
        type=float,
        default=1e-7,
        help="Relative tolerance for choosing the nuisance row-space rank.",
    )
    return parser.parse_args()


def load_split(embedding_dir, split_name, mean, std):
    path = embedding_dir / f"{split_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding split: {path}")
    return EmbeddingDataset(path, mean, std)


def extract_latents(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(
        dataset,
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
        "digits": dataset.digits,
        "digit_groups": dataset.digit_groups,
        "colors": dataset.colors,
    }


def make_probe(max_iter, seed, *, class_weight="balanced"):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
            class_weight=class_weight,
        ),
    )


def evaluate_labels(labels):
    labels = np.asarray(labels)
    values, counts = np.unique(labels, return_counts=True)
    count_dict = {str(int(value)): int(count) for value, count in zip(values, counts)}
    majority = int(counts.max()) if counts.size else 0
    return {
        "num_examples": int(labels.shape[0]),
        "class_counts": count_dict,
        "majority_baseline": float(majority / labels.shape[0])
        if labels.shape[0]
        else None,
    }


def evaluate_probe(probe, features, labels):
    info = evaluate_labels(labels)
    if len(labels) == 0:
        info["accuracy"] = None
        info["balanced_accuracy"] = None
        info["accuracy_minus_majority_baseline"] = None
        return info

    predictions = probe.predict(features)
    accuracy = float(accuracy_score(labels, predictions))
    info["accuracy"] = accuracy
    info["balanced_accuracy"] = (
        float(balanced_accuracy_score(labels, predictions))
        if len(np.unique(labels)) > 1
        else None
    )
    info["accuracy_minus_majority_baseline"] = (
        accuracy - info["majority_baseline"]
        if info["majority_baseline"] is not None
        else None
    )
    return info


def row_space_basis(weight_matrix, svd_tol):
    weights = np.asarray(weight_matrix, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    if weights.shape[0] > 1:
        weights = weights - weights.mean(axis=0, keepdims=True)

    _, singular_values, vh = np.linalg.svd(weights, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return np.empty((0, weights.shape[1]), dtype=np.float64), singular_values

    rank = int((singular_values > singular_values[0] * svd_tol).sum())
    return vh[:rank], singular_values


def transform_with_basis(features, scaler, basis):
    scaled = scaler.transform(features).astype(np.float64)
    if basis.shape[0] == 0:
        return scaled.astype(np.float32)
    projection = (scaled @ basis.T) @ basis
    return (scaled - projection).astype(np.float32)


def fit_residualizer(train_features, nuisance_labels, nuisance_name, args):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    nuisance_probe = LogisticRegression(
        max_iter=args.max_iter,
        random_state=args.seed,
        solver="lbfgs",
        class_weight="balanced",
    )
    nuisance_probe.fit(train_scaled, nuisance_labels)
    basis, singular_values = row_space_basis(nuisance_probe.coef_, args.svd_tol)
    transformed_train = transform_with_basis(train_features, scaler, basis)

    before_info = evaluate_labels(nuisance_labels)
    before_predictions = nuisance_probe.predict(train_scaled)
    before_info["accuracy"] = float(accuracy_score(nuisance_labels, before_predictions))
    before_info["balanced_accuracy"] = float(
        balanced_accuracy_score(nuisance_labels, before_predictions)
    )

    return {
        "name": nuisance_name,
        "scaler": scaler,
        "basis": basis,
        "train_features": transformed_train,
        "info": {
            "nuisance_label": nuisance_name,
            "nuisance_classes": [int(value) for value in nuisance_probe.classes_],
            "weight_shape": list(nuisance_probe.coef_.shape),
            "row_space_rank": int(basis.shape[0]),
            "singular_values": singular_values.tolist(),
            "train_nuisance_probe_before_residualization": before_info,
            "train_fresh_nuisance_probe_after_residualization": None,
        },
    }


def identity_residualizer(train_features):
    return {
        "name": "none",
        "scaler": None,
        "basis": None,
        "train_features": train_features,
        "info": {
            "nuisance_label": None,
            "row_space_rank": 0,
            "train_nuisance_probe_before_residualization": None,
            "train_fresh_nuisance_probe_after_residualization": None,
        },
    }


def transform_residualizer(features, residualizer):
    if residualizer["name"] == "none":
        return features
    return transform_with_basis(features, residualizer["scaler"], residualizer["basis"])


def fit_label_probes(train_features, train_latents, max_iter, seed):
    probes = {}
    for label_key in LABELS:
        probe = make_probe(max_iter, seed)
        probe.fit(train_features, train_latents[label_key])
        probes[label_key] = probe
    return probes


def evaluate_label_probes(probes, features, latents):
    return {
        label_key: evaluate_probe(probe, features, latents[label_key])
        for label_key, probe in probes.items()
    }


def feature_summary(features):
    values = np.asarray(features)
    return {
        "mean_abs": float(np.abs(values).mean()),
        "mean_l2_norm": float(np.linalg.norm(values, axis=1).mean()),
        "positive_fraction": float((values > 0).mean()),
        "positive_count": float((values > 0).mean() * values.shape[1]),
    }


def build_residualizers(train_features, train_latents, args):
    residualizers = {}
    if "none" in args.residualizations:
        residualizers["none"] = identity_residualizer(train_features)
    if "digit_group" in args.residualizations:
        residualizers["digit_group"] = fit_residualizer(
            train_features,
            train_latents["digit_groups"],
            "digit_group",
            args,
        )
    if "digit" in args.residualizations:
        residualizers["digit"] = fit_residualizer(
            train_features,
            train_latents["digits"],
            "digit",
            args,
        )
    return residualizers


def run_residualized_probes(train_latents, eval_latents_by_split, args):
    results = {}
    residualizer_info = {}
    for latent_name in args.latents:
        results[latent_name] = {}
        residualizer_info[latent_name] = {}
        residualizers = build_residualizers(train_latents[latent_name], train_latents, args)

        for residualization_name, residualizer in residualizers.items():
            train_features = residualizer["train_features"]
            residualizer_info[latent_name][residualization_name] = residualizer["info"]
            probes = fit_label_probes(train_features, train_latents, args.max_iter, args.seed)
            train_probe_accuracy = evaluate_label_probes(
                probes,
                train_features,
                train_latents,
            )
            nuisance_key = NUISANCE_LABEL_KEYS.get(residualization_name)
            if nuisance_key is not None:
                residualizer_info[latent_name][residualization_name][
                    "train_fresh_nuisance_probe_after_residualization"
                ] = train_probe_accuracy[nuisance_key]

            results[latent_name][residualization_name] = {}
            all_splits = {args.train_split: train_latents, **eval_latents_by_split}
            for split_name, latents in all_splits.items():
                features = (
                    train_features
                    if split_name == args.train_split
                    else transform_residualizer(latents[latent_name], residualizer)
                )
                results[latent_name][residualization_name][split_name] = {
                    "feature_summary": feature_summary(features),
                    "probe_accuracy": evaluate_label_probes(probes, features, latents),
                }

    return results, residualizer_info


def compute_color_drop_summary(results, eval_splits):
    summary = {}
    for latent_name, latent_results in results.items():
        summary[latent_name] = {}
        if "none" not in latent_results:
            continue
        for split_name in eval_splits:
            baseline = latent_results["none"][split_name]["probe_accuracy"]["colors"]
            baseline_bal = baseline["balanced_accuracy"]
            baseline_acc = baseline["accuracy"]
            row = {
                "color_balanced_accuracy_none": baseline_bal,
                "color_accuracy_none": baseline_acc,
            }
            for residualization_name in ["digit_group", "digit"]:
                if residualization_name not in latent_results:
                    continue
                color_result = latent_results[residualization_name][split_name][
                    "probe_accuracy"
                ]["colors"]
                residual_bal = color_result["balanced_accuracy"]
                residual_acc = color_result["accuracy"]
                row[f"color_balanced_accuracy_after_{residualization_name}"] = (
                    residual_bal
                )
                row[f"color_accuracy_after_{residualization_name}"] = residual_acc
                row[f"color_balanced_accuracy_drop_after_{residualization_name}"] = (
                    None
                    if baseline_bal is None or residual_bal is None
                    else float(baseline_bal - residual_bal)
                )
                row[f"color_accuracy_drop_after_{residualization_name}"] = (
                    None
                    if baseline_acc is None or residual_acc is None
                    else float(baseline_acc - residual_acc)
                )
            summary[latent_name][split_name] = row
    return summary


def compute_nuisance_recovery_summary(results, residualizer_info, eval_splits):
    summary = {}
    for latent_name, latent_results in results.items():
        summary[latent_name] = {}
        for residualization_name, nuisance_key in NUISANCE_LABEL_KEYS.items():
            if residualization_name not in latent_results:
                continue

            info = residualizer_info[latent_name][residualization_name]
            summary[latent_name][residualization_name] = {
                "nuisance_label": info["nuisance_label"],
                "row_space_rank": info["row_space_rank"],
                "train_nuisance_probe_before_residualization": info[
                    "train_nuisance_probe_before_residualization"
                ],
                "splits": {},
            }
            for split_name in eval_splits:
                nuisance_result = latent_results[residualization_name][split_name][
                    "probe_accuracy"
                ][nuisance_key]
                summary[latent_name][residualization_name]["splits"][split_name] = {
                    "nuisance_accuracy_after_residualization": nuisance_result[
                        "accuracy"
                    ],
                    "nuisance_balanced_accuracy_after_residualization": (
                        nuisance_result["balanced_accuracy"]
                    ),
                    "majority_baseline": nuisance_result["majority_baseline"],
                }
    return summary


def format_float(value):
    return "n/a" if value is None else f"{value:.4f}"


def print_residualizer_ranks(residualizer_info):
    print("")
    print("Residualizer ranks")
    print("latent | residualization | nuisance | rank")
    print("--- | --- | --- | ---:")
    for latent_name, latent_info in residualizer_info.items():
        for residualization_name, info in latent_info.items():
            if residualization_name == "none":
                continue
            nuisance = info["nuisance_label"]
            rank = info["row_space_rank"]
            print(f"{latent_name} | {residualization_name} | {nuisance} | {rank}")


def print_results(results, eval_splits):
    print("")
    print("Residualized probe summary")
    print("(digit/group columns are fresh nuisance probes after residualization)")
    print(
        "latent | residualization | split | digit_bal | group_bal | color_bal | color_acc"
    )
    print("--- | --- | --- | ---: | ---: | ---: | ---:")
    for latent_name, latent_results in results.items():
        for residualization_name, residualized_results in latent_results.items():
            for split_name in eval_splits:
                row = residualized_results[split_name]["probe_accuracy"]
                digit_bal = row["digits"]["balanced_accuracy"]
                group_bal = row["digit_groups"]["balanced_accuracy"]
                color_bal = row["colors"]["balanced_accuracy"]
                color_acc = row["colors"]["accuracy"]
                print(
                    f"{latent_name} | {residualization_name} | {split_name} | "
                    f"{format_float(digit_bal)} | {format_float(group_bal)} | "
                    f"{format_float(color_bal)} | {format_float(color_acc)}"
                )


def print_nuisance_recovery_summary(nuisance_recovery_summary, eval_splits):
    print("")
    print("Nuisance recovery after residualization")
    print(
        "latent | residualization | nuisance | rank | split | "
        "nuisance_bal_after | nuisance_acc_after"
    )
    print("--- | --- | --- | ---: | --- | ---: | ---:")
    for latent_name, latent_summary in nuisance_recovery_summary.items():
        for residualization_name, info in latent_summary.items():
            rank = info["row_space_rank"]
            nuisance = info["nuisance_label"]
            split_rows = info["splits"]
            for split_name in eval_splits:
                row = split_rows.get(split_name)
                if row is None:
                    continue
                print(
                    f"{latent_name} | {residualization_name} | {nuisance} | "
                    f"{rank} | {split_name} | "
                    f"{format_float(row['nuisance_balanced_accuracy_after_residualization'])} | "
                    f"{format_float(row['nuisance_accuracy_after_residualization'])}"
                )


def print_color_drop_summary(color_drop_summary, eval_splits):
    print("")
    print("Color balanced-accuracy drop after residualization")
    print(
        "latent | split | color_bal_none | color_bal_remove_group | "
        "color_bal_remove_digit | drop_after_group | drop_after_digit"
    )
    print("--- | --- | ---: | ---: | ---: | ---: | ---:")
    for latent_name, latent_summary in color_drop_summary.items():
        for split_name in eval_splits:
            row = latent_summary.get(split_name)
            if row is None:
                continue
            print(
                f"{latent_name} | {split_name} | "
                f"{format_float(row.get('color_balanced_accuracy_none'))} | "
                f"{format_float(row.get('color_balanced_accuracy_after_digit_group'))} | "
                f"{format_float(row.get('color_balanced_accuracy_after_digit'))} | "
                f"{format_float(row.get('color_balanced_accuracy_drop_after_digit_group'))} | "
                f"{format_float(row.get('color_balanced_accuracy_drop_after_digit'))}"
            )


def main():
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_dir, args.checkpoint_name)
    if args.scaler is None:
        args.scaler = checkpoint_path.parent / "embedding_scaler.npz"

    if not args.scaler.exists():
        raise FileNotFoundError(f"Missing embedding scaler: {args.scaler}")

    device = get_device()
    checkpoint = load_checkpoint(checkpoint_path)
    model_family = infer_model_family(checkpoint)
    model = load_split_sae_model(checkpoint, device, model_family=model_family)
    model_name = checkpoint["args"].get("model_name", model_family)
    mean, std = load_scaler(args.scaler)

    print(f"Using device: {device}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model family: {model_family}")
    print(f"Model: {model_name}")
    print(f"Scaler: {args.scaler}")
    print(f"Train split: {args.train_split}")
    print(f"Eval splits: {', '.join(args.eval_splits)}")

    train_dataset = load_split(args.embedding_dir, args.train_split, mean, std)
    train_latents = extract_latents(
        model,
        train_dataset,
        args.batch_size,
        args.num_workers,
        device,
    )

    eval_latents_by_split = {}
    for split_name in args.eval_splits:
        if split_name == args.train_split:
            continue
        dataset = load_split(args.embedding_dir, split_name, mean, std)
        eval_latents_by_split[split_name] = extract_latents(
            model,
            dataset,
            args.batch_size,
            args.num_workers,
            device,
        )

    results, residualizer_info = run_residualized_probes(
        train_latents,
        eval_latents_by_split,
        args,
    )

    output_dir = args.out_dir / args.checkpoint_dir.name
    results_path = output_dir / "residualized_color_probe_results.json"
    eval_order = [args.train_split] + [
        split_name for split_name in args.eval_splits if split_name != args.train_split
    ]
    color_drop_summary = compute_color_drop_summary(results, eval_order)
    nuisance_recovery_summary = compute_nuisance_recovery_summary(
        results,
        residualizer_info,
        eval_order,
    )
    print_residualizer_ranks(residualizer_info)
    print_results(results, eval_order)
    print_nuisance_recovery_summary(nuisance_recovery_summary, eval_order)
    print_color_drop_summary(color_drop_summary, eval_order)

    payload = {
        "command": command_string(),
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint": str(checkpoint_path),
        "model_family": model_family,
        "model": model_name,
        "checkpoint_args": checkpoint["args"],
        "probe_args": args_to_dict(args),
        "scaler": str(args.scaler),
        "embedding_dir": str(args.embedding_dir),
        "train_split": args.train_split,
        "eval_splits": args.eval_splits,
        "residualizer_info": residualizer_info,
        "nuisance_recovery_summary": nuisance_recovery_summary,
        "color_drop_summary": color_drop_summary,
        "results": results,
    }
    save_json(results_path, payload)
    print("")
    print(f"Saved residualized color probe results to {results_path}")


if __name__ == "__main__":
    main()
