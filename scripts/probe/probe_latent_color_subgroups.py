"""Train post-hoc color probes within fixed digit groups and digits.

This script is a stricter companion to the global latent probe scripts. A
global color probe can partly exploit digit/color correlations in biased
splits. Here, each probe is trained inside a fixed subgroup, such as
``digit_group = 0`` or ``digit = 7``. If color is still predictable within a
fixed digit or digit group, that is stronger evidence of direct color leakage.
"""

import argparse
import hashlib
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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
COLOR_NAMES = {0: "red", 1: "green"}
DIGIT_GROUP_NAMES = {0: "digits_0_4", 1: "digits_5_9"}


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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/latent_color_subgroup_probes"),
    )
    parser.add_argument("--train-split", default="split_train_biased")
    parser.add_argument("--eval-splits", nargs="+", default=DEFAULT_EVAL_SPLITS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-train-per-color",
        type=int,
        default=10,
        help="Skip subgroup probes with fewer than this many examples per color.",
    )
    parser.add_argument(
        "--warn-train-per-color",
        type=int,
        default=50,
        help=(
            "Mark subgroup probe training sets with fewer than this many examples "
            "for either color as low-count diagnostics. This does not skip them."
        ),
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


def subgroup_specs():
    specs = [
        {
            "key": "global",
            "type": "global",
            "value": None,
            "label": "all examples",
        }
    ]
    for digit_group in [0, 1]:
        specs.append(
            {
                "key": f"digit_group_{digit_group}",
                "type": "digit_group",
                "value": digit_group,
                "label": DIGIT_GROUP_NAMES[digit_group],
            }
        )
    for digit in range(10):
        specs.append(
            {
                "key": f"digit_{digit}",
                "type": "digit",
                "value": digit,
                "label": f"digit_{digit}",
            }
        )
    return specs


def subgroup_mask(latents, spec):
    if spec["type"] == "global":
        return np.ones_like(latents["colors"], dtype=bool)
    if spec["type"] == "digit_group":
        return latents["digit_groups"] == spec["value"]
    if spec["type"] == "digit":
        return latents["digits"] == spec["value"]
    raise ValueError(f"Unknown subgroup type: {spec['type']}")


def color_counts(colors):
    return {
        COLOR_NAMES[color]: int((colors == color).sum())
        for color in sorted(COLOR_NAMES)
    }


def subset_info(colors):
    counts = color_counts(colors)
    total = int(len(colors))
    majority = max(counts.values()) if counts else 0
    minority = min(counts.values()) if counts else 0
    return {
        "num_examples": total,
        "color_counts": counts,
        "min_color_count": int(minority),
        "max_color_count": int(majority),
        "majority_baseline": float(majority / total) if total else None,
    }


def can_fit_color_probe(colors, min_train_per_color):
    counts = color_counts(colors)
    return all(count >= min_train_per_color for count in counts.values())


def stable_seed(seed, *parts):
    text = "|".join([str(seed), *[str(part) for part in parts]])
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little") % (2**32 - 1)


def permute_colors_within_digits(colors, digits, seed, *context):
    rng = np.random.default_rng(stable_seed(seed, *context))
    colors = np.asarray(colors)
    digits = np.asarray(digits)
    permuted = colors.copy()
    for digit in np.unique(digits):
        indices = np.flatnonzero(digits == digit)
        if indices.size > 1:
            permuted[indices] = rng.permutation(permuted[indices])
    return permuted


def make_probe(max_iter, seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
            class_weight="balanced",
        ),
    )


def evaluate_color_probe(probe, features, colors):
    info = subset_info(colors)
    if len(colors) == 0:
        info["accuracy"] = None
        info["balanced_accuracy"] = None
        info["accuracy_minus_majority_baseline"] = None
        return info

    predictions = probe.predict(features)
    accuracy = float(accuracy_score(colors, predictions))
    info["accuracy"] = accuracy
    info["accuracy_minus_majority_baseline"] = (
        accuracy - info["majority_baseline"]
        if info["majority_baseline"] is not None
        else None
    )
    if len(np.unique(colors)) < 2:
        info["balanced_accuracy"] = None
    else:
        info["balanced_accuracy"] = float(balanced_accuracy_score(colors, predictions))
    return info


def color_direction_from_probe(probe):
    """Return color direction vectors from a fitted sklearn pipeline.

    The standardized direction is the logistic-regression coefficient after the
    per-probe StandardScaler. The raw direction is the equivalent direction in
    the original latent coordinates. Both directions are sign-aligned so that
    positive score points toward color label 1, which is green in this project.
    """
    scaler = probe.named_steps["standardscaler"]
    classifier = probe.named_steps["logisticregression"]
    classes = list(classifier.classes_)
    if len(classes) != 2 or 1 not in classes:
        return None

    direction = classifier.coef_[0].astype(np.float64)
    if classes.index(1) == 0:
        direction = -direction

    scale = np.maximum(scaler.scale_.astype(np.float64), 1e-12)
    return {
        "standardized": direction,
        "raw": direction / scale,
        "classes": classes,
    }


def mean_or_none(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def weighted_mean_or_none(values, weights):
    pairs = [
        (value, weight)
        for value, weight in zip(values, weights)
        if value is not None and weight is not None and weight > 0
    ]
    if not pairs:
        return None

    filtered_values = np.array([value for value, _ in pairs], dtype=np.float64)
    filtered_weights = np.array([weight for _, weight in pairs], dtype=np.float64)
    return float(np.average(filtered_values, weights=filtered_weights))


def off_diagonal_values(matrix):
    matrix = np.asarray(matrix)
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def nanmean_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def nanmin_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return None
    return float(np.min(values))


def nanmax_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return None
    return float(np.max(values))


def difference_or_none(left, right):
    if left is None or right is None:
        return None
    return float(left - right)


def matrix_to_jsonable(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return [
        [None if np.isnan(value) else float(value) for value in row]
        for row in matrix
    ]


def cosine_matrix(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def singular_energy(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    energy = singular_values**2
    if float(energy.sum()) == 0.0:
        energy_fraction = np.zeros_like(energy)
    else:
        energy_fraction = energy / energy.sum()
    components_for_90 = int(np.searchsorted(np.cumsum(energy_fraction), 0.9) + 1)
    return singular_values, energy_fraction, components_for_90


def build_direction_diagnostics(direction_vectors):
    diagnostics = {}
    for latent_name, latent_vectors in direction_vectors.items():
        diagnostics[latent_name] = {}
        for space in ["raw", "standardized"]:
            digit_vectors = latent_vectors[space]
            digits = sorted(digit_vectors)
            if len(digits) < 2:
                diagnostics[latent_name][space] = {
                    "digits": digits,
                    "available": False,
                    "reason": "fewer than two valid digit-specific color directions",
                }
                continue

            vectors = np.stack([digit_vectors[digit] for digit in digits], axis=0)
            signed_cosine = cosine_matrix(vectors)
            absolute_cosine = np.abs(signed_cosine)
            signed_offdiag = off_diagonal_values(signed_cosine)
            absolute_offdiag = off_diagonal_values(absolute_cosine)
            singular_values, energy_fraction, components_for_90 = singular_energy(vectors)

            diagnostics[latent_name][space] = {
                "digits": digits,
                "available": True,
                "direction_norms": {
                    str(digit): float(np.linalg.norm(digit_vectors[digit]))
                    for digit in digits
                },
                "signed_cosine": signed_cosine.tolist(),
                "absolute_cosine": absolute_cosine.tolist(),
                "mean_offdiag_signed_cosine": float(np.mean(signed_offdiag)),
                "mean_offdiag_absolute_cosine": float(np.mean(absolute_offdiag)),
                "min_offdiag_signed_cosine": float(np.min(signed_offdiag)),
                "max_offdiag_signed_cosine": float(np.max(signed_offdiag)),
                "min_offdiag_absolute_cosine": float(np.min(absolute_offdiag)),
                "max_offdiag_absolute_cosine": float(np.max(absolute_offdiag)),
                "singular_values": singular_values.tolist(),
                "singular_energy_fraction": energy_fraction.tolist(),
                "first_singular_energy_fraction": float(energy_fraction[0]),
                "components_for_90pct_energy": components_for_90,
            }
    return diagnostics


def plot_cosine_heatmap(matrix, digits, title, out_path, vmin, vmax, cmap):
    matrix = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(digits)), labels=digits)
    ax.set_yticks(range(len(digits)), labels=digits)
    ax.set_xlabel("digit-specific color probe")
    ax.set_ylabel("digit-specific color probe")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_direction_plots(direction_diagnostics, output_dir):
    plot_dir = output_dir / "direction_diagnostics"
    saved_paths = []
    for latent_name, latent_diagnostics in direction_diagnostics.items():
        for space, diagnostics in latent_diagnostics.items():
            if not diagnostics.get("available"):
                continue
            digits = diagnostics["digits"]
            mean_signed = diagnostics["mean_offdiag_signed_cosine"]
            mean_abs = diagnostics["mean_offdiag_absolute_cosine"]

            signed_path = (
                plot_dir
                / f"{latent_name}_digit_color_direction_{space}_signed_cosine.png"
            )
            plot_cosine_heatmap(
                diagnostics["signed_cosine"],
                digits,
                (
                    f"{latent_name} digit color directions ({space}, signed)\n"
                    f"mean off-diagonal cosine={mean_signed:.3f}"
                ),
                signed_path,
                vmin=-1.0,
                vmax=1.0,
                cmap="coolwarm",
            )
            saved_paths.append(str(signed_path))

            absolute_path = (
                plot_dir
                / f"{latent_name}_digit_color_direction_{space}_absolute_cosine.png"
            )
            plot_cosine_heatmap(
                diagnostics["absolute_cosine"],
                digits,
                (
                    f"{latent_name} digit color directions ({space}, absolute)\n"
                    f"mean off-diagonal |cosine|={mean_abs:.3f}"
                ),
                absolute_path,
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
            )
            saved_paths.append(str(absolute_path))
    return saved_paths


def cross_digit_matrix_summary(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    diagonal = np.diag(matrix)
    off_diagonal = off_diagonal_values(matrix)
    source_digits = np.arange(matrix.shape[0])[:, None]
    target_digits = np.arange(matrix.shape[1])[None, :]
    source_groups = source_digits >= 5
    target_groups = target_digits >= 5
    same_digit = source_digits == target_digits
    same_group_offdiag = (source_groups == target_groups) & (~same_digit)
    cross_group = source_groups != target_groups

    mean_diagonal = nanmean_or_none(diagonal)
    mean_offdiag = nanmean_or_none(off_diagonal)
    mean_same_group_offdiag = nanmean_or_none(matrix[same_group_offdiag])
    mean_cross_group = nanmean_or_none(matrix[cross_group])
    diag_minus_offdiag = difference_or_none(mean_diagonal, mean_offdiag)
    return {
        "mean_diagonal": mean_diagonal,
        "mean_offdiag": mean_offdiag,
        "diag_minus_offdiag": diag_minus_offdiag,
        "diagonal_minus_offdiag": diag_minus_offdiag,
        "min_offdiag": nanmin_or_none(off_diagonal),
        "max_offdiag": nanmax_or_none(off_diagonal),
        "mean_same_group_offdiag": mean_same_group_offdiag,
        "mean_cross_group": mean_cross_group,
        "same_group_minus_cross_group": difference_or_none(
            mean_same_group_offdiag,
            mean_cross_group,
        ),
        "min_same_group_offdiag": nanmin_or_none(matrix[same_group_offdiag]),
        "max_same_group_offdiag": nanmax_or_none(matrix[same_group_offdiag]),
        "min_cross_group": nanmin_or_none(matrix[cross_group]),
        "max_cross_group": nanmax_or_none(matrix[cross_group]),
    }


def build_cross_digit_transfer(digit_probes, eval_latents_by_split):
    transfer = {}
    digits = list(range(10))

    for latent_name, latent_digit_probes in digit_probes.items():
        transfer[latent_name] = {}
        for split_name, eval_latents in eval_latents_by_split.items():
            accuracy_matrix = np.full((10, 10), np.nan, dtype=np.float64)
            balanced_accuracy_matrix = np.full((10, 10), np.nan, dtype=np.float64)
            target_digit_info = {}

            for target_digit in digits:
                target_mask = eval_latents["digits"] == target_digit
                target_features = eval_latents[latent_name][target_mask]
                target_colors = eval_latents["colors"][target_mask]
                target_digit_info[str(target_digit)] = subset_info(target_colors)
                if len(target_colors) == 0:
                    continue

                for source_digit, probe in latent_digit_probes.items():
                    eval_info = evaluate_color_probe(
                        probe,
                        target_features,
                        target_colors,
                    )
                    accuracy = eval_info["accuracy"]
                    balanced_accuracy = eval_info["balanced_accuracy"]
                    if accuracy is not None:
                        accuracy_matrix[source_digit, target_digit] = accuracy
                    if balanced_accuracy is not None:
                        balanced_accuracy_matrix[source_digit, target_digit] = (
                            balanced_accuracy
                        )

            transfer[latent_name][split_name] = {
                "source_digits": digits,
                "target_digits": digits,
                "available_source_digits": sorted(latent_digit_probes),
                "skipped_source_digits": [
                    digit for digit in digits if digit not in latent_digit_probes
                ],
                "target_digit_info": target_digit_info,
                "accuracy_matrix": matrix_to_jsonable(accuracy_matrix),
                "balanced_accuracy_matrix": matrix_to_jsonable(
                    balanced_accuracy_matrix
                ),
                "accuracy_summary": cross_digit_matrix_summary(accuracy_matrix),
                "balanced_accuracy_summary": cross_digit_matrix_summary(
                    balanced_accuracy_matrix
                ),
            }

    return transfer


def plot_transfer_heatmap(matrix, title, out_path):
    matrix = np.asarray(matrix, dtype=np.float64)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(10), labels=list(range(10)))
    ax.set_yticks(range(10), labels=list(range(10)))
    ax.set_xlabel("target digit evaluated on")
    ax.set_ylabel("source digit probe trained on")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_cross_digit_transfer_plots(cross_digit_transfer, output_dir):
    plot_dir = output_dir / "cross_digit_transfer"
    saved_paths = []
    for latent_name, latent_transfer in cross_digit_transfer.items():
        for split_name, split_transfer in latent_transfer.items():
            for metric, matrix_key, summary_key in [
                ("accuracy", "accuracy_matrix", "accuracy_summary"),
                (
                    "balanced_accuracy",
                    "balanced_accuracy_matrix",
                    "balanced_accuracy_summary",
                ),
            ]:
                summary = split_transfer[summary_key]
                offdiag = summary["mean_offdiag"]
                gap = summary["diagonal_minus_offdiag"]
                offdiag_text = "n/a" if offdiag is None else f"{offdiag:.3f}"
                gap_text = "n/a" if gap is None else f"{gap:.3f}"
                title = (
                    f"{latent_name} cross-digit color probe transfer\n"
                    f"{split_name}, {metric}, off-diagonal={offdiag_text}, "
                    f"diag-gap={gap_text}"
                )
                out_path = (
                    plot_dir
                    / f"{latent_name}_{split_name}_cross_digit_transfer_{metric}.png"
                )
                plot_transfer_heatmap(split_transfer[matrix_key], title, out_path)
                saved_paths.append(str(out_path))
    return saved_paths


def summarize_results(results, split_names):
    summary = {}
    for latent_name, latent_results in results.items():
        summary[latent_name] = {}
        for split_name in split_names:
            summary[latent_name][split_name] = {}
            for subgroup_type in ["global", "digit_group", "digit"]:
                accuracies = []
                balanced_accuracies = []
                above_baseline = []
                weights = []
                num_valid = 0
                num_skipped = 0
                total_eval_examples = 0
                for probe_result in latent_results.values():
                    if probe_result["type"] != subgroup_type:
                        continue
                    if probe_result["skipped"]:
                        num_skipped += 1
                        continue
                    eval_result = probe_result["eval"].get(split_name)
                    if eval_result is None:
                        continue
                    num_valid += 1
                    num_examples = eval_result.get("num_examples")
                    if num_examples is not None:
                        total_eval_examples += num_examples
                    accuracies.append(eval_result.get("accuracy"))
                    balanced_accuracies.append(eval_result.get("balanced_accuracy"))
                    above_baseline.append(eval_result.get("accuracy_minus_majority_baseline"))
                    weights.append(num_examples)
                macro_accuracy = mean_or_none(accuracies)
                macro_balanced_accuracy = mean_or_none(balanced_accuracies)
                macro_above_baseline = mean_or_none(above_baseline)
                summary[latent_name][split_name][subgroup_type] = {
                    "num_valid_subgroups": num_valid,
                    "num_skipped_subgroups": num_skipped,
                    "total_eval_examples": int(total_eval_examples),
                    "macro_mean_accuracy": macro_accuracy,
                    "macro_mean_balanced_accuracy": macro_balanced_accuracy,
                    "macro_mean_accuracy_minus_majority_baseline": macro_above_baseline,
                    "weighted_mean_accuracy": weighted_mean_or_none(accuracies, weights),
                    "weighted_mean_balanced_accuracy": weighted_mean_or_none(
                        balanced_accuracies,
                        weights,
                    ),
                    "weighted_mean_accuracy_minus_majority_baseline": weighted_mean_or_none(
                        above_baseline,
                        weights,
                    ),
                    "mean_accuracy": macro_accuracy,
                    "mean_balanced_accuracy": macro_balanced_accuracy,
                    "mean_accuracy_minus_majority_baseline": macro_above_baseline,
                }
    return summary


def run_subgroup_probes(train_latents, eval_latents_by_split, args):
    results = {latent_name: {} for latent_name in LATENT_NAMES}
    permuted_control_results = {latent_name: {} for latent_name in LATENT_NAMES}
    direction_vectors = {
        latent_name: {"raw": {}, "standardized": {}}
        for latent_name in LATENT_NAMES
    }
    digit_probes = {latent_name: {} for latent_name in LATENT_NAMES}

    for latent_name in LATENT_NAMES:
        for spec in subgroup_specs():
            train_mask = subgroup_mask(train_latents, spec)
            train_colors = train_latents["colors"][train_mask]
            train_digits = train_latents["digits"][train_mask]
            train_info = subset_info(train_colors)
            train_info["low_count_warning"] = (
                train_info["min_color_count"] < args.warn_train_per_color
            )
            train_info["warn_train_per_color"] = args.warn_train_per_color

            probe_result = {
                "type": spec["type"],
                "value": spec["value"],
                "label": spec["label"],
                "train": train_info,
                "skipped": False,
                "skip_reason": None,
                "eval": {},
            }

            if not can_fit_color_probe(train_colors, args.min_train_per_color):
                probe_result["skipped"] = True
                probe_result["skip_reason"] = (
                    f"fewer than {args.min_train_per_color} training examples "
                    "for at least one color"
                )
                results[latent_name][spec["key"]] = probe_result
                permuted_control_results[latent_name][spec["key"]] = {
                    "type": spec["type"],
                    "value": spec["value"],
                    "label": spec["label"],
                    "train": train_info,
                    "skipped": True,
                    "skip_reason": probe_result["skip_reason"],
                    "control": "permuted_within_digit_train_colors",
                    "eval": {},
                }
                continue

            probe = make_probe(args.max_iter, args.seed)
            probe.fit(train_latents[latent_name][train_mask], train_colors)

            permuted_train_colors = permute_colors_within_digits(
                train_colors,
                train_digits,
                args.seed,
                latent_name,
                spec["key"],
            )
            permuted_probe = make_probe(
                args.max_iter,
                stable_seed(args.seed, latent_name, spec["key"], "permuted_control"),
            )
            permuted_probe.fit(
                train_latents[latent_name][train_mask],
                permuted_train_colors,
            )
            permuted_probe_result = {
                "type": spec["type"],
                "value": spec["value"],
                "label": spec["label"],
                "control": "permuted_within_digit_train_colors",
                "train": {
                    **subset_info(permuted_train_colors),
                    "original_color_counts": train_info["color_counts"],
                    "permutation_unit": "digit",
                    "note": (
                        "Color labels are randomly permuted within each digit "
                        "on the training subset; evaluation uses true labels."
                    ),
                },
                "skipped": False,
                "skip_reason": None,
                "eval": {},
            }
            if spec["type"] == "digit":
                direction = color_direction_from_probe(probe)
                if direction is not None:
                    digit = spec["value"]
                    digit_probes[latent_name][digit] = probe
                    direction_vectors[latent_name]["raw"][digit] = direction["raw"]
                    direction_vectors[latent_name]["standardized"][digit] = direction[
                        "standardized"
                    ]
                    probe_result["direction"] = {
                        "classes": direction["classes"],
                        "raw_norm": float(np.linalg.norm(direction["raw"])),
                        "standardized_norm": float(
                            np.linalg.norm(direction["standardized"])
                        ),
                        "sign_aligned_positive_class": "green",
                    }

            for split_name, eval_latents in eval_latents_by_split.items():
                eval_mask = subgroup_mask(eval_latents, spec)
                eval_features = eval_latents[latent_name][eval_mask]
                eval_colors = eval_latents["colors"][eval_mask]
                probe_result["eval"][split_name] = evaluate_color_probe(
                    probe,
                    eval_features,
                    eval_colors,
                )
                permuted_probe_result["eval"][split_name] = evaluate_color_probe(
                    permuted_probe,
                    eval_features,
                    eval_colors,
                )

            results[latent_name][spec["key"]] = probe_result
            permuted_control_results[latent_name][spec["key"]] = (
                permuted_probe_result
            )

    direction_diagnostics = build_direction_diagnostics(direction_vectors)
    cross_digit_transfer = build_cross_digit_transfer(
        digit_probes,
        eval_latents_by_split,
    )
    return (
        results,
        permuted_control_results,
        direction_diagnostics,
        cross_digit_transfer,
    )


def print_direction_summary(direction_diagnostics):
    print("")
    print("Digit-specific color direction diagnostics")
    print(
        "latent | space | mean_signed_cos | mean_abs_cos | "
        "first_svd_energy | components_90pct"
    )
    print("--- | --- | ---: | ---: | ---: | ---:")
    for latent_name, latent_diagnostics in direction_diagnostics.items():
        for space, diagnostics in latent_diagnostics.items():
            if not diagnostics.get("available"):
                print(f"{latent_name} | {space} | n/a | n/a | n/a | n/a")
                continue
            print(
                f"{latent_name} | {space} | "
                f"{diagnostics['mean_offdiag_signed_cosine']:.4f} | "
                f"{diagnostics['mean_offdiag_absolute_cosine']:.4f} | "
                f"{diagnostics['first_singular_energy_fraction']:.4f} | "
                f"{diagnostics['components_for_90pct_energy']}"
            )


def print_cross_digit_transfer_summary(cross_digit_transfer):
    print("")
    print("Cross-digit color probe transfer")
    print(
        "latent | eval_split | diag_bal_acc | offdiag_bal_acc | "
        "diag_gap | same_group_offdiag | cross_group | same_minus_cross"
    )
    print("--- | --- | ---: | ---: | ---: | ---: | ---: | ---:")
    for latent_name, latent_transfer in cross_digit_transfer.items():
        for split_name, split_transfer in latent_transfer.items():
            summary = split_transfer["balanced_accuracy_summary"]
            diag = summary["mean_diagonal"]
            offdiag = summary["mean_offdiag"]
            diag_gap = summary["diagonal_minus_offdiag"]
            same_group = summary["mean_same_group_offdiag"]
            cross_group = summary["mean_cross_group"]
            same_minus_cross = summary["same_group_minus_cross_group"]
            diag_text = "n/a" if diag is None else f"{diag:.4f}"
            offdiag_text = "n/a" if offdiag is None else f"{offdiag:.4f}"
            gap_text = "n/a" if diag_gap is None else f"{diag_gap:.4f}"
            same_text = "n/a" if same_group is None else f"{same_group:.4f}"
            cross_text = "n/a" if cross_group is None else f"{cross_group:.4f}"
            same_minus_cross_text = (
                "n/a" if same_minus_cross is None else f"{same_minus_cross:.4f}"
            )
            print(
                f"{latent_name} | {split_name} | {diag_text} | "
                f"{offdiag_text} | {gap_text} | {same_text} | "
                f"{cross_text} | {same_minus_cross_text}"
            )


def print_summary(summary):
    print("")
    print("Within-subgroup color probe summary")
    print(
        "latent | eval_split | subgroup_type | valid/skipped | "
        "macro_bal_acc | weighted_bal_acc | macro_acc | weighted_acc"
    )
    print("--- | --- | --- | ---: | ---: | ---: | ---: | ---:")
    for latent_name, latent_summary in summary.items():
        for split_name, split_summary in latent_summary.items():
            for subgroup_type, row in split_summary.items():
                acc = row["macro_mean_accuracy"]
                bal = row["macro_mean_balanced_accuracy"]
                weighted_acc = row["weighted_mean_accuracy"]
                weighted_bal = row["weighted_mean_balanced_accuracy"]
                acc_text = "n/a" if acc is None else f"{acc:.4f}"
                bal_text = "n/a" if bal is None else f"{bal:.4f}"
                weighted_acc_text = "n/a" if weighted_acc is None else f"{weighted_acc:.4f}"
                weighted_bal_text = "n/a" if weighted_bal is None else f"{weighted_bal:.4f}"
                valid_skipped = (
                    f"{row['num_valid_subgroups']}/{row['num_skipped_subgroups']}"
                )
                print(
                    f"{latent_name} | {split_name} | {subgroup_type} | "
                    f"{valid_skipped} | {bal_text} | {weighted_bal_text} | "
                    f"{acc_text} | {weighted_acc_text}"
                )


def print_permuted_control_summary(real_summary, permuted_summary):
    print("")
    print("Permuted-within-digit label control")
    print(
        "latent | eval_split | subgroup_type | real_macro_bal | "
        "permuted_macro_bal | real_minus_permuted"
    )
    print("--- | --- | --- | ---: | ---: | ---:")
    for latent_name, latent_summary in real_summary.items():
        control_latent_summary = permuted_summary.get(latent_name, {})
        for split_name, split_summary in latent_summary.items():
            control_split_summary = control_latent_summary.get(split_name, {})
            for subgroup_type, row in split_summary.items():
                control_row = control_split_summary.get(subgroup_type, {})
                real_bal = row["macro_mean_balanced_accuracy"]
                control_bal = control_row.get("macro_mean_balanced_accuracy")
                gap = difference_or_none(real_bal, control_bal)
                print(
                    f"{latent_name} | {split_name} | {subgroup_type} | "
                    f"{format_optional_float(real_bal)} | "
                    f"{format_optional_float(control_bal)} | "
                    f"{format_optional_float(gap)}"
                )


def format_optional_float(value):
    return "n/a" if value is None else f"{value:.4f}"


def print_detailed_digit_results(results, split_name="test_balanced"):
    print("")
    print(f"Per-digit color probe accuracy on {split_name}")
    print(
        "latent | digit | train red/green | eval red/green | "
        "acc | balanced_acc | majority_baseline | n"
    )
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for latent_name in LATENT_NAMES:
        for digit in range(10):
            key = f"digit_{digit}"
            row = results[latent_name][key]
            train_counts = row["train"]["color_counts"]
            train_text = f"{train_counts['red']}/{train_counts['green']}"
            eval_row = row["eval"].get(split_name)
            if row["skipped"] or eval_row is None:
                print(
                    f"{latent_name} | {digit} | {train_text} | skipped | "
                    "skipped | skipped | skipped | 0"
                )
                continue
            acc = eval_row["accuracy"]
            bal = eval_row["balanced_accuracy"]
            baseline = eval_row["majority_baseline"]
            eval_counts = eval_row["color_counts"]
            eval_text = f"{eval_counts['red']}/{eval_counts['green']}"
            print(
                f"{latent_name} | {digit} | "
                f"{train_text} | "
                f"{eval_text} | "
                f"{acc:.4f} | "
                f"{bal:.4f} | "
                f"{baseline:.4f} | "
                f"{eval_row['num_examples']}"
            )


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

    print(f"Using device: {device}")
    print(f"Checkpoint directory: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model family: {model_family}")
    print(f"Scaler: {scaler}")
    print(f"Train split: {args.train_split}")
    print(f"Eval splits: {', '.join(args.eval_splits)}")
    print(f"Output directory: {output_dir}")

    train_dataset = load_split(args.embedding_dir, args.train_split, mean, std)
    train_latents = extract_latents(
        model,
        train_dataset,
        args.batch_size,
        args.num_workers,
        device,
    )

    eval_latents_by_split = {}
    all_eval_splits = [args.train_split]
    for split_name in args.eval_splits:
        if split_name not in all_eval_splits:
            all_eval_splits.append(split_name)

    for split_name in all_eval_splits:
        if split_name == args.train_split:
            eval_latents_by_split[split_name] = train_latents
            continue
        dataset = load_split(args.embedding_dir, split_name, mean, std)
        eval_latents_by_split[split_name] = extract_latents(
            model,
            dataset,
            args.batch_size,
            args.num_workers,
            device,
        )

    (
        results,
        permuted_control_results,
        direction_diagnostics,
        cross_digit_transfer,
    ) = run_subgroup_probes(
        train_latents,
        eval_latents_by_split,
        args,
    )
    summary = summarize_results(results, all_eval_splits)
    permuted_control_summary = summarize_results(
        permuted_control_results,
        all_eval_splits,
    )
    direction_plot_paths = save_direction_plots(direction_diagnostics, output_dir)
    cross_digit_transfer_plot_paths = save_cross_digit_transfer_plots(
        cross_digit_transfer,
        output_dir,
    )

    print_summary(summary)
    print_permuted_control_summary(summary, permuted_control_summary)
    if "test_balanced" in all_eval_splits:
        print_detailed_digit_results(results, split_name="test_balanced")
    print_direction_summary(direction_diagnostics)
    print_cross_digit_transfer_summary(cross_digit_transfer)

    results_path = output_dir / "subgroup_color_probe_results.json"
    payload = {
        "command": command_string(),
        "checkpoint_dir": str(run_dir),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_args": checkpoint.get("args", {}),
        "model_family": model_family,
        "probe_args": args_to_dict(args),
        "scaler": str(scaler),
        "embedding_dir": str(args.embedding_dir),
        "train_split": args.train_split,
        "eval_splits": all_eval_splits,
        "summary": summary,
        "permuted_within_digit_control": {
            "summary": permuted_control_summary,
            "results": permuted_control_results,
        },
        "direction_diagnostics": direction_diagnostics,
        "direction_plot_paths": direction_plot_paths,
        "cross_digit_transfer": cross_digit_transfer,
        "cross_digit_transfer_plot_paths": cross_digit_transfer_plot_paths,
        "results": results,
    }
    save_json(results_path, payload)
    print("")
    print(f"Saved subgroup color probe results to {results_path}")


if __name__ == "__main__":
    main()
