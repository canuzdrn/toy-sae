"""Evaluate forced red/green interventions in a factorized split SAE.

For each underlying ColoredMNIST image, this script constructs exact red and
green counterfactuals with the same foreground mask. It then:

1. extracts canonical content from the red counterfactual,
2. extracts canonical content from the green counterfactual,
3. forces both red and green styles on each content representation,
4. compares every output with the exact paired base-AE target.

The resulting four conditions test whether the style operation changes color
without changing digit morphology, and whether the output is independent of
the source color used to obtain content.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import scripts.train.train_split_sae_canonical_good as canonical_good
from toy_sae.utils.checkpoints import load_checkpoint, resolve_checkpoint_path
from toy_sae.utils.embeddings import load_scaler
from toy_sae.utils.script_utils import args_to_dict, command_string, save_json
from toy_sae.utils.split_sae_loading import infer_model_family, load_split_sae_model
from toy_sae.utils.torch_utils import get_device


COLOR_NAMES = {0: "red", 1: "green"}
SOURCE_COLORS = (0, 1)
COLOR_INDICES = (0, 1)
STYLE_INDICES = (0, 1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Run directory containing the factorized checkpoint.",
    )
    parser.add_argument("--checkpoint-name", default="best_recon")
    parser.add_argument(
        "--scaler",
        type=Path,
        default=None,
        help="Defaults to embedding_scaler.npz beside the checkpoint.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/colored_mnist"),
    )
    parser.add_argument(
        "--probe-embedding-dir",
        type=Path,
        default=Path("data/base_ae_embeddings"),
    )
    parser.add_argument(
        "--probe-train-split",
        default="ae_train_balanced",
        help="Balanced split used to fit frozen digit and color probes.",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("checkpoints/base_ae/best.pt"),
    )
    parser.add_argument("--splits", nargs="+", default=["test_balanced"])
    parser.add_argument("--encode-batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional deterministic subset size for smoke tests.",
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_args(args):
    checkpoint_path = resolve_checkpoint_path(
        args.checkpoint_dir,
        args.checkpoint_name,
    )
    if args.scaler is None:
        args.scaler = checkpoint_path.parent / "embedding_scaler.npz"
    if args.out_dir is None:
        args.out_dir = (
            Path("outputs/factorized_counterfactual_style_eval")
            / f"{checkpoint_path.parent.name}_{checkpoint_path.stem}"
        )

    required_paths = [
        checkpoint_path,
        args.scaler,
        args.base_checkpoint,
        args.probe_embedding_dir / f"{args.probe_train_split}.npz",
    ]
    required_paths.extend(
        args.image_dir / f"{split_name}.npz"
        for split_name in args.splits
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    if args.encode_batch_size <= 0:
        raise ValueError("--encode-batch-size must be positive")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be positive")

    return checkpoint_path


def load_images(path, max_examples=None, seed=0):
    data = np.load(path)
    result = {
        "images": data["images"].astype(np.float32),
        "digits": data["digits"].astype(np.int64),
        "colors": data["colors"].astype(np.int64),
        "digit_groups": data["digit_groups"].astype(np.int64),
    }
    data.close()
    if max_examples is not None and max_examples < len(result["images"]):
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(
                len(result["images"]),
                size=max_examples,
                replace=False,
            )
        )
        result = {
            key: values[indices]
            for key, values in result.items()
        }
    return result


def standardize(raw_embeddings, mean, std):
    return ((raw_embeddings - mean) / std).astype(np.float32)


def make_semantic_probe(max_iter, seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def fit_semantic_probes(embedding_path, mean, std, max_iter, seed):
    data = np.load(embedding_path)
    embeddings = standardize(
        data["embeddings"].astype(np.float32),
        mean,
        std,
    )
    digits = data["digits"].astype(np.int64)
    colors = data["colors"].astype(np.int64)
    data.close()

    digit_probe = make_semantic_probe(max_iter, seed)
    color_probe = make_semantic_probe(max_iter, seed)
    digit_probe.fit(embeddings, digits)
    color_probe.fit(embeddings, colors)
    return {
        "digit": digit_probe,
        "color": color_probe,
    }


def encode_paired_targets(images, base_ae, device, mean, std, batch_size):
    paired_embeddings = {}
    grayscale_shape = images.max(axis=1).astype(np.float32)
    for color_index in COLOR_INDICES:
        color_name = COLOR_NAMES[color_index]
        recolored = canonical_good.recolor_images(
            images,
            color_index=color_index,
        )
        raw_embeddings = canonical_good.encode_images(
            base_ae,
            recolored,
            device,
            batch_size,
        )
        paired_embeddings[color_name] = standardize(
            raw_embeddings,
            mean,
            std,
        )
    return grayscale_shape, paired_embeddings


def infer_style_mapping(
    model,
    paired_embeddings,
    batch_size,
    device,
):
    """Align unordered learned style indices with semantic red/green labels."""
    counts = np.zeros((2, 2), dtype=np.int64)
    probability_sums = np.zeros((2, 2), dtype=np.float64)
    num_examples = len(paired_embeddings["red"])

    with torch.no_grad():
        for color_index in COLOR_INDICES:
            color_name = COLOR_NAMES[color_index]
            embeddings = paired_embeddings[color_name]
            for start in range(0, num_examples, batch_size):
                end = min(start + batch_size, num_examples)
                embedding_tensor = torch.from_numpy(
                    embeddings[start:end]
                ).to(device)
                _, z_bad = model.encode(embedding_tensor)
                _, style_probabilities = model.decode_style(z_bad)
                probabilities = style_probabilities.cpu().numpy()
                predictions = probabilities.argmax(axis=1)
                counts[color_index] += np.bincount(
                    predictions,
                    minlength=2,
                )
                probability_sums[color_index] += probabilities.sum(axis=0)

    identity_correct = int(counts[0, 0] + counts[1, 1])
    swapped_correct = int(counts[0, 1] + counts[1, 0])
    total = int(counts.sum())
    if swapped_correct > identity_correct:
        color_to_style = {0: 1, 1: 0}
        selected_correct = swapped_correct
        alternative_correct = identity_correct
    else:
        color_to_style = {0: 0, 1: 1}
        selected_correct = identity_correct
        alternative_correct = swapped_correct

    style_to_color = {
        style_index: color_index
        for color_index, style_index in color_to_style.items()
    }
    mean_probabilities = probability_sums / num_examples

    return {
        "color_to_style": color_to_style,
        "diagnostics": {
            "color_to_style_index": {
                COLOR_NAMES[color_index]: int(style_index)
                for color_index, style_index in color_to_style.items()
            },
            "style_index_to_color": {
                str(style_index): COLOR_NAMES[color_index]
                for style_index, color_index in style_to_color.items()
            },
            "argmax_counts_by_color": {
                COLOR_NAMES[color_index]: counts[color_index].tolist()
                for color_index in COLOR_INDICES
            },
            "mean_style_probabilities_by_color": {
                COLOR_NAMES[color_index]: mean_probabilities[
                    color_index
                ].tolist()
                for color_index in COLOR_INDICES
            },
            "selected_mapping_accuracy": float(selected_correct / total),
            "alternative_mapping_accuracy": float(alternative_correct / total),
            "mapping_accuracy_margin": float(
                (selected_correct - alternative_correct) / total
            ),
        },
    }


def collect_forced_outputs(
    model,
    paired_embeddings,
    color_to_style,
    batch_size,
    device,
):
    num_examples = len(paired_embeddings["red"])
    outputs = {}
    contents = {}

    for source_index in SOURCE_COLORS:
        source_name = COLOR_NAMES[source_index]
        outputs[source_name] = {
            COLOR_NAMES[color_index]: []
            for color_index in COLOR_INDICES
        }
        contents[source_name] = []

    with torch.no_grad():
        for start in range(0, num_examples, batch_size):
            end = min(start + batch_size, num_examples)
            target_tensors = {
                color_name: torch.from_numpy(
                    paired_embeddings[color_name][start:end]
                ).to(device)
                for color_name in ("red", "green")
            }

            for source_index in SOURCE_COLORS:
                source_name = COLOR_NAMES[source_index]
                z_good, _ = model.encode(target_tensors[source_name])
                content = model.decode_good(z_good)
                contents[source_name].append(content.cpu().numpy())

                for color_index in COLOR_INDICES:
                    color_name = COLOR_NAMES[color_index]
                    style_index = color_to_style[color_index]
                    one_hot = torch.zeros(
                        len(content),
                        model.num_styles,
                        device=device,
                        dtype=content.dtype,
                    )
                    one_hot[:, style_index] = 1.0
                    forced = model.apply_style(content, one_hot)
                    outputs[source_name][color_name].append(
                        forced.cpu().numpy()
                    )

    for source_name in ("red", "green"):
        contents[source_name] = np.concatenate(
            contents[source_name],
            axis=0,
        ).astype(np.float32)
        for style_name in ("red", "green"):
            outputs[source_name][style_name] = np.concatenate(
                outputs[source_name][style_name],
                axis=0,
            ).astype(np.float32)

    return {
        "outputs": outputs,
        "contents": contents,
    }


def row_mse(left, right):
    axes = tuple(range(1, left.ndim))
    return np.mean((left - right) ** 2, axis=axes)


def mean_cosine(left, right, eps=1e-12):
    numerator = np.sum(left * right, axis=1)
    denominator = (
        np.linalg.norm(left, axis=1)
        * np.linalg.norm(right, axis=1)
    )
    return float(np.mean(numerator / np.maximum(denominator, eps)))


def classification_metrics(probe, values, labels):
    predictions = probe.predict(values)
    if len(np.unique(labels)) > 1:
        balanced_accuracy = balanced_accuracy_score(labels, predictions)
    else:
        balanced_accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy),
    }


def decode_image_metrics(
    base_ae,
    predicted,
    target,
    grayscale_shape,
    forced_style,
    mean,
    std,
    batch_size,
    device,
):
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)
    decoded_target_sum = 0.0
    raw_target_sum = 0.0
    shape_sum = 0.0
    total_examples = 0

    with torch.no_grad():
        for start in range(0, len(predicted), batch_size):
            end = min(start + batch_size, len(predicted))
            predicted_tensor = torch.from_numpy(
                predicted[start:end]
            ).to(device)
            target_tensor = torch.from_numpy(
                target[start:end]
            ).to(device)
            grayscale_tensor = torch.from_numpy(
                grayscale_shape[start:end]
            ).to(device)

            decoded_predicted = base_ae.decode(
                predicted_tensor * std_tensor + mean_tensor
            )
            decoded_target = base_ae.decode(
                target_tensor * std_tensor + mean_tensor
            )
            raw_target = torch.zeros_like(decoded_predicted)
            raw_target[:, forced_style] = grayscale_tensor

            current_batch_size = end - start
            total_examples += current_batch_size
            decoded_target_sum += (
                (decoded_predicted - decoded_target)
                .pow(2)
                .flatten(1)
                .mean(dim=1)
                .sum()
                .item()
            )
            raw_target_sum += (
                (decoded_predicted - raw_target)
                .pow(2)
                .flatten(1)
                .mean(dim=1)
                .sum()
                .item()
            )
            shape_sum += (
                (
                    decoded_predicted.max(dim=1).values
                    - grayscale_tensor
                )
                .pow(2)
                .flatten(1)
                .mean(dim=1)
                .sum()
                .item()
            )

    return {
        "decoded_target_mse": float(
            decoded_target_sum / total_examples
        ),
        "raw_target_image_mse": float(
            raw_target_sum / total_examples
        ),
        "grayscale_shape_mse": float(shape_sum / total_examples),
    }


def summarize_embedding_condition(
    predicted,
    target,
    wrong_target,
    digits,
    forced_style,
    probes,
):
    target_errors = row_mse(predicted, target)
    wrong_target_errors = row_mse(predicted, wrong_target)
    forced_color_labels = np.full(
        len(predicted),
        forced_style,
        dtype=np.int64,
    )

    return {
        "embedding_mse": float(target_errors.mean()),
        "wrong_color_embedding_mse": float(
            wrong_target_errors.mean()
        ),
        "wrong_minus_correct_embedding_mse": float(
            wrong_target_errors.mean() - target_errors.mean()
        ),
        "correct_target_preference_rate": float(
            (target_errors < wrong_target_errors).mean()
        ),
        "digit_probe": classification_metrics(
            probes["digit"],
            predicted,
            digits,
        ),
        "forced_color_probe": classification_metrics(
            probes["color"],
            predicted,
            forced_color_labels,
        ),
    }


def summarize_condition(
    base_ae,
    predicted,
    target,
    wrong_target,
    grayscale_shape,
    digits,
    forced_style,
    probes,
    mean,
    std,
    batch_size,
    device,
):
    result = summarize_embedding_condition(
        predicted,
        target,
        wrong_target,
        digits,
        forced_style,
        probes,
    )
    result.update(
        decode_image_metrics(
            base_ae,
            predicted,
            target,
            grayscale_shape,
            forced_style,
            mean,
            std,
            batch_size,
            device,
        )
    )
    return result


def summarize_target_ceiling(
    base_ae,
    target,
    grayscale_shape,
    digits,
    style_index,
    probes,
    mean,
    std,
    batch_size,
    device,
):
    target_color_labels = np.full(
        len(target),
        style_index,
        dtype=np.int64,
    )
    result = {
        "digit_probe": classification_metrics(
            probes["digit"],
            target,
            digits,
        ),
        "color_probe": classification_metrics(
            probes["color"],
            target,
            target_color_labels,
        ),
    }
    image_metrics = decode_image_metrics(
        base_ae,
        target,
        target,
        grayscale_shape,
        style_index,
        mean,
        std,
        batch_size,
        device,
    )
    result.update(
        {
            "raw_target_image_mse": image_metrics[
                "raw_target_image_mse"
            ],
            "grayscale_shape_mse": image_metrics[
                "grayscale_shape_mse"
            ],
        }
    )
    return result


def summarize_cross_source(
    red_source_output,
    green_source_output,
    digits,
):
    errors = row_mse(red_source_output, green_source_output)
    return {
        "embedding_mse": float(errors.mean()),
        "per_digit_embedding_mse": {
            str(digit): float(errors[digits == digit].mean())
            for digit in range(10)
        },
    }


def per_digit_conditions(
    generated,
    paired_embeddings,
    digits,
    probes,
):
    results = {}

    for digit in range(10):
        mask = digits == digit
        digit_results = {}
        for source_name in ("red", "green"):
            for color_index in COLOR_INDICES:
                color_name = COLOR_NAMES[color_index]
                other_name = COLOR_NAMES[1 - color_index]
                key = f"source_{source_name}_force_{color_name}"
                digit_results[key] = summarize_embedding_condition(
                    generated["outputs"][source_name][color_name][mask],
                    paired_embeddings[color_name][mask],
                    paired_embeddings[other_name][mask],
                    digits[mask],
                    color_index,
                    probes,
                )
        results[str(digit)] = digit_results
    return results


def evaluate_split(
    model,
    base_ae,
    image_data,
    mean,
    std,
    probes,
    encode_batch_size,
    eval_batch_size,
    device,
):
    grayscale_shape, paired_embeddings = encode_paired_targets(
        image_data["images"],
        base_ae,
        device,
        mean,
        std,
        encode_batch_size,
    )
    mapping = infer_style_mapping(
        model,
        paired_embeddings,
        eval_batch_size,
        device,
    )
    generated = collect_forced_outputs(
        model,
        paired_embeddings,
        mapping["color_to_style"],
        eval_batch_size,
        device,
    )

    digits = image_data["digits"]
    conditions = {}
    for source_name in ("red", "green"):
        for color_index in COLOR_INDICES:
            color_name = COLOR_NAMES[color_index]
            other_name = COLOR_NAMES[1 - color_index]
            key = f"source_{source_name}_force_{color_name}"
            conditions[key] = summarize_condition(
                base_ae,
                generated["outputs"][source_name][color_name],
                paired_embeddings[color_name],
                paired_embeddings[other_name],
                grayscale_shape,
                digits,
                color_index,
                probes,
                mean,
                std,
                eval_batch_size,
                device,
            )

    target_ceilings = {}
    for color_index in COLOR_INDICES:
        color_name = COLOR_NAMES[color_index]
        target_ceilings[color_name] = summarize_target_ceiling(
            base_ae,
            paired_embeddings[color_name],
            grayscale_shape,
            digits,
            color_index,
            probes,
            mean,
            std,
            eval_batch_size,
            device,
        )

    content_pair_mse = row_mse(
        generated["contents"]["red"],
        generated["contents"]["green"],
    )
    cross_source = {
        style_name: summarize_cross_source(
            generated["outputs"]["red"][style_name],
            generated["outputs"]["green"][style_name],
            digits,
        )
        for style_name in ("red", "green")
    }

    style_effects = {}
    target_delta = paired_embeddings["red"] - paired_embeddings["green"]
    for source_name in ("red", "green"):
        predicted_delta = (
            generated["outputs"][source_name]["red"]
            - generated["outputs"][source_name]["green"]
        )
        style_effects[source_name] = {
            "predicted_red_green_mse": float(
                row_mse(
                    generated["outputs"][source_name]["red"],
                    generated["outputs"][source_name]["green"],
                ).mean()
            ),
            "target_red_green_mse": float(
                row_mse(
                    paired_embeddings["red"],
                    paired_embeddings["green"],
                ).mean()
            ),
            "delta_cosine_to_target": mean_cosine(
                predicted_delta,
                target_delta,
            ),
        }

    return {
        "num_examples": int(len(digits)),
        "style_mapping": mapping["diagnostics"],
        "conditions": conditions,
        "target_ceilings": target_ceilings,
        "content_red_green_mse": float(content_pair_mse.mean()),
        "content_red_green_mse_per_digit": {
            str(digit): float(content_pair_mse[digits == digit].mean())
            for digit in range(10)
        },
        "cross_source_consistency": cross_source,
        "style_effects": style_effects,
        "per_digit_conditions": per_digit_conditions(
            generated,
            paired_embeddings,
            digits,
            probes,
        ),
    }


def print_results(results):
    for split_name, split_results in results.items():
        print("")
        print(f"Counterfactual forced-style evaluation: {split_name}")
        mapping = split_results["style_mapping"]
        print(
            "Inferred style mapping: "
            + ", ".join(
                f"{color}=style_{style_index}"
                for color, style_index in mapping[
                    "color_to_style_index"
                ].items()
            )
        )
        print(
            "Mapping accuracy: "
            f"{mapping['selected_mapping_accuracy']:.4f} "
            f"(margin={mapping['mapping_accuracy_margin']:.4f})"
        )
        print(
            "condition | emb_mse | wrong_emb_mse | preference | "
            "digit_acc | forced_color_acc | shape_mse"
        )
        print("--- | ---: | ---: | ---: | ---: | ---: | ---:")
        for condition_name, row in split_results["conditions"].items():
            print(
                f"{condition_name} | "
                f"{row['embedding_mse']:.6f} | "
                f"{row['wrong_color_embedding_mse']:.6f} | "
                f"{row['correct_target_preference_rate']:.4f} | "
                f"{row['digit_probe']['accuracy']:.4f} | "
                f"{row['forced_color_probe']['accuracy']:.4f} | "
                f"{row['grayscale_shape_mse']:.6f}"
            )

        print("")
        print("Target ceilings")
        print("style | digit_acc | color_acc | shape_mse")
        print("--- | ---: | ---: | ---:")
        for style_name, row in split_results["target_ceilings"].items():
            print(
                f"{style_name} | "
                f"{row['digit_probe']['accuracy']:.4f} | "
                f"{row['color_probe']['accuracy']:.4f} | "
                f"{row['grayscale_shape_mse']:.6f}"
            )

        print("")
        print(
            "Content red/green MSE: "
            f"{split_results['content_red_green_mse']:.6f}"
        )
        print("Forced-style cross-source consistency")
        print("style | source_red_vs_green_mse")
        print("--- | ---:")
        for style_name, row in split_results[
            "cross_source_consistency"
        ].items():
            print(f"{style_name} | {row['embedding_mse']:.6f}")

        print("")
        print("Style-effect alignment")
        print("source | predicted_red_green_mse | target_mse | delta_cosine")
        print("--- | ---: | ---: | ---:")
        for source_name, row in split_results["style_effects"].items():
            print(
                f"{source_name} | "
                f"{row['predicted_red_green_mse']:.6f} | "
                f"{row['target_red_green_mse']:.6f} | "
                f"{row['delta_cosine_to_target']:.4f}"
            )


def save_condition_csv(path, results):
    fieldnames = [
        "split",
        "condition",
        "embedding_mse",
        "wrong_color_embedding_mse",
        "wrong_minus_correct_embedding_mse",
        "correct_target_preference_rate",
        "decoded_target_mse",
        "raw_target_image_mse",
        "grayscale_shape_mse",
        "digit_accuracy",
        "digit_balanced_accuracy",
        "forced_color_accuracy",
        "forced_color_balanced_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, split_results in results.items():
            for condition_name, row in split_results["conditions"].items():
                writer.writerow(
                    {
                        "split": split_name,
                        "condition": condition_name,
                        "embedding_mse": row["embedding_mse"],
                        "wrong_color_embedding_mse": row[
                            "wrong_color_embedding_mse"
                        ],
                        "wrong_minus_correct_embedding_mse": row[
                            "wrong_minus_correct_embedding_mse"
                        ],
                        "correct_target_preference_rate": row[
                            "correct_target_preference_rate"
                        ],
                        "decoded_target_mse": row["decoded_target_mse"],
                        "raw_target_image_mse": row[
                            "raw_target_image_mse"
                        ],
                        "grayscale_shape_mse": row[
                            "grayscale_shape_mse"
                        ],
                        "digit_accuracy": row["digit_probe"]["accuracy"],
                        "digit_balanced_accuracy": row["digit_probe"][
                            "balanced_accuracy"
                        ],
                        "forced_color_accuracy": row[
                            "forced_color_probe"
                        ]["accuracy"],
                        "forced_color_balanced_accuracy": row[
                            "forced_color_probe"
                        ]["balanced_accuracy"],
                    }
                )


def main():
    args = parse_args()
    checkpoint_path = resolve_args(args)
    device = get_device()

    checkpoint = load_checkpoint(checkpoint_path)
    model_family = infer_model_family(checkpoint)
    if model_family != "factorized_style":
        raise ValueError(
            "This evaluator requires a factorized_style checkpoint; "
            f"inferred {model_family!r}"
        )
    model = load_split_sae_model(
        checkpoint,
        device,
        model_family="factorized_style",
    )
    if model.num_styles != 2:
        raise ValueError(
            "Counterfactual red/green evaluation requires exactly 2 styles"
        )

    base_ae = canonical_good.load_base_ae(
        args.base_checkpoint,
        device,
    )
    mean, std = load_scaler(args.scaler)
    probes = fit_semantic_probes(
        args.probe_embedding_dir / f"{args.probe_train_split}.npz",
        mean,
        std,
        args.max_iter,
        args.seed,
    )

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Scaler: {args.scaler}")
    print(f"Base AE: {args.base_checkpoint}")
    print(
        "Semantic probes trained on balanced split: "
        f"{args.probe_train_split}"
    )

    results = {}
    for split_name in args.splits:
        image_data = load_images(
            args.image_dir / f"{split_name}.npz",
            max_examples=args.max_examples,
            seed=args.seed,
        )
        results[split_name] = evaluate_split(
            model,
            base_ae,
            image_data,
            mean,
            std,
            probes,
            args.encode_batch_size,
            args.eval_batch_size,
            device,
        )

    print_results(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "counterfactual_style_results.json"
    csv_path = args.out_dir / "counterfactual_style_conditions.csv"
    payload = {
        "command": command_string(),
        "checkpoint": str(checkpoint_path),
        "model": model_family,
        "checkpoint_args": checkpoint["args"],
        "eval_args": args_to_dict(args),
        "style_mapping_method": (
            "best permutation from paired red/green selector argmax counts"
        ),
        "results": results,
    }
    save_json(json_path, payload)
    save_condition_csv(csv_path, results)

    print("")
    print(f"Saved detailed results to {json_path}")
    print(f"Saved condition summary to {csv_path}")


if __name__ == "__main__":
    main()
