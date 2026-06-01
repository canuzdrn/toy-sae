"""Generate the foreground-colored MNIST splits for the toy split-SAE project."""

import argparse
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_sae.datasets.colored_mnist import (
    build_colored_split,
    load_mnist_arrays,
    save_colored_split,
    summarize_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/colored_mnist"))
    parser.add_argument("--bias-strength", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def mnist_exists(raw_dir):
    return (raw_dir / "MNIST").exists()


def make_stratified_train_val(images, digits, val_size, seed):
    if val_size <= 0:
        raise ValueError("validation size must be greater than 0")
    if val_size >= len(digits):
        raise ValueError("validation size must be smaller than the number of train examples")

    rng = np.random.default_rng(seed)
    val_indices = []

    for digit in range(10):
        digit_indices = np.flatnonzero(digits == digit)
        rng.shuffle(digit_indices)
        digit_val_size = round(val_size * len(digit_indices) / len(digits))
        val_indices.append(digit_indices[:digit_val_size])

    val_indices = np.concatenate(val_indices)

    if len(val_indices) > val_size:
        val_indices = rng.choice(val_indices, size=val_size, replace=False)
    elif len(val_indices) < val_size:
        chosen = np.zeros(len(digits), dtype=bool)
        chosen[val_indices] = True
        remaining = np.flatnonzero(~chosen)
        extra = rng.choice(remaining, size=val_size - len(val_indices), replace=False)
        val_indices = np.concatenate([val_indices, extra])

    val_mask = np.zeros(len(digits), dtype=bool)
    val_mask[val_indices] = True
    train_indices = np.flatnonzero(~val_mask)
    val_indices = np.flatnonzero(val_mask)

    return (
        images[train_indices],
        digits[train_indices],
        images[val_indices],
        digits[val_indices],
    )


def main():
    args = parse_args()
    download = not mnist_exists(args.raw_dir)

    if download:
        print(f"MNIST not found under {args.raw_dir}/MNIST; downloading it now.")
    else:
        print(f"Using existing MNIST data under {args.raw_dir}/MNIST.")

    train_images, train_digits = load_mnist_arrays(
        args.raw_dir,
        train=True,
        download=download,
    )
    test_images, test_digits = load_mnist_arrays(
        args.raw_dir,
        train=False,
        download=download,
    )
    val_size = len(test_digits)
    train_partition_images, train_partition_digits, val_partition_images, val_partition_digits = (
        make_stratified_train_val(
            train_images,
            train_digits,
            val_size=val_size,
            seed=args.seed + 5,
        )
    )

    specs = [
        ("ae_train_balanced", train_partition_images, train_partition_digits, "balanced", args.seed + 11),
        ("ae_val_balanced", val_partition_images, val_partition_digits, "balanced", args.seed + 17),
        ("split_train_biased", train_partition_images, train_partition_digits, "biased", args.seed + 23),
        ("split_val_biased", val_partition_images, val_partition_digits, "biased", args.seed + 29),
        ("test_id_biased", test_images, test_digits, "biased", args.seed + 37),
        ("test_balanced", test_images, test_digits, "balanced", args.seed + 41),
        ("test_reversed", test_images, test_digits, "reversed", args.seed + 53),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, images, digits, scheme, seed in specs:
        split = build_colored_split(
            images,
            digits,
            split=name,
            scheme=scheme,
            bias_strength=args.bias_strength,
            seed=seed,
        )
        path = args.out_dir / f"{name}.npz"
        save_colored_split(path, split)
        summary = summarize_split(split)
        print(f"{path}")
        print(
            "  n={num_examples} red={red_count} green={green_count} "
            "green_rate_all={green_rate_all:.3f} "
            "green_rate_0_4={green_rate_digits_0_4:.3f} "
            "green_rate_5_9={green_rate_digits_5_9:.3f}".format(**summary)
        )


if __name__ == "__main__":
    main()
