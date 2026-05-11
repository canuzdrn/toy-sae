"""Foreground-colored MNIST generation utilities.

This adapts the core ColoredMNIST idea from the IRM reference implementation:
make color statistically correlated with the digit-derived label. For this
project we keep the original 10-way digit labels, color only the foreground
strokes, and emit explicit color/domain labels for later split-SAE training.

The code intentionally uses plain Python signatures so the data-generation path
is easy to read and modify while the project is still taking shape.
"""

import json
from pathlib import Path

import numpy as np


COLOR_NAMES = ("red", "green")
COLOR_TO_INDEX = {name: index for index, name in enumerate(COLOR_NAMES)}
SPLIT_SCHEMES = ("biased", "balanced", "reversed")

RGB_PALETTE = np.array(
    [
        [1.0, 0.0, 0.0],  # red
        [0.0, 1.0, 0.0],  # green
    ],
    dtype=np.float32,
)


def assign_colors(
    digits,
    *,
    scheme,
    bias_strength=0.9,
    seed=0,
):
    """Assign binary color labels to MNIST digits.

    Color labels follow ``0=red`` and ``1=green``. Under the biased scheme,
    digits 0-4 are green with probability ``bias_strength`` and digits 5-9 are
    red with that same probability. The reversed scheme swaps those
    associations. The balanced scheme uses 50/50 color assignment for every
    digit.
    """

    if scheme not in SPLIT_SCHEMES:
        raise ValueError(f"Unknown scheme {scheme!r}; expected one of {SPLIT_SCHEMES}")
    if not 0.5 <= bias_strength <= 1.0:
        raise ValueError("bias_strength must be in [0.5, 1.0]")

    digits = np.asarray(digits)
    if digits.ndim != 1:
        raise ValueError(f"digits must be a 1D array, got shape {digits.shape}")
    if np.any((digits < 0) | (digits > 9)):
        raise ValueError("digits must contain MNIST labels in [0, 9]")

    is_lower_digit = digits < 5
    if scheme == "balanced":
        p_green = np.full(digits.shape, 0.5, dtype=np.float32)
    elif scheme == "biased":
        p_green = np.where(is_lower_digit, bias_strength, 1.0 - bias_strength)
    else:
        p_green = np.where(is_lower_digit, 1.0 - bias_strength, bias_strength)

    rng = np.random.default_rng(seed)
    return (rng.random(digits.shape[0]) < p_green).astype(np.int64)


def colorize_foreground(
    grayscale_images,
    colors,
    *,
    palette=RGB_PALETTE,
):
    """Convert grayscale MNIST images to black-background RGB stroke images.

    The foreground intensity is preserved: a pixel with grayscale value ``v``
    becomes ``v * red`` or ``v * green``. Background pixels remain black because
    MNIST backgrounds are zero.
    """

    grayscale_images = np.asarray(grayscale_images)
    colors = np.asarray(colors)

    if grayscale_images.ndim != 3:
        raise ValueError(
            "grayscale_images must have shape (N, H, W), "
            f"got {grayscale_images.shape}"
        )
    if colors.shape != (grayscale_images.shape[0],):
        raise ValueError(
            "colors must have shape (N,), matching images; "
            f"got {colors.shape} for {grayscale_images.shape[0]} images"
        )
    if np.any((colors < 0) | (colors >= len(palette))):
        raise ValueError(f"colors must be integer labels in [0, {len(palette) - 1}]")

    images = grayscale_images.astype(np.float32)
    if images.max(initial=0.0) > 1.0:
        images /= 255.0

    rgb_images = images[..., None] * palette[colors][:, None, None, :]
    return np.moveaxis(rgb_images, -1, 1).astype(np.float32)


def build_colored_split(
    grayscale_images,
    digits,
    *,
    split,
    scheme,
    bias_strength=0.9,
    seed=0,
):
    """Build one ColoredMNIST split as arrays plus JSON metadata."""

    colors = assign_colors(
        digits,
        scheme=scheme,
        bias_strength=bias_strength,
        seed=seed,
    )
    images = colorize_foreground(grayscale_images, colors)
    digits = np.asarray(digits, dtype=np.int64)
    groups = (digits >= 5).astype(np.int64)

    metadata = {
        "split": split,
        "scheme": scheme,
        "bias_strength": bias_strength,
        "seed": seed,
        "num_examples": int(digits.shape[0]),
        "image_shape": [int(dim) for dim in images.shape[1:]],
        "color_names": list(COLOR_NAMES),
        "lower_digit_group": [0, 1, 2, 3, 4],
        "upper_digit_group": [5, 6, 7, 8, 9],
        "color_label_meaning": "0=red, 1=green",
        "array_layout": "NCHW, float32, values in [0, 1]",
    }

    return {
        "images": images,
        "digits": digits,
        "colors": colors,
        "digit_groups": groups,
        "metadata": json.dumps(metadata, sort_keys=True),
    }


def save_colored_split(path, split):
    """Save a generated split as a compressed NumPy archive."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **split)


def summarize_split(split):
    """Return compact counts/rates for sanity-checking a generated split."""

    digits = np.asarray(split["digits"])
    colors = np.asarray(split["colors"])
    lower = digits < 5
    upper = ~lower

    def _mean(mask):
        if not np.any(mask):
            return float("nan")
        return float(colors[mask].mean())

    return {
        "num_examples": int(digits.shape[0]),
        "green_rate_all": float(colors.mean()),
        "green_rate_digits_0_4": _mean(lower),
        "green_rate_digits_5_9": _mean(upper),
        "red_count": int((colors == COLOR_TO_INDEX["red"]).sum()),
        "green_count": int((colors == COLOR_TO_INDEX["green"]).sum()),
    }


def load_mnist_arrays(raw_dir, *, train, download):
    """Load MNIST through torchvision, returning NumPy image and digit arrays."""

    try:
        from torchvision.datasets import MNIST
    except Exception as exc:  # pragma: no cover - depends on local torch install
        raise RuntimeError(
            "torchvision is required to download/load MNIST. Install the project "
            "environment first, then rerun the generator."
        ) from exc

    dataset = MNIST(str(raw_dir), train=train, download=download)
    images = dataset.data.detach().cpu().numpy()
    digits = dataset.targets.detach().cpu().numpy()
    return images, digits
