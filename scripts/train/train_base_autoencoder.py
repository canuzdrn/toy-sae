"""Train the base convolutional autoencoder on balanced ColoredMNIST."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_sae.datasets.npz_dataset import ColoredMNISTNPZDataset
from toy_sae.models.base_autoencoder import ConvAutoencoder
from toy_sae.utils.script_utils import args_to_dict, save_json
from toy_sae.utils.torch_utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/colored_mnist"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/base_ae"))
    parser.add_argument("--train-split", default="ae_train_balanced")
    parser.add_argument("--val-split", default="ae_val_balanced")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--foreground-weight", type=float, default=10.0)
    parser.add_argument("--foreground-threshold", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num-reconstruction-examples", type=int, default=8)
    return parser.parse_args()


def make_loader(path, batch_size, shuffle, num_workers):
    dataset = ColoredMNISTNPZDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def torch_foreground_mask(images, foreground_threshold):
    return (images.max(dim=1, keepdim=True).values > foreground_threshold).float()


def numpy_foreground_mask(images, foreground_threshold):
    return images.max(axis=1, keepdims=True) > foreground_threshold


def foreground_weighted_mse(
    reconstructions,
    targets,
    foreground_weight,
    foreground_threshold,
):
    foreground = torch_foreground_mask(targets, foreground_threshold)
    weights = 1.0 + (foreground_weight - 1.0) * foreground
    squared_error = (reconstructions - targets) ** 2
    return (squared_error * weights).mean()


def plain_mse(reconstructions, targets):
    return ((reconstructions - targets) ** 2).mean()


def zero_reconstruction_baselines(dataset, foreground_weight, foreground_threshold):
    total_plain_loss = 0.0
    total_weighted_loss = 0.0
    total_values = 0
    chunk_size = 1024

    for start in range(0, len(dataset.images), chunk_size):
        images = dataset.images[start : start + chunk_size]
        squared_error = images ** 2
        foreground = numpy_foreground_mask(images, foreground_threshold)
        weights = 1.0 + (foreground_weight - 1.0) * foreground

        total_plain_loss += float(squared_error.sum())
        total_weighted_loss += float((squared_error * weights).sum())
        total_values += squared_error.size

    return total_plain_loss / total_values, total_weighted_loss / total_values


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    training,
    foreground_weight,
    foreground_threshold,
):
    if training:
        model.train()
    else:
        model.eval()

    total_weighted_loss = 0.0
    total_plain_loss = 0.0
    total_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False, disable=True):
            images = batch["image"].to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            reconstructions = model(images)
            loss = foreground_weighted_mse(
                reconstructions,
                images,
                foreground_weight,
                foreground_threshold,
            )
            mse = plain_mse(reconstructions, images)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            total_weighted_loss += loss.item() * batch_size
            total_plain_loss += mse.item() * batch_size
            total_examples += batch_size

    return total_weighted_loss / total_examples, total_plain_loss / total_examples


def save_checkpoint(
    path,
    model,
    optimizer,
    args,
    epoch,
    train_weighted_loss,
    val_weighted_loss,
    train_mse,
    val_mse,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "embedding_dim": args.embedding_dim,
            "epoch": epoch,
            "train_weighted_loss": train_weighted_loss,
            "val_weighted_loss": val_weighted_loss,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "args": args_to_dict(args),
        },
        path,
    )


def image_for_grid(image, scale):
    from PIL import Image

    image = image.detach().cpu().clamp(0.0, 1.0).numpy()
    image = np.moveaxis(image, 0, -1)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image.resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)


def save_reconstruction_grid(model, loader, device, path, num_examples):
    if num_examples <= 0:
        raise ValueError("--num-reconstruction-examples must be greater than 0")

    from PIL import Image, ImageDraw

    model.eval()
    batch = next(iter(loader))
    images = batch["image"][:num_examples].to(device)
    digits = batch["digit"][:num_examples].numpy()
    colors = batch["color"][:num_examples].numpy()

    with torch.no_grad():
        reconstructions = model(images)

    num_examples = images.shape[0]
    scale = 4
    tile_size = 28 * scale
    label_height = 30
    row_label_width = 80
    width = row_label_width + num_examples * tile_size
    height = label_height + 2 * tile_size
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    color_names = {0: "red", 1: "green"}
    for column in range(num_examples):
        x = row_label_width + column * tile_size
        title = f"{int(digits[column])}, {color_names[int(colors[column])]}"
        draw.text((x + 6, 8), title, fill="black")

        original = image_for_grid(images[column], scale)
        reconstruction = image_for_grid(reconstructions[column], scale)
        canvas.paste(original, (x, label_height))
        canvas.paste(reconstruction, (x, label_height + tile_size))

    draw.text((8, label_height + tile_size // 2 - 6), "original", fill="black")
    draw.text(
        (8, label_height + tile_size + tile_size // 2 - 6),
        "recon",
        fill="black",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    args = parse_args()
    if args.foreground_weight < 1.0:
        raise ValueError("--foreground-weight must be at least 1.0")
    if args.foreground_threshold < 0.0:
        raise ValueError("--foreground-threshold must be non-negative")

    set_seed(args.seed, deterministic=args.deterministic)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.checkpoint_dir / "config.json"
    save_json(config_path, args_to_dict(args))

    train_path = args.data_dir / f"{args.train_split}.npz"
    val_path = args.data_dir / f"{args.val_split}.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation split: {val_path}")

    device = get_device()
    print(f"Using device: {device}")
    print(f"Train split: {train_path}")
    print(f"Val split: {val_path}")

    train_loader = make_loader(train_path, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_path, args.batch_size, False, args.num_workers)
    print(f"Train examples: {len(train_loader.dataset)}")
    print(f"Val examples: {len(val_loader.dataset)}")
    print(f"Foreground weight: {args.foreground_weight}")
    print(f"Foreground threshold: {args.foreground_threshold}")

    train_zero_mse, train_zero_weighted = zero_reconstruction_baselines(
        train_loader.dataset,
        args.foreground_weight,
        args.foreground_threshold,
    )
    val_zero_mse, val_zero_weighted = zero_reconstruction_baselines(
        val_loader.dataset,
        args.foreground_weight,
        args.foreground_threshold,
    )
    print(
        "Zero reconstruction baseline | "
        f"train_mse={train_zero_mse:.6f} | "
        f"train_weighted={train_zero_weighted:.6f} | "
        f"val_mse={val_zero_mse:.6f} | "
        f"val_weighted={val_zero_weighted:.6f}"
    )

    model = ConvAutoencoder(embedding_dim=args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_model_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_weighted_loss, train_mse = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            True,
            args.foreground_weight,
            args.foreground_threshold,
        )
        val_weighted_loss, val_mse = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            False,
            args.foreground_weight,
            args.foreground_threshold,
        )
        history.append(
            {
                "epoch": epoch,
                "train_weighted_loss": train_weighted_loss,
                "val_weighted_loss": val_weighted_loss,
                "train_mse": train_mse,
                "val_mse": val_mse,
            }
        )

        print(
            f"epoch {epoch:03d} | "
            f"train_weighted={train_weighted_loss:.6f} | "
            f"val_weighted={val_weighted_loss:.6f} | "
            f"train_mse={train_mse:.6f} | "
            f"val_mse={val_mse:.6f}"
        )

        latest_path = args.checkpoint_dir / "latest.pt"
        save_checkpoint(
            latest_path,
            model,
            optimizer,
            args,
            epoch,
            train_weighted_loss,
            val_weighted_loss,
            train_mse,
            val_mse,
        )

        if val_weighted_loss < best_val_loss:
            best_val_loss = val_weighted_loss
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_path = args.checkpoint_dir / "best.pt"
            save_checkpoint(
                best_path,
                model,
                optimizer,
                args,
                epoch,
                train_weighted_loss,
                val_weighted_loss,
                train_mse,
                val_mse,
            )
            print(f"  saved new best checkpoint to {best_path}")

    history_path = args.checkpoint_dir / "history.json"
    save_json(history_path, history)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    reconstruction_path = args.checkpoint_dir / "reconstructions.png"
    save_reconstruction_grid(
        model,
        val_loader,
        device,
        reconstruction_path,
        args.num_reconstruction_examples,
    )
    print(f"Saved config to {config_path}")
    print(f"Saved training history to {history_path}")
    print(f"Saved reconstruction grid to {reconstruction_path}")


if __name__ == "__main__":
    main()
