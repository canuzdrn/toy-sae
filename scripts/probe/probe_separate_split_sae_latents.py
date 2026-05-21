#!/usr/bin/env python
"""Post-hoc probes for z_good and z_bad from a separate-encoder split SAE."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
from toy_sae.utils.split_sae_loading import load_split_sae_model
from toy_sae.utils.torch_utils import get_device


DEFAULT_EVAL_SPLITS = [
    "split_val_biased",
    "test_id_biased",
    "test_balanced",
    "test_reversed",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/base_ae_embeddings"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/separate_split_sae/best_recon.pt"),
    )
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/separate_split_sae_latent_probes"))
    parser.add_argument("--train-split", default="split_train_biased")
    parser.add_argument("--eval-splits", nargs="+", default=DEFAULT_EVAL_SPLITS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
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
    total_examples = 0
    total_recon_mse = 0.0
    total_good_recon_mse = 0.0
    total_badcon_loss = 0.0
    good_head_correct = 0
    bad_head_correct = 0

    with torch.no_grad():
        for batch in tqdm(loader, leave=False, disable=True):
            embeddings = batch["embedding"].to(device)
            colors = batch["color"].to(device)
            outputs = model(embeddings, grl_lambda=0.0)
            reconstruction = outputs["reconstruction"]
            good_reconstruction = outputs["good_reconstruction"]
            bad_reconstruction = outputs["bad_reconstruction"]
            z_good = outputs["z_good"]
            z_bad = outputs["z_bad"]
            good_color_logits = model.classify_good_color_no_grl(z_good)
            bad_color_logits = model.classify_bad_color(z_bad)

            batch_size = embeddings.shape[0]
            total_examples += batch_size
            total_recon_mse += ((reconstruction - embeddings) ** 2).mean().item() * batch_size
            total_good_recon_mse += ((good_reconstruction - embeddings) ** 2).mean().item() * batch_size
            total_badcon_loss += (bad_reconstruction ** 2).mean().item() * batch_size
            good_head_correct += (good_color_logits.argmax(dim=1) == colors).sum().item()
            bad_head_correct += (bad_color_logits.argmax(dim=1) == colors).sum().item()

            z_good_batches.append(z_good.cpu().numpy())
            z_bad_batches.append(z_bad.cpu().numpy())

    return {
        "z_good": np.concatenate(z_good_batches, axis=0).astype(np.float32),
        "z_bad": np.concatenate(z_bad_batches, axis=0).astype(np.float32),
        "digits": dataset.digits,
        "digit_groups": dataset.digit_groups,
        "colors": dataset.colors,
        "diagnostics": {
            "recon_mse": float(total_recon_mse / total_examples),
            "good_recon_mse": float(total_good_recon_mse / total_examples),
            "badcon_loss": float(total_badcon_loss / total_examples),
            "good_head_color_acc": float(good_head_correct / total_examples),
            "bad_head_color_acc": float(bad_head_correct / total_examples),
        },
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


def fit_probes(latents, max_iter, seed):
    probes = {}
    for latent_name in ["z_good", "z_bad"]:
        probes[latent_name] = {}
        for label_name in ["digits", "digit_groups", "colors"]:
            probe = make_probe(max_iter, seed)
            probe.fit(latents[latent_name], latents[label_name])
            probes[latent_name][label_name] = probe
    return probes


def evaluate_probes(probes, latents):
    results = {}
    for latent_name in ["z_good", "z_bad"]:
        results[latent_name] = {}
        for label_name in ["digits", "digit_groups", "colors"]:
            predictions = probes[latent_name][label_name].predict(latents[latent_name])
            accuracy = accuracy_score(latents[label_name], predictions)
            results[latent_name][label_name] = float(accuracy)
    return results


def latent_summary(latents):
    summary = {}
    for latent_name in ["z_good", "z_bad"]:
        values = latents[latent_name]
        summary[latent_name] = {
            "mean_abs": float(np.abs(values).mean()),
            "active_fraction": float((values > 0).mean()),
            "active_count": float((values > 0).mean() * values.shape[1]),
        }
    return summary


def print_results(results):
    print("Diagnostics")
    print(
        "split | recon_mse | good_branch_mse | badcon | "
        "good_head_color_acc | bad_head_color_acc | z_good_active | z_bad_active"
    )
    print("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for split_name, split_results in results.items():
        diagnostics = split_results["diagnostics"]
        z_good = split_results["latent_summary"]["z_good"]
        z_bad = split_results["latent_summary"]["z_bad"]
        print(
            f"{split_name} | "
            f"{diagnostics['recon_mse']:.6f} | "
            f"{diagnostics['good_recon_mse']:.6f} | "
            f"{diagnostics['badcon_loss']:.6f} | "
            f"{diagnostics['good_head_color_acc']:.4f} | "
            f"{diagnostics['bad_head_color_acc']:.4f} | "
            f"{z_good['active_count']:.1f} | "
            f"{z_bad['active_count']:.1f}"
        )

    print("")
    print("Post-hoc probe accuracy")
    print("split | latent | digit_acc | group_acc | color_acc")
    print("--- | --- | ---: | ---: | ---:")
    for split_name, split_results in results.items():
        for latent_name in ["z_good", "z_bad"]:
            row = split_results["probe_accuracy"][latent_name]
            print(
                f"{split_name} | {latent_name} | "
                f"{row['digits']:.4f} | "
                f"{row['digit_groups']:.4f} | "
                f"{row['colors']:.4f}"
            )


def main():
    args = parse_args()
    if args.scaler is None:
        args.scaler = args.checkpoint.parent / "embedding_scaler.npz"

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing separate split-SAE checkpoint: {args.checkpoint}")
    if not args.scaler.exists():
        raise FileNotFoundError(f"Missing embedding scaler: {args.scaler}")

    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint)
    model = load_split_sae_model(checkpoint, device, model_family="separate")
    model_name = checkpoint["args"].get("model_name", "separate_encoder_sae")
    mean, std = load_scaler(args.scaler)

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {model_name}")
    print(f"Scaler: {args.scaler}")
    print(f"Probe train split: {args.train_split}")

    train_dataset = load_split(args.embedding_dir, args.train_split, mean, std)
    train_latents = extract_latents(
        model,
        train_dataset,
        args.batch_size,
        args.num_workers,
        device,
    )
    probes = fit_probes(train_latents, args.max_iter, args.seed)

    all_eval_splits = [args.train_split]
    for split_name in args.eval_splits:
        if split_name not in all_eval_splits:
            all_eval_splits.append(split_name)

    results = {}
    for split_name in all_eval_splits:
        if split_name == args.train_split:
            latents = train_latents
        else:
            dataset = load_split(args.embedding_dir, split_name, mean, std)
            latents = extract_latents(
                model,
                dataset,
                args.batch_size,
                args.num_workers,
                device,
            )
        results[split_name] = {
            "diagnostics": latents["diagnostics"],
            "probe_accuracy": evaluate_probes(probes, latents),
            "latent_summary": latent_summary(latents),
        }

    print("")
    print_results(results)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "probe_results.json"
    payload = {
        "command": command_string(),
        "checkpoint": str(args.checkpoint),
        "model": model_name,
        "checkpoint_args": checkpoint["args"],
        "probe_args": args_to_dict(args),
        "scaler": str(args.scaler),
        "embedding_dir": str(args.embedding_dir),
        "train_split": args.train_split,
        "results": results,
    }
    save_json(results_path, payload)
    print("")
    print(f"Saved separate split-SAE latent probe results to {results_path}")


if __name__ == "__main__":
    main()
