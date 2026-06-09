"""Train canonical-good split SAE with a three-phase schedule.

The static canonical-good trainer starts with both branches active. This
scheduled variant is intended to prevent early z_bad takeover:

1. Good warm-up: train only D_good(z_good) against the canonical target.
2. Residual warm-up: introduce full reconstruction and the bad/color branch.
3. Full disentanglement: use the full canonical-good objective and ramp the
   good-branch color adversary if requested.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import scripts.train.train_split_sae_canonical_good as canonical_good
from scripts.train.train_split_sae_adv_update import (
    accuracy_from_logits,
    backbone_parameters,
    color_head_parameters,
    latent_stats,
    make_loader,
    make_model,
    set_backbone_trainable,
    set_color_heads_trainable,
    train_adversary_heads,
)
from toy_sae.utils.history_plots import plot_history_metrics
from toy_sae.utils.script_utils import args_to_dict, save_json
from toy_sae.utils.torch_utils import get_device, set_seed


def make_paired_embeddings(images, base_ae, device, *, mean, std, batch_size):
    red_embeddings = canonical_good.encode_images(
        base_ae,
        canonical_good.recolor_images(images, color_index=0),
        device,
        batch_size,
    )
    green_embeddings = canonical_good.encode_images(
        base_ae,
        canonical_good.recolor_images(images, color_index=1),
        device,
        batch_size,
    )
    red_embeddings = ((red_embeddings - mean) / std).astype(np.float32)
    green_embeddings = ((green_embeddings - mean) / std).astype(np.float32)
    return red_embeddings, green_embeddings


class CanonicalGoodPairedEmbeddingDataset(canonical_good.CanonicalGoodEmbeddingDataset):
    """Canonical-good dataset with counterfactual red/green paired embeddings."""

    def __init__(
        self,
        embedding_path,
        image_path,
        base_ae,
        device,
        *,
        mean=None,
        std=None,
        batch_size=512,
        canonical_target="green",
    ):
        super().__init__(
            embedding_path,
            image_path,
            base_ae,
            device,
            mean=mean,
            std=std,
            batch_size=batch_size,
            canonical_target=canonical_target,
        )

        image_data = np.load(image_path)
        images = image_data["images"].astype(np.float32)
        image_data.close()

        self.red_embeddings, self.green_embeddings = make_paired_embeddings(
            images,
            base_ae,
            device,
            mean=self.mean,
            std=self.std,
            batch_size=batch_size,
        )

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["red_embedding"] = torch.from_numpy(self.red_embeddings[index])
        item["green_embedding"] = torch.from_numpy(self.green_embeddings[index])
        return item


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/base_ae_embeddings"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/colored_mnist"))
    parser.add_argument("--base-ae-checkpoint", type=Path, default=Path("checkpoints/base_ae/best.pt"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/split_sae"))
    parser.add_argument("--train-split", default="split_train_biased")
    parser.add_argument("--val-split", default="split_val_biased")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--good-latent-dim", type=int, default=128)
    parser.add_argument("--bad-latent-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--canonical-encode-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-good-recon", type=float, default=0.0)
    parser.add_argument("--lambda-badcon", type=float, default=0.05)
    parser.add_argument("--lambda-sparse-good", type=float, default=0.0)
    parser.add_argument("--lambda-sparse-bad", type=float, default=0.0)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-dom", type=float, default=1.0)
    parser.add_argument("--lambda-canonical-good", type=float, default=1.0)
    parser.add_argument("--lambda-bad-residual", type=float, default=0.05)
    parser.add_argument(
        "--lambda-pair",
        type=float,
        default=0.0,
        help=(
            "Final Phase 3 weight for ||z_good(red) - z_good(green)||^2. "
            "If phase-specific pair weights are omitted, this value is used "
            "in every phase."
        ),
    )
    parser.add_argument(
        "--pair-variance-eps",
        type=float,
        default=1e-4,
        help=(
            "Epsilon for variance-normalized pair loss denominator. The pair "
            "objective is mean((z_red - z_green)^2 / (Var(z_pair) + eps))."
        ),
    )
    parser.add_argument(
        "--good-full-recon-grad-scale",
        type=float,
        default=0.25,
        help=(
            "Final phase scale for full-reconstruction gradient into the good "
            "branch. Phase 2 uses --phase2-good-full-recon-grad-scale."
        ),
    )
    parser.add_argument(
        "--detach-good-in-full-recon",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compatibility shortcut for the final phase. "
            "--detach-good-in-full-recon sets --good-full-recon-grad-scale 0.0; "
            "--no-detach-good-in-full-recon sets it to 1.0."
        ),
    )
    parser.add_argument(
        "--canonical-good-target",
        choices=["avg-red-green", "gray", "red", "green"],
        default="green",
    )
    parser.add_argument("--grl-lambda", type=float, default=5.0)
    parser.add_argument("--adversary-steps", type=int, default=5)
    parser.add_argument("--phase1-epochs", type=int, default=20)
    parser.add_argument(
        "--phase1-lambda-pair",
        type=float,
        default=None,
        help="Pair-invariance weight during Phase 1. Defaults to --lambda-pair.",
    )
    parser.add_argument("--phase2-epochs", type=int, default=30)
    parser.add_argument(
        "--phase2-lambda-pair",
        type=float,
        default=None,
        help="Pair-invariance weight during Phase 2. Defaults to --lambda-pair.",
    )
    parser.add_argument(
        "--phase2-good-full-recon-grad-scale",
        type=float,
        default=0.25,
        help="Good-branch full-reconstruction gradient scale during phase 2.",
    )
    parser.add_argument(
        "--phase2-lambda-bad-residual",
        type=float,
        default=0.0,
        help="Gentle bad-residual weight during phase 2.",
    )
    parser.add_argument(
        "--phase2-lambda-badcon",
        type=float,
        default=None,
        help="Bad contribution penalty during phase 2. Defaults to --lambda-badcon.",
    )
    parser.add_argument(
        "--phase2-lambda-adv",
        type=float,
        default=0.0,
        help="Good-branch adversary weight during phase 2; usually 0.",
    )
    parser.add_argument(
        "--phase2-lambda-dom-scale",
        type=float,
        default=0.5,
        help=(
            "Multiplier for --lambda-dom during phase 2. Keeps the bad color "
            "objective gentler while z_bad is first introduced."
        ),
    )
    parser.add_argument(
        "--phase2-adversary-steps",
        type=int,
        default=1,
        help="Color-head update steps per batch during phase 2.",
    )
    parser.add_argument(
        "--phase3-adv-ramp-epochs",
        type=int,
        default=20,
        help="Ramp lambda_adv and grl_lambda over the first N phase-3 epochs.",
    )
    parser.add_argument(
        "--phase3-bad-ramp-epochs",
        type=int,
        default=0,
        help=(
            "Ramp lambda_bad_residual and lambda_badcon from their phase-2 values "
            "to final values over the first N phase-3 epochs."
        ),
    )
    parser.add_argument(
        "--phase3-pair-ramp-epochs",
        type=int,
        default=0,
        help=(
            "Ramp lambda_pair from --phase2-lambda-pair to --lambda-pair over "
            "the first N phase-3 epochs."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def resolve_args(args):
    if args.detach_good_in_full_recon is not None:
        shortcut_scale = 0.0 if args.detach_good_in_full_recon else 1.0
        if args.good_full_recon_grad_scale != 0.25:
            raise ValueError(
                "Use either --good-full-recon-grad-scale or "
                "--detach-good-in-full-recon/--no-detach-good-in-full-recon, not both."
            )
        args.good_full_recon_grad_scale = shortcut_scale

    if args.phase2_lambda_badcon is None:
        args.phase2_lambda_badcon = args.lambda_badcon
    if args.phase1_lambda_pair is None:
        args.phase1_lambda_pair = args.lambda_pair
    if args.phase2_lambda_pair is None:
        args.phase2_lambda_pair = args.lambda_pair

    scale_names = [
        "good_full_recon_grad_scale",
        "phase2_good_full_recon_grad_scale",
        "phase2_lambda_dom_scale",
    ]
    for name in scale_names:
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0.0 and 1.0")

    non_negative_names = [
        "phase1_epochs",
        "phase2_epochs",
        "phase3_adv_ramp_epochs",
        "phase3_bad_ramp_epochs",
        "phase3_pair_ramp_epochs",
        "lambda_pair",
        "phase1_lambda_pair",
        "phase2_lambda_pair",
        "phase2_lambda_bad_residual",
        "phase2_lambda_badcon",
        "phase2_lambda_adv",
        "phase2_adversary_steps",
    ]
    for name in non_negative_names:
        value = getattr(args, name)
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")

    if args.pair_variance_eps <= 0.0:
        raise ValueError("--pair-variance-eps must be positive")

    if args.head_lr is None:
        args.head_lr = args.lr


def ramp_value(step, total_steps):
    if total_steps <= 0:
        return 1.0
    return min(max(step / total_steps, 0.0), 1.0)


def lerp(start, end, progress):
    return start + progress * (end - start)


def effective_args_for_epoch(args, epoch):
    effective = argparse.Namespace(**vars(args))
    phase2_start = args.phase1_epochs + 1
    phase3_start = args.phase1_epochs + args.phase2_epochs + 1

    if epoch < phase2_start:
        phase_name = "good_warmup"
        phase_index = 1
        bad_disabled = True
        adv_progress = 0.0
        bad_progress = 0.0
        pair_progress = 0.0
        effective.lambda_recon = 0.0
        effective.lambda_good_recon = 0.0
        effective.lambda_bad_residual = 0.0
        effective.lambda_badcon = 0.0
        effective.lambda_pair = args.phase1_lambda_pair
        effective.lambda_sparse_bad = 0.0
        effective.lambda_adv = 0.0
        effective.lambda_dom = 0.0
        effective.grl_lambda = 0.0
        effective.adversary_steps = 0
        effective.good_full_recon_grad_scale = 0.0
    elif epoch < phase3_start:
        phase_name = "residual_warmup"
        phase_index = 2
        bad_disabled = False
        adv_progress = 0.0
        bad_progress = 0.0
        pair_progress = 0.0
        effective.lambda_recon = args.lambda_recon
        effective.lambda_good_recon = 0.0
        effective.lambda_bad_residual = args.phase2_lambda_bad_residual
        effective.lambda_badcon = args.phase2_lambda_badcon
        effective.lambda_pair = args.phase2_lambda_pair
        effective.lambda_sparse_bad = args.lambda_sparse_bad
        effective.lambda_adv = args.phase2_lambda_adv
        effective.lambda_dom = args.lambda_dom * args.phase2_lambda_dom_scale
        effective.grl_lambda = args.grl_lambda if args.phase2_lambda_adv > 0.0 else 0.0
        effective.adversary_steps = (
            args.phase2_adversary_steps
            if effective.lambda_dom > 0.0 or effective.lambda_adv > 0.0
            else 0
        )
        effective.good_full_recon_grad_scale = args.phase2_good_full_recon_grad_scale
    else:
        phase_name = "full_disentanglement"
        phase_index = 3
        bad_disabled = False
        phase3_epoch = epoch - phase3_start + 1
        adv_progress = ramp_value(phase3_epoch, args.phase3_adv_ramp_epochs)
        bad_progress = ramp_value(phase3_epoch, args.phase3_bad_ramp_epochs)
        pair_progress = ramp_value(phase3_epoch, args.phase3_pair_ramp_epochs)
        effective.lambda_recon = args.lambda_recon
        effective.lambda_good_recon = args.lambda_good_recon
        effective.lambda_bad_residual = lerp(
            args.phase2_lambda_bad_residual,
            args.lambda_bad_residual,
            bad_progress,
        )
        effective.lambda_badcon = lerp(args.phase2_lambda_badcon, args.lambda_badcon, bad_progress)
        effective.lambda_pair = lerp(args.phase2_lambda_pair, args.lambda_pair, pair_progress)
        effective.lambda_sparse_bad = args.lambda_sparse_bad
        effective.lambda_adv = args.lambda_adv * adv_progress
        effective.lambda_dom = args.lambda_dom
        effective.grl_lambda = args.grl_lambda * adv_progress
        effective.adversary_steps = (
            args.adversary_steps
            if args.adversary_steps > 0 and (args.lambda_dom > 0.0 or effective.lambda_adv > 0.0)
            else 0
        )
        effective.good_full_recon_grad_scale = args.good_full_recon_grad_scale

    return effective, {
        "phase_name": phase_name,
        "phase_index": phase_index,
        "bad_disabled": bad_disabled,
        "lambda_recon_current": effective.lambda_recon,
        "lambda_canonical_current": effective.lambda_canonical_good,
        "lambda_bad_residual_current": effective.lambda_bad_residual,
        "lambda_badcon_current": effective.lambda_badcon,
        "lambda_pair_current": effective.lambda_pair,
        "lambda_adv_current": effective.lambda_adv,
        "lambda_dom_current": effective.lambda_dom,
        "grl_lambda_current": effective.grl_lambda,
        "adversary_steps_current": effective.adversary_steps,
        "good_full_recon_grad_scale_current": effective.good_full_recon_grad_scale,
        "adv_progress": adv_progress,
        "bad_progress": bad_progress,
        "pair_progress": pair_progress,
    }


def pair_losses(model, red_embeddings, green_embeddings, variance_eps):
    z_good_red, _ = model.encode(red_embeddings)
    z_good_green, _ = model.encode(green_embeddings)
    pair_diff_squared = (z_good_red - z_good_green) ** 2
    pair_variance = torch.cat([z_good_red, z_good_green], dim=0).var(
        dim=0,
        unbiased=False,
    )
    normalized_pair_mse = (
        pair_diff_squared / (pair_variance.detach().unsqueeze(0) + variance_eps)
    ).mean()
    pair_mse = pair_diff_squared.mean()
    pair_cosine = F.cosine_similarity(z_good_red, z_good_green, dim=1).mean()
    return normalized_pair_mse, pair_mse, pair_cosine, pair_variance.mean()


def run_epoch(model, loader, main_optimizer, head_optimizer, device, args, schedule, training):
    if training:
        model.train()
    else:
        model.eval()

    ce_loss = nn.CrossEntropyLoss()
    totals = {
        "total_loss": 0.0,
        "base_total_loss": 0.0,
        "recon_mse": 0.0,
        "good_recon_mse": 0.0,
        "canonical_good_mse": 0.0,
        "bad_residual_mse": 0.0,
        "badcon_loss": 0.0,
        "good_pair_loss": 0.0,
        "good_pair_mse": 0.0,
        "good_pair_cosine": 0.0,
        "good_pair_variance_mean": 0.0,
        "good_color_loss": 0.0,
        "bad_color_loss": 0.0,
        "good_sparsity": 0.0,
        "bad_sparsity": 0.0,
        "good_color_acc": 0.0,
        "bad_color_acc": 0.0,
        "z_good_mean_abs": 0.0,
        "z_bad_mean_abs": 0.0,
        "z_good_active_frac": 0.0,
        "z_bad_active_frac": 0.0,
        "z_good_active_count": 0.0,
        "z_bad_active_count": 0.0,
    }
    if training:
        totals["head_update_loss"] = 0.0
    total_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            embeddings = batch["embedding"].to(device)
            colors = batch["color"].to(device)
            canonical_embeddings = batch["canonical_good_embedding"].to(device)
            red_embeddings = batch["red_embedding"].to(device)
            green_embeddings = batch["green_embedding"].to(device)
            batch_size = embeddings.shape[0]

            head_update_loss = None
            if training:
                if args.adversary_steps > 0 and not schedule["bad_disabled"]:
                    head_update_loss = train_adversary_heads(
                        model,
                        embeddings,
                        colors,
                        head_optimizer,
                        ce_loss,
                        args,
                    )
                set_backbone_trainable(model, True)
                set_color_heads_trainable(model, False)
                main_optimizer.zero_grad(set_to_none=True)

            z_good, z_bad = model.encode(embeddings)
            good_reconstruction = model.decode_good(z_good)
            if schedule["bad_disabled"]:
                z_bad_for_stats = torch.zeros_like(z_bad)
                bad_reconstruction = torch.zeros_like(good_reconstruction)
                good_color_logits = None
                bad_color_logits = None
            else:
                z_bad_for_stats = z_bad
                bad_reconstruction = model.decode_bad(z_bad)
                good_color_logits = (
                    model.classify_good_color(z_good, grl_lambda=args.grl_lambda)
                    if args.lambda_adv > 0.0
                    else None
                )
                bad_color_logits = (
                    model.classify_bad_color(z_bad) if args.lambda_dom > 0.0 else None
                )

            reconstruction = good_reconstruction + bad_reconstruction
            good_for_recon_loss = (
                args.good_full_recon_grad_scale * good_reconstruction
                + (1.0 - args.good_full_recon_grad_scale) * good_reconstruction.detach()
            )
            reconstruction_for_loss = good_for_recon_loss + bad_reconstruction

            recon_mse = ((reconstruction - embeddings) ** 2).mean()
            recon_mse_for_loss = ((reconstruction_for_loss - embeddings) ** 2).mean()
            good_recon_mse = ((good_reconstruction - embeddings) ** 2).mean()
            canonical_good_mse = ((good_reconstruction - canonical_embeddings) ** 2).mean()
            bad_residual_mse = torch.zeros((), device=device)
            if args.lambda_bad_residual > 0.0 and not schedule["bad_disabled"]:
                bad_residual_target = (embeddings - canonical_embeddings).detach()
                bad_residual_mse = ((bad_reconstruction - bad_residual_target) ** 2).mean()

            badcon_loss = torch.zeros((), device=device)
            if args.lambda_badcon > 0.0 and not schedule["bad_disabled"]:
                badcon_loss = (bad_reconstruction ** 2).mean()

            good_pair_loss, good_pair_mse, good_pair_cosine, good_pair_variance_mean = pair_losses(
                model,
                red_embeddings,
                green_embeddings,
                args.pair_variance_eps,
            )
            good_sparsity = z_good.abs().mean()
            bad_sparsity = z_bad_for_stats.abs().mean()
            good_color_loss = (
                ce_loss(good_color_logits, colors)
                if good_color_logits is not None
                else torch.zeros((), device=device)
            )
            bad_color_loss = (
                ce_loss(bad_color_logits, colors)
                if bad_color_logits is not None
                else torch.zeros((), device=device)
            )
            good_color_acc = (
                accuracy_from_logits(good_color_logits, colors)
                if good_color_logits is not None
                else torch.zeros((), device=device)
            )
            bad_color_acc = (
                accuracy_from_logits(bad_color_logits, colors)
                if bad_color_logits is not None
                else torch.zeros((), device=device)
            )

            base_total_loss = (
                args.lambda_recon * recon_mse_for_loss
                + args.lambda_good_recon * good_recon_mse
                + args.lambda_badcon * badcon_loss
                + args.lambda_pair * good_pair_loss
                + args.lambda_sparse_good * good_sparsity
                + args.lambda_sparse_bad * bad_sparsity
                + args.lambda_adv * good_color_loss
                + args.lambda_dom * bad_color_loss
            )
            total_loss = (
                base_total_loss
                + args.lambda_canonical_good * canonical_good_mse
                + args.lambda_bad_residual * bad_residual_mse
            )

            if training:
                total_loss.backward()
                main_optimizer.step()
                set_color_heads_trainable(model, True)

            good_stats = latent_stats(z_good)
            bad_stats = latent_stats(z_bad_for_stats)
            total_examples += batch_size

            totals["total_loss"] += total_loss.item() * batch_size
            totals["base_total_loss"] += base_total_loss.item() * batch_size
            totals["recon_mse"] += recon_mse.item() * batch_size
            totals["good_recon_mse"] += good_recon_mse.item() * batch_size
            totals["canonical_good_mse"] += canonical_good_mse.item() * batch_size
            totals["bad_residual_mse"] += bad_residual_mse.item() * batch_size
            totals["badcon_loss"] += badcon_loss.item() * batch_size
            totals["good_pair_loss"] += good_pair_loss.item() * batch_size
            totals["good_pair_mse"] += good_pair_mse.item() * batch_size
            totals["good_pair_cosine"] += good_pair_cosine.item() * batch_size
            totals["good_pair_variance_mean"] += good_pair_variance_mean.item() * batch_size
            totals["good_color_loss"] += good_color_loss.item() * batch_size
            totals["bad_color_loss"] += bad_color_loss.item() * batch_size
            totals["good_sparsity"] += good_sparsity.item() * batch_size
            totals["bad_sparsity"] += bad_sparsity.item() * batch_size
            totals["good_color_acc"] += good_color_acc.item() * batch_size
            totals["bad_color_acc"] += bad_color_acc.item() * batch_size
            totals["z_good_mean_abs"] += good_stats["mean_abs"].item() * batch_size
            totals["z_bad_mean_abs"] += bad_stats["mean_abs"].item() * batch_size
            totals["z_good_active_frac"] += good_stats["active_fraction"].item() * batch_size
            totals["z_bad_active_frac"] += bad_stats["active_fraction"].item() * batch_size
            totals["z_good_active_count"] += good_stats["active_count"].item() * batch_size
            totals["z_bad_active_count"] += bad_stats["active_count"].item() * batch_size
            if training and head_update_loss is not None:
                totals["head_update_loss"] += head_update_loss * batch_size

    set_backbone_trainable(model, True)
    set_color_heads_trainable(model, True)
    return {key: value / total_examples for key, value in totals.items()}


def add_schedule_metrics(metrics, schedule):
    metrics = dict(metrics)
    for key, value in schedule.items():
        if isinstance(value, bool):
            metrics[key] = float(value)
        elif isinstance(value, (int, float)):
            metrics[key] = value
    return metrics


def save_checkpoint(path, model, main_optimizer, head_optimizer, args, epoch, train_metrics, val_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": main_optimizer.state_dict(),
            "main_optimizer_state_dict": main_optimizer.state_dict(),
            "head_optimizer_state_dict": head_optimizer.state_dict(),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "args": args_to_dict(
                args,
                extra={"training_procedure": "scheduled_canonical_good_reconstruction"},
            ),
        },
        path,
    )


def main():
    args = parse_args()
    resolve_args(args)

    set_seed(args.seed, deterministic=args.deterministic)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_embedding_path = args.embedding_dir / f"{args.train_split}.npz"
    val_embedding_path = args.embedding_dir / f"{args.val_split}.npz"
    train_image_path = args.image_dir / f"{args.train_split}.npz"
    val_image_path = args.image_dir / f"{args.val_split}.npz"
    for path in [train_embedding_path, val_embedding_path, train_image_path, val_image_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    device = get_device()
    print(f"Using device: {device}")
    print(f"Train embeddings: {train_embedding_path}")
    print(f"Val embeddings: {val_embedding_path}")
    print(f"Train images: {train_image_path}")
    print(f"Val images: {val_image_path}")
    print(f"Base AE checkpoint: {args.base_ae_checkpoint}")
    print("Model: shared-encoder split SAE")
    print("Training procedure: scheduled canonical-good reconstruction")
    print(
        "Schedule: "
        f"phase1={args.phase1_epochs} epoch(s), "
        f"phase2={args.phase2_epochs} epoch(s), "
        f"phase3_adv_ramp={args.phase3_adv_ramp_epochs} epoch(s), "
        f"phase3_bad_ramp={args.phase3_bad_ramp_epochs} epoch(s)"
    )

    base_ae = canonical_good.load_base_ae(args.base_ae_checkpoint, device)
    train_dataset = CanonicalGoodPairedEmbeddingDataset(
        train_embedding_path,
        train_image_path,
        base_ae,
        device,
        batch_size=args.canonical_encode_batch_size,
        canonical_target=args.canonical_good_target,
    )
    val_dataset = CanonicalGoodPairedEmbeddingDataset(
        val_embedding_path,
        val_image_path,
        base_ae,
        device,
        mean=train_dataset.mean,
        std=train_dataset.std,
        batch_size=args.canonical_encode_batch_size,
        canonical_target=args.canonical_good_target,
    )

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")
    print(f"Embedding scaler mean={train_dataset.mean.mean():.4f} std={train_dataset.std.mean():.4f}")
    print(f"canonical_good_target={args.canonical_good_target}")
    print(f"bad_latent_dim={args.bad_latent_dim}")
    print(
        "lambda_pair schedule: "
        f"phase1={args.phase1_lambda_pair}, "
        f"phase2={args.phase2_lambda_pair}, "
        f"phase3={args.lambda_pair}, "
        f"phase3_ramp={args.phase3_pair_ramp_epochs}"
    )

    model = make_model(args).to(device)
    main_optimizer = torch.optim.AdamW(
        backbone_parameters(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    head_optimizer = torch.optim.AdamW(
        color_head_parameters(model),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    config_path = args.checkpoint_dir / "config.json"
    history_path = args.checkpoint_dir / "history.json"
    scaler_path = args.checkpoint_dir / "embedding_scaler.npz"
    config = args_to_dict(
        args,
        extra={"training_procedure": "scheduled_canonical_good_reconstruction"},
    )
    save_json(config_path, config)
    canonical_good.save_scaler(scaler_path, train_dataset)

    best_val_recon = float("inf")
    best_phase3_val_recon = float("inf")
    history = []
    phase1_end_epoch = args.phase1_epochs
    phase2_end_epoch = args.phase1_epochs + args.phase2_epochs

    for epoch in range(1, args.epochs + 1):
        current_args, schedule = effective_args_for_epoch(args, epoch)
        train_metrics = run_epoch(
            model,
            train_loader,
            main_optimizer,
            head_optimizer,
            device,
            current_args,
            schedule,
            True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            None,
            device,
            current_args,
            schedule,
            False,
        )
        train_metrics = add_schedule_metrics(train_metrics, schedule)
        val_metrics = add_schedule_metrics(val_metrics, schedule)
        history.append(
            {
                "epoch": epoch,
                "schedule": schedule,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        message = (
            f"epoch {epoch:03d} | "
            f"phase={schedule['phase_name']} "
            f"bad_disabled={int(schedule['bad_disabled'])} "
            f"lambda_recon={schedule['lambda_recon_current']:.3g} "
            f"lambda_bad_residual={schedule['lambda_bad_residual_current']:.3g} "
            f"lambda_badcon={schedule['lambda_badcon_current']:.3g} "
            f"lambda_pair={schedule['lambda_pair_current']:.3g} "
            f"lambda_adv={schedule['lambda_adv_current']:.3g} "
            f"lambda_dom={schedule['lambda_dom_current']:.3g} "
            f"good_grad={schedule['good_full_recon_grad_scale_current']:.3g} | "
            f"train_loss={train_metrics['total_loss']:.6f} "
            f"val_loss={val_metrics['total_loss']:.6f} | "
            f"val_recon={val_metrics['recon_mse']:.6f} "
            f"val_good_canonical={val_metrics['canonical_good_mse']:.6f} "
            f"val_bad_residual={val_metrics['bad_residual_mse']:.6f} "
            f"val_badcon={val_metrics['badcon_loss']:.6f} "
            f"val_good_pair_loss={val_metrics['good_pair_loss']:.6f} "
            f"val_good_pair_mse={val_metrics['good_pair_mse']:.6f} "
            f"val_good_pair_cos={val_metrics['good_pair_cosine']:.4f} "
            f"val_good_color_acc={val_metrics['good_color_acc']:.4f} "
            f"val_bad_color_acc={val_metrics['bad_color_acc']:.4f} | "
            f"val_z_good_active={val_metrics['z_good_active_frac']:.4f} "
            f"({val_metrics['z_good_active_count']:.1f}/{args.good_latent_dim}) "
            f"val_z_bad_active={val_metrics['z_bad_active_frac']:.4f} "
            f"({val_metrics['z_bad_active_count']:.1f}/{args.bad_latent_dim})"
        )
        print(message)

        if args.phase1_epochs > 0 and epoch == phase1_end_epoch:
            save_checkpoint(
                args.checkpoint_dir / "phase1_end.pt",
                model,
                main_optimizer,
                head_optimizer,
                args,
                epoch,
                train_metrics,
                val_metrics,
            )
            print(f"  saved phase 1 endpoint checkpoint to {args.checkpoint_dir / 'phase1_end.pt'}")

        if args.phase2_epochs > 0 and epoch == phase2_end_epoch:
            save_checkpoint(
                args.checkpoint_dir / "phase2_end.pt",
                model,
                main_optimizer,
                head_optimizer,
                args,
                epoch,
                train_metrics,
                val_metrics,
            )
            print(f"  saved phase 2 endpoint checkpoint to {args.checkpoint_dir / 'phase2_end.pt'}")

        if val_metrics["recon_mse"] < best_val_recon:
            best_val_recon = val_metrics["recon_mse"]
            save_checkpoint(
                args.checkpoint_dir / "best_recon.pt",
                model,
                main_optimizer,
                head_optimizer,
                args,
                epoch,
                train_metrics,
                val_metrics,
            )
            print(f"  saved new best reconstruction checkpoint to {args.checkpoint_dir / 'best_recon.pt'}")

        if schedule["phase_index"] == 3 and val_metrics["recon_mse"] < best_phase3_val_recon:
            best_phase3_val_recon = val_metrics["recon_mse"]
            save_checkpoint(
                args.checkpoint_dir / "best_phase3_recon.pt",
                model,
                main_optimizer,
                head_optimizer,
                args,
                epoch,
                train_metrics,
                val_metrics,
            )
            print(
                "  saved new best phase 3 reconstruction checkpoint to "
                f"{args.checkpoint_dir / 'best_phase3_recon.pt'}"
            )

    save_json(history_path, history)
    print(f"Saved config to {config_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved history to {history_path}")

    if not args.no_plots:
        plot_dir = args.checkpoint_dir / "plots"
        plot_result = plot_history_metrics(history_path, out_dir=plot_dir)
        print(f"Saved {len(plot_result['saved_paths'])} history plots to {plot_dir}")


if __name__ == "__main__":
    main()
