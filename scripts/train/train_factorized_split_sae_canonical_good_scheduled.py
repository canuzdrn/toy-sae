"""Train a factorized split SAE with canonical and paired invariance losses.

This is an independent architecture trial. The good branch reconstructs a
canonical content embedding. The bad branch selects a mixture of learned
content-conditioned style transforms; it cannot emit a standalone 64D
reconstruction.
"""

import argparse
from pathlib import Path
import sys

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import scripts.train.train_split_sae_canonical_good as canonical_good
import scripts.train.train_split_sae_canonical_good_scheduled as scheduled
from scripts.train.train_split_sae_adv_update import (
    accuracy_from_logits,
    latent_stats,
    make_loader,
)
from toy_sae.models.factorized_split_sae import FactorizedSplitSparseAutoencoder
from toy_sae.utils.history_plots import plot_history_metrics
from toy_sae.utils.script_utils import args_to_dict, save_json
from toy_sae.utils.torch_utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", type=Path, default=Path("data/base_ae_embeddings"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/colored_mnist"))
    parser.add_argument(
        "--base-ae-checkpoint",
        type=Path,
        default=Path("checkpoints/base_ae/best.pt"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/factorized_split_sae"),
    )
    parser.add_argument("--train-split", default="split_train_biased")
    parser.add_argument("--val-split", default="split_val_biased")

    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--good-latent-dim", type=int, default=128)
    parser.add_argument("--bad-latent-dim", type=int, default=4)
    parser.add_argument("--num-styles", type=int, default=2)
    parser.add_argument("--style-hidden-dim", type=int, default=16)
    parser.add_argument("--style-temperature", type=float, default=1.0)
    parser.add_argument("--style-init-scale", type=float, default=1e-3)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--canonical-encode-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-good-recon", type=float, default=0.0)
    parser.add_argument(
        "--lambda-badcon",
        type=float,
        default=0.1,
        help="Penalty on the style-induced embedding change.",
    )
    parser.add_argument("--lambda-sparse-good", type=float, default=0.0)
    parser.add_argument("--lambda-sparse-bad", type=float, default=0.0)
    parser.add_argument("--lambda-adv", type=float, default=0.0)
    parser.add_argument(
        "--lambda-dom",
        type=float,
        default=1.0,
        help=(
            "Additional frozen-head color objective on z_bad. The style "
            "selection loss already supervises decoder color, so this term is "
            "optional and mainly preserves comparability with earlier runs."
        ),
    )
    parser.add_argument(
        "--lambda-style-selection",
        type=float,
        default=1.0,
        help=(
            "Cross-entropy weight aligning the decoder style selector with "
            "the red/green label."
        ),
    )
    parser.add_argument("--lambda-canonical-good", type=float, default=1.0)
    parser.add_argument(
        "--lambda-bad-residual",
        type=float,
        default=0.0,
        help=(
            "MSE weight matching the content-conditioned style contribution "
            "to original minus canonical embedding."
        ),
    )
    parser.add_argument("--lambda-pair", type=float, default=0.5)
    parser.add_argument("--pair-variance-eps", type=float, default=1e-4)
    parser.add_argument(
        "--good-full-recon-grad-scale",
        type=float,
        default=0.25,
        help=(
            "Final direct full-reconstruction gradient scale into z_good and "
            "the content decoder. Because the encoder trunk is shared, style "
            "losses can still update the trunk through z_bad."
        ),
    )
    parser.add_argument(
        "--canonical-good-target",
        choices=["avg-red-green", "gray", "red", "green"],
        default="green",
    )
    parser.add_argument("--grl-lambda", type=float, default=0.0)
    parser.add_argument("--adversary-steps", type=int, default=5)

    parser.add_argument("--phase1-epochs", type=int, default=10)
    parser.add_argument("--phase1-lambda-pair", type=float, default=0.1)
    parser.add_argument("--phase2-epochs", type=int, default=30)
    parser.add_argument("--phase2-lambda-pair", type=float, default=0.5)
    parser.add_argument(
        "--phase2-good-full-recon-grad-scale",
        type=float,
        default=0.25,
    )
    parser.add_argument("--phase2-lambda-bad-residual", type=float, default=0.0)
    parser.add_argument("--phase2-lambda-badcon", type=float, default=0.05)
    parser.add_argument("--phase2-lambda-adv", type=float, default=0.0)
    parser.add_argument("--phase2-lambda-dom-scale", type=float, default=0.5)
    parser.add_argument(
        "--phase2-lambda-style-selection",
        type=float,
        default=0.5,
    )
    parser.add_argument("--phase2-adversary-steps", type=int, default=1)
    parser.add_argument("--phase3-adv-ramp-epochs", type=int, default=0)
    parser.add_argument("--phase3-bad-ramp-epochs", type=int, default=10)
    parser.add_argument("--phase3-pair-ramp-epochs", type=int, default=10)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def resolve_args(args):
    if args.head_lr is None:
        args.head_lr = args.lr

    bounded_scales = [
        "good_full_recon_grad_scale",
        "phase2_good_full_recon_grad_scale",
        "phase2_lambda_dom_scale",
    ]
    for name in bounded_scales:
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")

    non_negative = [
        "bad_latent_dim",
        "style_hidden_dim",
        "style_init_scale",
        "phase1_epochs",
        "phase2_epochs",
        "phase3_adv_ramp_epochs",
        "phase3_bad_ramp_epochs",
        "phase3_pair_ramp_epochs",
        "phase2_adversary_steps",
        "adversary_steps",
        "lambda_pair",
        "phase1_lambda_pair",
        "phase2_lambda_pair",
        "lambda_bad_residual",
        "phase2_lambda_bad_residual",
        "lambda_badcon",
        "phase2_lambda_badcon",
        "lambda_style_selection",
        "phase2_lambda_style_selection",
    ]
    for name in non_negative:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")

    if args.num_styles < 2:
        raise ValueError("--num-styles must be at least 2")
    if (
        args.lambda_style_selection > 0.0
        or args.phase2_lambda_style_selection > 0.0
    ) and args.num_styles != 2:
        raise ValueError(
            "--lambda-style-selection requires --num-styles 2 for red/green labels"
        )
    if args.style_temperature <= 0.0:
        raise ValueError("--style-temperature must be positive")
    if args.pair_variance_eps <= 0.0:
        raise ValueError("--pair-variance-eps must be positive")


def set_requires_grad(module, value):
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def set_backbone_trainable(model, value):
    for module in [
        model.encoder,
        model.good_encoder,
        model.bad_encoder,
        model.good_decoder,
        model.style_selector,
    ]:
        set_requires_grad(module, value)
    model.style_residual_matrices.requires_grad_(value)
    model.style_residual_biases.requires_grad_(value)


def set_color_heads_trainable(model, value):
    set_requires_grad(model.good_color_head, value)
    set_requires_grad(model.bad_color_head, value)


def backbone_parameters(model):
    parameters = []
    for module in [
        model.encoder,
        model.good_encoder,
        model.bad_encoder,
        model.good_decoder,
        model.style_selector,
    ]:
        parameters.extend(module.parameters())
    parameters.extend(
        [
            model.style_residual_matrices,
            model.style_residual_biases,
        ]
    )
    return parameters


def color_head_parameters(model):
    return list(model.good_color_head.parameters()) + list(model.bad_color_head.parameters())


def train_adversary_heads(model, embeddings, colors, optimizer, ce_loss, args):
    if (
        args.adversary_steps <= 0
        or (args.lambda_adv <= 0.0 and args.lambda_dom <= 0.0)
    ):
        return None

    set_backbone_trainable(model, False)
    set_color_heads_trainable(model, True)
    with torch.no_grad():
        z_good, z_bad = model.encode(embeddings)

    total_loss = 0.0
    for _ in range(args.adversary_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device=embeddings.device)
        if args.lambda_adv > 0.0:
            good_logits = model.classify_good_color_no_grl(z_good)
            loss = loss + args.lambda_adv * ce_loss(good_logits, colors)
        if args.lambda_dom > 0.0:
            bad_logits = model.classify_bad_color(z_bad)
            loss = loss + args.lambda_dom * ce_loss(bad_logits, colors)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / args.adversary_steps


def make_model(args):
    return FactorizedSplitSparseAutoencoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        good_latent_dim=args.good_latent_dim,
        bad_latent_dim=args.bad_latent_dim,
        num_styles=args.num_styles,
        style_hidden_dim=args.style_hidden_dim,
        style_temperature=args.style_temperature,
        style_init_scale=args.style_init_scale,
    )


def effective_args_for_epoch(args, epoch):
    effective, schedule = scheduled.effective_args_for_epoch(args, epoch)
    if schedule["phase_index"] == 1:
        effective.lambda_style_selection = 0.0
    elif schedule["phase_index"] == 2:
        effective.lambda_style_selection = args.phase2_lambda_style_selection
    else:
        effective.lambda_style_selection = args.lambda_style_selection
    schedule["lambda_style_selection_current"] = (
        effective.lambda_style_selection
    )
    return effective, schedule


def run_epoch(model, loader, main_optimizer, head_optimizer, device, args, schedule, training):
    model.train(training)
    ce_loss = nn.CrossEntropyLoss()
    metric_names = [
        "total_loss",
        "base_total_loss",
        "recon_mse",
        "good_recon_mse",
        "canonical_good_mse",
        "bad_residual_mse",
        "badcon_loss",
        "good_pair_loss",
        "good_pair_mse",
        "good_pair_cosine",
        "good_pair_variance_mean",
        "good_color_loss",
        "bad_color_loss",
        "style_selection_loss",
        "style_selection_acc",
        "good_sparsity",
        "bad_sparsity",
        "good_color_acc",
        "bad_color_acc",
        "style_entropy",
        "style_max_probability",
        "style_contribution_mse",
        "z_good_mean_abs",
        "z_bad_mean_abs",
        "z_good_active_frac",
        "z_bad_active_frac",
        "z_good_active_count",
        "z_bad_active_count",
    ]
    totals = {name: 0.0 for name in metric_names}
    for style_index in range(model.num_styles):
        totals[f"style_usage_{style_index}"] = 0.0
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
            outputs = model.decode(
                z_good,
                z_bad,
                good_grad_scale=args.good_full_recon_grad_scale,
                style_enabled=not schedule["bad_disabled"],
            )
            reconstruction = outputs["reconstruction"]
            good_reconstruction = outputs["good_reconstruction"]
            style_contribution = outputs["style_contribution_for_loss"]
            style_weights = outputs["style_weights"]

            if schedule["bad_disabled"]:
                z_bad_for_stats = torch.zeros_like(z_bad)
                good_color_logits = None
                bad_color_logits = None
            else:
                z_bad_for_stats = z_bad
                good_color_logits = (
                    model.classify_good_color(z_good, args.grl_lambda)
                    if args.lambda_adv > 0.0
                    else None
                )
                bad_color_logits = (
                    model.classify_bad_color(z_bad)
                    if args.lambda_dom > 0.0
                    else None
                )

            recon_mse = ((reconstruction - embeddings) ** 2).mean()
            good_recon_mse = ((good_reconstruction - embeddings) ** 2).mean()
            canonical_good_mse = (
                (good_reconstruction - canonical_embeddings) ** 2
            ).mean()

            bad_residual_mse = torch.zeros((), device=device)
            if args.lambda_bad_residual > 0.0 and not schedule["bad_disabled"]:
                residual_target = (embeddings - canonical_embeddings).detach()
                bad_residual_mse = (
                    (style_contribution - residual_target) ** 2
                ).mean()

            badcon_loss = torch.zeros((), device=device)
            if args.lambda_badcon > 0.0 and not schedule["bad_disabled"]:
                badcon_loss = (style_contribution ** 2).mean()

            (
                good_pair_loss,
                good_pair_mse,
                good_pair_cosine,
                good_pair_variance_mean,
            ) = scheduled.pair_losses(
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
            style_selection_loss = torch.zeros((), device=device)
            style_selection_acc = torch.zeros((), device=device)
            if (
                args.lambda_style_selection > 0.0
                and not schedule["bad_disabled"]
            ):
                style_selection_loss = ce_loss(outputs["style_logits"], colors)
                style_selection_acc = accuracy_from_logits(
                    outputs["style_logits"],
                    colors,
                )

            style_entropy = -(
                style_weights * style_weights.clamp_min(1e-8).log()
            ).sum(dim=1).mean()
            style_max_probability = style_weights.max(dim=1).values.mean()
            style_contribution_mse = (
                outputs["style_contribution"].detach() ** 2
            ).mean()

            base_total_loss = (
                args.lambda_recon * recon_mse
                + args.lambda_good_recon * good_recon_mse
                + args.lambda_badcon * badcon_loss
                + args.lambda_pair * good_pair_loss
                + args.lambda_sparse_good * good_sparsity
                + args.lambda_sparse_bad * bad_sparsity
                + args.lambda_adv * good_color_loss
                + args.lambda_dom * bad_color_loss
                + args.lambda_style_selection * style_selection_loss
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
            values = {
                "total_loss": total_loss,
                "base_total_loss": base_total_loss,
                "recon_mse": recon_mse,
                "good_recon_mse": good_recon_mse,
                "canonical_good_mse": canonical_good_mse,
                "bad_residual_mse": bad_residual_mse,
                "badcon_loss": badcon_loss,
                "good_pair_loss": good_pair_loss,
                "good_pair_mse": good_pair_mse,
                "good_pair_cosine": good_pair_cosine,
                "good_pair_variance_mean": good_pair_variance_mean,
                "good_color_loss": good_color_loss,
                "bad_color_loss": bad_color_loss,
                "style_selection_loss": style_selection_loss,
                "style_selection_acc": style_selection_acc,
                "good_sparsity": good_sparsity,
                "bad_sparsity": bad_sparsity,
                "good_color_acc": good_color_acc,
                "bad_color_acc": bad_color_acc,
                "style_entropy": style_entropy,
                "style_max_probability": style_max_probability,
                "style_contribution_mse": style_contribution_mse,
                "z_good_mean_abs": good_stats["mean_abs"],
                "z_bad_mean_abs": bad_stats["mean_abs"],
                "z_good_active_frac": good_stats["active_fraction"],
                "z_bad_active_frac": bad_stats["active_fraction"],
                "z_good_active_count": good_stats["active_count"],
                "z_bad_active_count": bad_stats["active_count"],
            }
            total_examples += batch_size
            for name, value in values.items():
                totals[name] += value.item() * batch_size
            style_usage = style_weights.detach().mean(dim=0)
            for style_index in range(model.num_styles):
                totals[f"style_usage_{style_index}"] += (
                    style_usage[style_index].item() * batch_size
                )
            if training and head_update_loss is not None:
                totals["head_update_loss"] += head_update_loss * batch_size

    set_backbone_trainable(model, True)
    set_color_heads_trainable(model, True)
    return {name: value / total_examples for name, value in totals.items()}


def save_checkpoint(
    path,
    model,
    main_optimizer,
    head_optimizer,
    args,
    epoch,
    train_metrics,
    val_metrics,
):
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
                extra={
                    "model_architecture": "factorized_style",
                    "training_procedure": (
                        "scheduled_factorized_canonical_good_reconstruction"
                    ),
                },
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
    for path in [
        train_embedding_path,
        val_embedding_path,
        train_image_path,
        val_image_path,
        args.base_ae_checkpoint,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    device = get_device()
    print(f"Using device: {device}")
    print("Model: factorized style split SAE")
    print(
        "Decoder: canonical content plus a mixture of "
        f"{args.num_styles} content-conditioned affine style transforms"
    )
    print(
        "Schedule: "
        f"phase1={args.phase1_epochs}, "
        f"phase2={args.phase2_epochs}, "
        f"phase3_bad_ramp={args.phase3_bad_ramp_epochs}, "
        f"phase3_pair_ramp={args.phase3_pair_ramp_epochs}"
    )

    base_ae = canonical_good.load_base_ae(args.base_ae_checkpoint, device)
    train_dataset = scheduled.CanonicalGoodPairedEmbeddingDataset(
        train_embedding_path,
        train_image_path,
        base_ae,
        device,
        batch_size=args.canonical_encode_batch_size,
        canonical_target=args.canonical_good_target,
    )
    val_dataset = scheduled.CanonicalGoodPairedEmbeddingDataset(
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
        extra={
            "model_architecture": "factorized_style",
            "training_procedure": "scheduled_factorized_canonical_good_reconstruction",
        },
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
        train_metrics = scheduled.add_schedule_metrics(train_metrics, schedule)
        val_metrics = scheduled.add_schedule_metrics(val_metrics, schedule)
        history.append(
            {
                "epoch": epoch,
                "schedule": schedule,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        style_usage = ", ".join(
            f"{val_metrics[f'style_usage_{index}']:.3f}"
            for index in range(args.num_styles)
        )
        print(
            f"epoch {epoch:03d} | phase={schedule['phase_name']} "
            f"bad_disabled={int(schedule['bad_disabled'])} "
            f"lambda_recon={schedule['lambda_recon_current']:.3g} "
            f"lambda_pair={schedule['lambda_pair_current']:.3g} "
            f"lambda_badcon={schedule['lambda_badcon_current']:.3g} "
            f"lambda_style={schedule['lambda_style_selection_current']:.3g} | "
            f"val_loss={val_metrics['total_loss']:.6f} "
            f"val_recon={val_metrics['recon_mse']:.6f} "
            f"val_canonical={val_metrics['canonical_good_mse']:.6f} "
            f"val_pair={val_metrics['good_pair_loss']:.6f} "
            f"val_bad_color={val_metrics['bad_color_acc']:.4f} | "
            f"val_style_acc={val_metrics['style_selection_acc']:.4f} "
            f"z_good={val_metrics['z_good_active_count']:.1f}/{args.good_latent_dim} "
            f"z_bad={val_metrics['z_bad_active_count']:.1f}/{args.bad_latent_dim} "
            f"style_usage=[{style_usage}] "
            f"style_max={val_metrics['style_max_probability']:.3f}"
        )

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
        if (
            schedule["phase_index"] == 3
            and val_metrics["recon_mse"] < best_phase3_val_recon
        ):
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

    save_json(history_path, history)
    print(f"Saved config to {config_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved history to {history_path}")

    if not args.no_plots:
        plot_dir = args.checkpoint_dir / "plots"
        result = plot_history_metrics(history_path, out_dir=plot_dir)
        print(f"Saved {len(result['saved_paths'])} history plots to {plot_dir}")


if __name__ == "__main__":
    main()
