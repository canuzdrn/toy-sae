"""Model-loading helpers for split-SAE checkpoints."""

from __future__ import annotations

from toy_sae.models.factorized_split_sae import FactorizedSplitSparseAutoencoder
from toy_sae.models.separate_split_sae import SeparateEncoderSAE
from toy_sae.models.simple_split_sae import SimpleSplitSparseAutoencoder
from toy_sae.models.split_sae import SplitSparseAutoencoder


def infer_model_family(checkpoint: dict) -> str:
    """Infer whether a checkpoint belongs to the shared or separate model family."""
    args = checkpoint.get("args", {})
    if args.get("model_architecture") == "factorized_style":
        return "factorized_style"
    if args.get("model_architecture") == "simple_shared":
        return "simple_shared"
    if args.get("model_architecture") == "separate_encoder":
        return "separate"
    if args.get("model_architecture") == "shared_encoder":
        return "shared"

    state_dict = checkpoint.get("model_state_dict", {})
    if any(key.startswith("encoder.") for key in state_dict):
        if "style_residual_matrices" in state_dict:
            return "factorized_style"
        if "good_decoder.weight" in state_dict or "bad_decoder.weight" in state_dict:
            return "simple_shared"
        return "shared"
    return "separate"


def load_split_sae_model(checkpoint: dict, device, model_family: str = "auto"):
    """Instantiate a split-SAE model from checkpoint args and load weights."""
    if model_family == "auto":
        model_family = infer_model_family(checkpoint)

    args = checkpoint["args"]
    if model_family == "separate":
        model = SeparateEncoderSAE(
            input_dim=args["input_dim"],
            hidden_dim=args["hidden_dim"],
            good_latent_dim=args["good_latent_dim"],
            bad_latent_dim=args["bad_latent_dim"],
        )
    elif model_family == "simple_shared":
        model = SimpleSplitSparseAutoencoder(
            input_dim=args["input_dim"],
            hidden_dim=args["hidden_dim"],
            good_latent_dim=args["good_latent_dim"],
            bad_latent_dim=args["bad_latent_dim"],
        )
    elif model_family == "factorized_style":
        model = FactorizedSplitSparseAutoencoder(
            input_dim=args["input_dim"],
            hidden_dim=args["hidden_dim"],
            good_latent_dim=args["good_latent_dim"],
            bad_latent_dim=args["bad_latent_dim"],
            num_styles=args.get("num_styles", 2),
            style_hidden_dim=args.get("style_hidden_dim", 16),
            style_temperature=args.get("style_temperature", 1.0),
            style_init_scale=args.get("style_init_scale", 1e-3),
        )
    elif model_family == "shared":
        model = SplitSparseAutoencoder(
            input_dim=args["input_dim"],
            hidden_dim=args["hidden_dim"],
            good_latent_dim=args["good_latent_dim"],
            bad_latent_dim=args["bad_latent_dim"],
        )
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
