"""Model-loading helpers for split-SAE checkpoints."""

from __future__ import annotations

from toy_sae.models.separate_split_sae import SeparateEncoderSAE
from toy_sae.models.split_sae import SplitSparseAutoencoder


def infer_model_family(checkpoint: dict) -> str:
    """Infer whether a checkpoint belongs to the shared or separate model family."""
    state_dict = checkpoint.get("model_state_dict", {})
    if any(key.startswith("encoder.") for key in state_dict):
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
