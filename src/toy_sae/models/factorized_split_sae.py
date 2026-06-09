"""Factorized split SAE with content-conditioned style reconstruction."""

import torch
from torch import nn

from toy_sae.models.split_sae import GradientReverseFunction


class FactorizedSplitSparseAutoencoder(nn.Module):
    """Split SAE where z_bad selects a transform of z_good-derived content.

    Unlike the additive split SAE, the bad branch never emits an independent
    input-dimensional reconstruction. It selects a mixture of learned affine
    residual transforms that are applied to the canonical content embedding.
    """

    def __init__(
        self,
        input_dim=64,
        hidden_dim=128,
        good_latent_dim=128,
        bad_latent_dim=4,
        num_colors=2,
        num_styles=2,
        style_hidden_dim=16,
        style_temperature=1.0,
        style_init_scale=1e-3,
    ):
        super().__init__()
        if num_styles < 2:
            raise ValueError("num_styles must be at least 2")
        if style_temperature <= 0.0:
            raise ValueError("style_temperature must be positive")
        if style_init_scale < 0.0:
            raise ValueError("style_init_scale must be non-negative")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.good_latent_dim = good_latent_dim
        self.bad_latent_dim = bad_latent_dim
        self.num_colors = num_colors
        self.num_styles = num_styles
        self.style_hidden_dim = style_hidden_dim
        self.style_temperature = float(style_temperature)
        self.style_init_scale = float(style_init_scale)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.good_encoder = nn.Sequential(
            nn.Linear(hidden_dim, good_latent_dim),
            nn.ReLU(),
        )
        self.bad_encoder = nn.Sequential(
            nn.Linear(hidden_dim, bad_latent_dim),
            nn.ReLU(),
        )

        self.good_decoder = nn.Sequential(
            nn.Linear(good_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.style_selector = nn.Sequential(
            nn.Linear(bad_latent_dim, style_hidden_dim),
            nn.ReLU(),
            nn.Linear(style_hidden_dim, num_styles),
        )
        self.style_residual_matrices = nn.Parameter(
            torch.empty(num_styles, input_dim, input_dim)
        )
        self.style_residual_biases = nn.Parameter(torch.empty(num_styles, input_dim))

        self.good_color_head = nn.Sequential(
            nn.Linear(good_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_colors),
        )
        self.bad_color_head = nn.Sequential(
            nn.Linear(bad_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_colors),
        )

        self.reset_style_parameters()

    def reset_style_parameters(self):
        """Initialize every style near identity with small symmetry breaking."""
        nn.init.normal_(
            self.style_residual_matrices,
            mean=0.0,
            std=self.style_init_scale,
        )
        nn.init.normal_(
            self.style_residual_biases,
            mean=0.0,
            std=self.style_init_scale,
        )

    def encode(self, embeddings):
        hidden = self.encoder(embeddings)
        z_good = self.good_encoder(hidden)
        z_bad = self.bad_encoder(hidden)
        return z_good, z_bad

    def decode_good(self, z_good):
        return self.good_decoder(z_good)

    def decode_style(self, z_bad):
        style_logits = self.style_selector(z_bad)
        style_weights = torch.softmax(
            style_logits / self.style_temperature,
            dim=1,
        )
        return style_logits, style_weights

    def apply_style(self, content, style_weights):
        """Apply a content-conditioned mixture of affine residual transforms."""
        style_residuals = torch.einsum(
            "bi,koi->bko",
            content,
            self.style_residual_matrices,
        )
        style_residuals = style_residuals + self.style_residual_biases.unsqueeze(0)
        mixed_residual = torch.einsum("bk,bki->bi", style_weights, style_residuals)
        return content + mixed_residual

    def decode(
        self,
        z_good,
        z_bad,
        *,
        good_grad_scale=1.0,
        style_enabled=True,
    ):
        if not 0.0 <= good_grad_scale <= 1.0:
            raise ValueError("good_grad_scale must be between 0.0 and 1.0")

        good_reconstruction = self.decode_good(z_good)
        style_logits, style_weights = self.decode_style(z_bad)

        if style_enabled:
            good_for_full_reconstruction = (
                good_grad_scale * good_reconstruction
                + (1.0 - good_grad_scale) * good_reconstruction.detach()
            )
            reconstruction = self.apply_style(
                good_for_full_reconstruction,
                style_weights,
            )
            detached_content = good_reconstruction.detach()
            style_contribution_for_loss = (
                self.apply_style(detached_content, style_weights)
                - detached_content
            )
        else:
            reconstruction = good_reconstruction
            style_contribution_for_loss = torch.zeros_like(good_reconstruction)

        style_contribution = reconstruction - good_reconstruction
        return {
            "reconstruction": reconstruction,
            "good_reconstruction": good_reconstruction,
            "style_contribution": style_contribution,
            "style_contribution_for_loss": style_contribution_for_loss,
            "bad_reconstruction": style_contribution_for_loss,
            "style_logits": style_logits,
            "style_weights": style_weights,
        }

    def classify_good_color(self, z_good, grl_lambda=1.0):
        reversed_z_good = GradientReverseFunction.apply(z_good, grl_lambda)
        return self.good_color_head(reversed_z_good)

    def classify_good_color_no_grl(self, z_good):
        return self.good_color_head(z_good)

    def classify_bad_color(self, z_bad):
        return self.bad_color_head(z_bad)

    def forward(
        self,
        embeddings,
        grl_lambda=1.0,
        *,
        good_grad_scale=1.0,
        style_enabled=True,
    ):
        z_good, z_bad = self.encode(embeddings)
        outputs = self.decode(
            z_good,
            z_bad,
            good_grad_scale=good_grad_scale,
            style_enabled=style_enabled,
        )
        outputs.update(
            {
                "z_good": z_good,
                "z_bad": z_bad,
                "good_color_logits": self.classify_good_color(z_good, grl_lambda),
                "bad_color_logits": self.classify_bad_color(z_bad),
            }
        )
        return outputs
