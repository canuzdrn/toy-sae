"""Simplified separate-encoder split SAE for frozen base-AE embeddings."""

import torch
from torch import nn


class GradientReverseFunction(torch.autograd.Function):
    """Identity forward pass with a negative scaled gradient in the backward pass."""

    @staticmethod
    def forward(ctx, inputs, lambda_value):
        ctx.lambda_value = float(lambda_value)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_value * grad_output, None


class SimpleSeparateEncoderSAE(nn.Module):
    """Separate-encoder split SAE with single-layer encoders, decoders, and color heads.

    This model keeps the same latent dimensions and public interface as
    ``SeparateEncoderSAE`` but removes hidden MLP layers where possible. The goal is
    interpretability: each latent is produced by one input-space linear direction,
    and each reconstruction branch is one linear map back to embedding space.
    """

    def __init__(
        self,
        input_dim=64,
        hidden_dim=128,
        good_latent_dim=128,
        bad_latent_dim=32,
        num_colors=2,
    ):
        super().__init__()
        self.input_dim = input_dim
        # Kept for checkpoint/training-script compatibility with the MLP model.
        self.hidden_dim = hidden_dim
        self.good_latent_dim = good_latent_dim
        self.bad_latent_dim = bad_latent_dim
        self.num_colors = num_colors

        self.good_encoder = nn.Sequential(
            nn.Linear(input_dim, good_latent_dim),
            nn.ReLU(),
        )
        self.bad_encoder = nn.Sequential(
            nn.Linear(input_dim, bad_latent_dim),
            nn.ReLU(),
        )

        self.good_decoder = nn.Linear(good_latent_dim, input_dim)
        self.bad_decoder = nn.Linear(bad_latent_dim, input_dim)

        self.good_color_head = nn.Linear(good_latent_dim, num_colors)
        self.bad_color_head = nn.Linear(bad_latent_dim, num_colors)

    def encode(self, embeddings):
        z_good = self.good_encoder(embeddings)
        z_bad = self.bad_encoder(embeddings)
        return z_good, z_bad

    def decode_good(self, z_good):
        return self.good_decoder(z_good)

    def decode_bad(self, z_bad):
        return self.bad_decoder(z_bad)

    def decode(self, z_good, z_bad):
        good_reconstruction = self.decode_good(z_good)
        bad_reconstruction = self.decode_bad(z_bad)
        reconstruction = good_reconstruction + bad_reconstruction
        return reconstruction, good_reconstruction, bad_reconstruction

    def classify_good_color(self, z_good, grl_lambda=1.0):
        reversed_z_good = GradientReverseFunction.apply(z_good, grl_lambda)
        return self.good_color_head(reversed_z_good)

    def classify_good_color_no_grl(self, z_good):
        return self.good_color_head(z_good)

    def classify_bad_color(self, z_bad):
        return self.bad_color_head(z_bad)

    def forward(self, embeddings, grl_lambda=1.0):
        z_good, z_bad = self.encode(embeddings)
        reconstruction, good_reconstruction, bad_reconstruction = self.decode(z_good, z_bad)
        good_color_logits = self.classify_good_color(z_good, grl_lambda)
        bad_color_logits = self.classify_bad_color(z_bad)

        return {
            "reconstruction": reconstruction,
            "good_reconstruction": good_reconstruction,
            "bad_reconstruction": bad_reconstruction,
            "z_good": z_good,
            "z_bad": z_bad,
            "good_color_logits": good_color_logits,
            "bad_color_logits": bad_color_logits,
        }
