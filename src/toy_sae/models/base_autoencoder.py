"""Base convolutional autoencoder used as a toy frozen embedding model."""

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """Small convolutional autoencoder for 3x28x28 ColoredMNIST images."""

    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder_linear = nn.Linear(128 * 4 * 4, embedding_dim)

        self.decoder_linear = nn.Linear(embedding_dim, 128 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, images):
        features = self.encoder_conv(images)
        features = torch.flatten(features, start_dim=1)
        return self.encoder_linear(features)

    def decode(self, embeddings):
        features = self.decoder_linear(embeddings)
        features = features.view(-1, 128, 4, 4)
        return self.decoder_conv(features)

    def forward(self, images, return_embeddings=False):
        embeddings = self.encode(images)
        reconstructions = self.decode(embeddings)
        if return_embeddings:
            return reconstructions, embeddings
        return reconstructions
