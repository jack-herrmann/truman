"""PersonalityVAE — Variational Autoencoder over personality embeddings."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class PersonalityVAE(nn.Module):
    def __init__(self, input_dim: int = 1536, latent_dim: int = 32,
                 hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        hidden = hidden_dims or [512, 256]
        encoder_layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            encoder_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
            prev = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)
        decoder_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden):
            decoder_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)])
            prev = h
        decoder_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    @staticmethod
    def loss(recon, x, mu, log_var, kl_weight=0.001):
        recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss

    @torch.no_grad()
    def encode_to_numpy(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.eval()
        x = torch.tensor(embeddings, dtype=torch.float32)
        mu, log_var = self.encode(x)
        sigma = torch.exp(0.5 * log_var)
        return mu.numpy(), sigma.numpy()

    @torch.no_grad()
    def decode_from_numpy(self, z: np.ndarray) -> np.ndarray:
        self.eval()
        return self.decode(torch.tensor(z, dtype=torch.float32)).numpy()

    def fit(self, embeddings: np.ndarray, epochs=200, batch_size=32, lr=1e-3, kl_weight=0.001):
        dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=min(batch_size, len(embeddings)), shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        history = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                recon, mu, log_var = self(batch)
                loss, _, _ = self.loss(recon, batch, mu, log_var, kl_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            history.append(epoch_loss / len(loader))
            if (epoch + 1) % 50 == 0:
                logger.info("VAE epoch %d/%d loss=%.6f", epoch + 1, epochs, history[-1])
        return history

    def save(self, path: str) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "input_dim": self.input_dim,
                     "latent_dim": self.latent_dim}, str(dest))

    @classmethod
    def load(cls, path: str) -> "PersonalityVAE":
        data = torch.load(str(path), map_location="cpu", weights_only=True)
        model = cls(input_dim=data["input_dim"], latent_dim=data["latent_dim"])
        model.load_state_dict(data["state_dict"])
        model.eval()
        return model
