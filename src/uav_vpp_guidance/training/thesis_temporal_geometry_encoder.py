"""Self-supervised, past-only temporal geometry encoder for five-state v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TemporalEncoderBatch:
    """A normalized past window and future geometry deltas for two horizons."""

    history: torch.Tensor
    future_deltas: Dict[int, torch.Tensor]


class TemporalGeometryEncoder(nn.Module):
    """Encode ten observed geometry frames without retaining recurrent episode state."""

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        horizons: Sequence[int] = (1, 5),
    ):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or embedding_dim < 1:
            raise ValueError("Temporal encoder dimensions must be positive")
        if not horizons or any(int(horizon) < 1 for horizon in horizons):
            raise ValueError("Temporal encoder requires positive prediction horizons")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.horizons = tuple(sorted({int(horizon) for horizon in horizons}))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.input_dim))
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True)
        self.embedding_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.Tanh(),
        )
        self.reconstruction_head = nn.Linear(self.hidden_dim, self.input_dim)
        self.future_heads = nn.ModuleDict(
            {
                str(horizon): nn.Linear(self.embedding_dim, self.input_dim)
                for horizon in self.horizons
            }
        )

    def forward(self, history: torch.Tensor, mask: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """Return a fixed embedding, masked reconstruction, and horizon predictions."""

        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError(
                "history must have shape [batch, steps, "
                f"{self.input_dim}], got {tuple(history.shape)}"
            )
        if not torch.isfinite(history).all():
            raise ValueError("history contains non-finite values")
        if mask is None:
            mask = torch.zeros_like(history, dtype=torch.bool)
        if mask.shape != history.shape or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean and have the same shape as history")
        masked_history = torch.where(mask, self.mask_token.expand_as(history), history)
        recurrent_output, last_hidden = self.gru(masked_history)
        embedding = self.embedding_projection(last_hidden[-1])
        return {
            "embedding": embedding,
            "reconstruction": self.reconstruction_head(recurrent_output),
            "future_deltas": {
                int(horizon): self.future_heads[str(horizon)](embedding)
                for horizon in self.horizons
            },
        }

    @torch.no_grad()
    def encode_history(self, history: np.ndarray | torch.Tensor) -> np.ndarray:
        """Encode one or more standalone past windows with no cross-episode state."""

        values = torch.as_tensor(history, dtype=torch.float32)
        if values.ndim == 2:
            values = values.unsqueeze(0)
        output = self.forward(values)
        return output["embedding"].cpu().numpy()


def deterministic_mask(history: torch.Tensor, ratio: float, generator: torch.Generator) -> torch.Tensor:
    """Create a reproducible mask while guaranteeing at least one masked value."""

    if not 0.0 < float(ratio) < 1.0:
        raise ValueError("mask ratio must lie strictly between zero and one")
    mask = torch.rand(history.shape, generator=generator, device=history.device) < float(ratio)
    if not bool(mask.any()):
        mask.reshape(-1)[0] = True
    return mask


def self_supervised_loss(
    output: Dict[str, torch.Tensor],
    batch: TemporalEncoderBatch,
    mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute masked reconstruction plus equally weighted future-delta losses."""

    if not bool(mask.any()):
        raise ValueError("masked reconstruction requires at least one masked element")
    reconstruction = output["reconstruction"]
    reconstruction_loss = torch.mean((reconstruction[mask] - batch.history[mask]) ** 2)
    prediction_losses: Dict[int, torch.Tensor] = {}
    for horizon, target in batch.future_deltas.items():
        prediction = output["future_deltas"].get(int(horizon))
        if prediction is None:
            raise KeyError(f"Model is missing horizon {horizon}")
        prediction_losses[int(horizon)] = torch.mean((prediction - target) ** 2)
    prediction_loss = torch.stack(list(prediction_losses.values())).mean()
    total = reconstruction_loss + prediction_loss
    return {
        "total": total,
        "reconstruction": reconstruction_loss,
        "prediction": prediction_loss,
        **{f"horizon_{horizon}": value for horizon, value in prediction_losses.items()},
    }
