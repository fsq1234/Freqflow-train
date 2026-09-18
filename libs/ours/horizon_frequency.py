"""Lead-time and history-conditioned routing for latent frequency bands."""

import torch
from torch import nn


class HorizonFrequencyRouter(nn.Module):
    """Predict one low-frequency gate per sample, lead and latent channel.

    The final projection is zero-initialized, so every gate starts at 0.5.
    With the factor-of-two normalization in ``forward``, enabling this module
    is therefore an identity transformation at initialization.
    """

    def __init__(
        self,
        frames_out: int,
        latent_channels: int,
        history_dim: int,
        hidden_dim: int = 128,
        mode: str = "time_lead_history",
    ):
        super().__init__()
        if frames_out <= 0 or latent_channels <= 0:
            raise ValueError("frames_out and latent_channels must be positive")

        self.frames_out = frames_out
        self.latent_channels = latent_channels
        self.channels = frames_out * latent_channels
        valid_modes = {
            "time",
            "time_lead",
            "time_lead_history",
            "time_lead_history_energy",
        }
        if mode not in valid_modes:
            raise ValueError(f"unsupported horizon frequency router mode: {mode}")
        self.mode = mode

        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lead_embedding = nn.Embedding(frames_out, hidden_dim)
        self.history_encoder = nn.Sequential(
            nn.Linear(history_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.energy_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # sigmoid(0) = 0.5. Combined with the 2x normalization below, the
        # initial router exactly preserves both frequency-band inputs.
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.zeros_(self.gate_head[-1].bias)

    def forward(self, low, high, history_features, flow_time):
        if low.shape != high.shape:
            raise ValueError("low- and high-frequency tensors must have identical shapes")
        if low.ndim != 4 or low.shape[1] != self.channels:
            raise ValueError(
                f"expected frequency tensors [B,{self.channels},H,W], got {tuple(low.shape)}"
            )
        if history_features.ndim != 4 or history_features.shape[0] != low.shape[0]:
            raise ValueError("history features must be [B,C,H,W] with matching batch size")
        if flow_time is None:
            raise ValueError("flow_time is required when the horizon frequency router is enabled")

        batch_size, _, height, width = low.shape
        low_by_lead = low.reshape(
            batch_size, self.frames_out, self.latent_channels, height, width
        )
        high_by_lead = high.reshape_as(low_by_lead)

        # The router observes per-band RMS energy without using future truth.
        low_energy = low_by_lead.float().square().mean(dim=(-2, -1)).sqrt()
        high_energy = high_by_lead.float().square().mean(dim=(-2, -1)).sqrt()
        energy = torch.stack((torch.log1p(low_energy), torch.log1p(high_energy)), dim=-1)

        history_vector = history_features.float().mean(dim=(-2, -1))
        time_vector = flow_time.float().reshape(batch_size, 1)
        lead_indices = torch.arange(self.frames_out, device=low.device)

        context = self.time_encoder(time_vector)[:, None, None, :]
        if self.mode != "time":
            context = context + self.lead_embedding(lead_indices)[None, :, None, :]
        if self.mode in {"time_lead_history", "time_lead_history_energy"}:
            context = context + self.history_encoder(history_vector)[:, None, None, :]
        if self.mode == "time_lead_history_energy":
            context = context + self.energy_encoder(energy)

        low_weight = torch.sigmoid(self.gate_head(context))
        low_weight = low_weight.expand(
            batch_size, self.frames_out, self.latent_channels, 1
        ).unsqueeze(-1)
        low_weight = low_weight.to(dtype=low.dtype)
        high_weight = 1.0 - low_weight

        routed_low = (2.0 * low_weight * low_by_lead).reshape_as(low)
        routed_high = (2.0 * high_weight * high_by_lead).reshape_as(high)
        return routed_low, routed_high, low_weight
