"""Lead-time and history-conditioned routing for latent frequency bands."""

import torch
from torch import nn


class HorizonFrequencyRouter(nn.Module):
    """Condition frequency routing on flow time, horizon and precipitation state.

    Legacy modes use one complementary low-frequency gate.  The
    ``state_adaptive_residual`` predicts independent bounded residual gains.
    ``state_adaptive_contrast`` removes their common-scale degree of freedom:
    it predicts one bounded contrast and enforces ``low_gain + high_gain = 2``.
    Every parameterization is an exact identity transformation initially.
    """

    def __init__(
        self,
        frames_out: int,
        latent_channels: int,
        history_dim: int,
        frames_in: int = 5,
        hidden_dim: int = 128,
        mode: str = "time_lead_history",
    ):
        super().__init__()
        if frames_out <= 0 or latent_channels <= 0:
            raise ValueError("frames_out and latent_channels must be positive")

        self.frames_out = frames_out
        self.frames_in = frames_in
        self.latent_channels = latent_channels
        self.channels = frames_out * latent_channels
        valid_modes = {
            "time",
            "time_lead",
            "time_lead_history",
            "time_lead_history_energy",
            "state_adaptive_residual",
            "state_adaptive_contrast",
        }
        if mode not in valid_modes:
            raise ValueError(f"unsupported horizon frequency router mode: {mode}")
        self.mode = mode
        self.state_adaptive = mode in {
            "state_adaptive_residual", "state_adaptive_contrast"
        }

        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if self.state_adaptive:
            self.lead_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            # Retain the original discrete encoding for reproducibility of the
            # first router experiment.
            self.lead_embedding = nn.Embedding(frames_out, hidden_dim)
        self.history_encoder = nn.Sequential(
            nn.Linear(history_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if self.state_adaptive:
            self.channel_embedding = nn.Embedding(latent_channels, hidden_dim)
            self.state_energy_encoder = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # Per latent channel: temporal mean, latest value and first-to-last
            # trend for each of the low/high historical bands.
            self.history_spectrum_encoder = nn.Sequential(
                nn.Linear(6, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gain_head = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    hidden_dim,
                    1 if self.mode == "state_adaptive_contrast" else 2,
                ),
            )
            nn.init.zeros_(self.gain_head[-1].weight)
            nn.init.zeros_(self.gain_head[-1].bias)
        else:
            self.energy_encoder = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gate_head = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            # sigmoid(0) = 0.5. Combined with the 2x normalization below,
            # legacy routing exactly preserves both bands initially.
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.zeros_(self.gate_head[-1].bias)

    @staticmethod
    def _rms(value):
        return value.float().square().mean(dim=(-2, -1)).sqrt()

    def forward(
        self, low, high, history_features, flow_time,
        history_low=None, history_high=None,
    ):
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

        # Keep the small routing network in FP32 under BF16/FP16 autocast. Near
        # 0.5, BF16 gates otherwise move in visible 0.00390625-wide steps.
        with torch.autocast(device_type=low.device.type, enabled=False):
            # The router observes per-band RMS energy without using future truth.
            low_energy = self._rms(low_by_lead)
            high_energy = self._rms(high_by_lead)
            energy = torch.stack(
                (torch.log1p(low_energy), torch.log1p(high_energy)), dim=-1
            )

            history_vector = history_features.float().mean(dim=(-2, -1))
            time_vector = flow_time.float().reshape(batch_size, 1)
            lead_indices = torch.arange(self.frames_out, device=low.device)

            context = self.time_encoder(time_vector)[:, None, None, :]
            if self.mode != "time":
                if self.state_adaptive:
                    normalized_lead = (
                        (lead_indices.float() + 1.0) / self.frames_out
                    ).unsqueeze(-1)
                    lead_context = self.lead_encoder(normalized_lead)
                else:
                    lead_context = self.lead_embedding(lead_indices)
                context = context + lead_context[None, :, None, :]
            if self.mode in {
                "time_lead_history", "time_lead_history_energy",
                "state_adaptive_residual", "state_adaptive_contrast",
            }:
                context = context + self.history_encoder(history_vector)[:, None, None, :]

            if self.state_adaptive:
                if history_low is None or history_high is None:
                    raise ValueError(
                        "state-adaptive routing requires low/high historical bands"
                    )
                expected_history_channels = self.frames_in * self.latent_channels
                if history_low.shape != history_high.shape or history_low.shape[1] != expected_history_channels:
                    raise ValueError(
                        "expected historical bands "
                        f"[B,{expected_history_channels},H,W], got {tuple(history_low.shape)}"
                    )
                history_low = history_low.reshape(
                    batch_size, self.frames_in, self.latent_channels,
                    history_low.shape[-2], history_low.shape[-1],
                )
                history_high = history_high.reshape_as(history_low)
                historical_energy = torch.stack(
                    (self._rms(history_low), self._rms(history_high)), dim=-1
                )
                historical_energy = torch.log1p(historical_energy)
                historical_descriptor = torch.cat(
                    (
                        historical_energy.mean(dim=1),
                        historical_energy[:, -1],
                        historical_energy[:, -1] - historical_energy[:, 0],
                    ),
                    dim=-1,
                )
                channel_indices = torch.arange(self.latent_channels, device=low.device)
                context = context + self.channel_embedding(channel_indices)[None, None, :, :]
                context = context + self.state_energy_encoder(energy)
                context = context + self.history_spectrum_encoder(
                    historical_descriptor
                )[:, None, :, :]

                if self.mode == "state_adaptive_contrast":
                    # A single state-dependent contrast prevents the common
                    # attenuation shortcut observed in the independent-gain
                    # experiment. Gains remain in [0.75, 1.25] and sum to 2.
                    contrast = 0.25 * torch.tanh(self.gain_head(context)[..., 0])
                    low_gain_fp32 = (1.0 + contrast).unsqueeze(-1).unsqueeze(-1)
                    high_gain_fp32 = (1.0 - contrast).unsqueeze(-1).unsqueeze(-1)
                else:
                    # Independent residual multipliers in [0.5, 1.5].
                    residual = 0.5 * torch.tanh(self.gain_head(context))
                    low_gain_fp32 = (
                        1.0 + residual[..., 0]
                    ).unsqueeze(-1).unsqueeze(-1)
                    high_gain_fp32 = (
                        1.0 + residual[..., 1]
                    ).unsqueeze(-1).unsqueeze(-1)
            else:
                if self.mode == "time_lead_history_energy":
                    context = context + self.energy_encoder(energy)
                low_weight_fp32 = torch.sigmoid(self.gate_head(context))
                low_weight_fp32 = low_weight_fp32.expand(
                    batch_size, self.frames_out, self.latent_channels, 1
                ).unsqueeze(-1)
                low_gain_fp32 = 2.0 * low_weight_fp32
                high_gain_fp32 = 2.0 * (1.0 - low_weight_fp32)

        low_gain = low_gain_fp32.to(dtype=low.dtype)
        high_gain = high_gain_fp32.to(dtype=high.dtype)
        routed_low = (low_gain * low_by_lead).reshape_as(low)
        routed_high = (high_gain * high_by_lead).reshape_as(high)
        return routed_low, routed_high, low_gain_fp32, high_gain_fp32
