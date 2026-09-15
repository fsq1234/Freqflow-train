"""Radar VAE utilities for the latent FreqFlow MeteoNet experiment.

The data transform intentionally matches ``alphapre_compat.dataset_meteonet``:
raw uint8 radar / 90.0 followed by ``torchvision.transforms.Resize(128)``.
Each dataset item is one frame, even though the source HDF5 stores 25-frame
events.  This keeps VAE memory independent of the event length.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import h5py
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


PIXEL_SCALE = 90.0


@dataclass(frozen=True)
class MeteoVAEConfig:
    image_size: int = 128
    latent_channels: int = 4
    block_out_channels: Tuple[int, ...] = (128, 256, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    act_fn: str = "silu"

    def as_dict(self) -> Dict[str, object]:
        value = asdict(self)
        value["block_out_channels"] = list(self.block_out_channels)
        return value


def _autoencoder_kl_class():
    """Support the two import locations used by recent Diffusers releases."""
    try:
        from diffusers.models.autoencoders import AutoencoderKL
    except ImportError:
        from diffusers import AutoencoderKL
    return AutoencoderKL


def build_autoencoder(config: MeteoVAEConfig) -> nn.Module:
    """Build a radar AutoencoderKL with factor-four spatial compression."""
    AutoencoderKL = _autoencoder_kl_class()
    num_blocks = len(config.block_out_channels)
    return AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=tuple("DownEncoderBlock2D" for _ in range(num_blocks)),
        up_block_types=tuple("UpDecoderBlock2D" for _ in range(num_blocks)),
        block_out_channels=tuple(config.block_out_channels),
        act_fn=config.act_fn,
        latent_channels=config.latent_channels,
        norm_num_groups=config.norm_num_groups,
        layers_per_block=config.layers_per_block,
        sample_size=config.image_size,
    )


class RadarAutoencoder(nn.Module):
    """DDP-safe wrapper returning reconstruction and Gaussian posterior together."""

    def __init__(self, config: MeteoVAEConfig):
        super().__init__()
        self.config = config
        self.vae = build_autoencoder(config)

    @staticmethod
    def _latent_distribution(encoded):
        return encoded.latent_dist if hasattr(encoded, "latent_dist") else encoded

    @staticmethod
    def _decoded_sample(decoded):
        return decoded.sample if hasattr(decoded, "sample") else decoded

    def encode(self, sample: torch.Tensor):
        return self._latent_distribution(self.vae.encode(sample))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self._decoded_sample(self.vae.decode(latent))

    def forward(self, sample: torch.Tensor, sample_posterior: bool = False):
        posterior = self.encode(sample)
        latent = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(latent), posterior

    @property
    def last_layer(self):
        return self.vae.decoder.conv_out.weight


class MeteoFrameDataset(Dataset):
    """Expose every frame of each AlphaPre Meteo event as one VAE sample.

    The underlying HDF5 is opened inside ``__getitem__`` to remain safe with
    PyTorch worker processes.  ``split='val'`` deliberately follows AlphaPre's
    current Meteo contract and reads the ``test`` group.
    """

    def __init__(self, data_path: str | Path, split: str, image_size: int = 128):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported Meteo split: {split}")
        self.data_path = str(data_path)
        self.split = "test" if split == "val" else split
        self.image_size = image_size
        self.transform = transforms.Resize((image_size, image_size))
        with h5py.File(self.data_path, "r") as handle:
            self.num_events = int(handle[f"{self.split}_len"][()])
            if self.num_events < 1:
                raise ValueError(f"Meteo split '{self.split}' is empty")
            self.frames_per_event = int(handle[self.split]["0"].shape[0])
        if self.frames_per_event < 1:
            raise ValueError("Meteo events must contain at least one frame")

    def __len__(self) -> int:
        return self.num_events * self.frames_per_event

    def __getitem__(self, index: int):
        event_index, frame_index = divmod(index, self.frames_per_event)
        with h5py.File(self.data_path, "r") as handle:
            frame = handle[self.split][str(event_index)][frame_index]
        frame = torch.from_numpy(frame).float().div_(PIXEL_SCALE)
        # A single radar frame is [H, W].  Unlike AlphaPre's whole event
        # [T, H, W], it needs an explicit C=1 dimension before Resize.
        frame = self.transform(frame.unsqueeze(0))
        return frame, event_index, frame_index


class MeteoSequenceDataset(Dataset):
    """Return an entire normalized/resized Meteo event for latent precomputation."""

    def __init__(self, data_path: str | Path, split: str, image_size: int = 128):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported Meteo split: {split}")
        self.data_path = str(data_path)
        self.split = "test" if split == "val" else split
        self.transform = transforms.Resize((image_size, image_size))
        with h5py.File(self.data_path, "r") as handle:
            self.num_events = int(handle[f"{self.split}_len"][()])
            self.frames_per_event = int(handle[self.split]["0"].shape[0])

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, event_index: int):
        with h5py.File(self.data_path, "r") as handle:
            frames = handle[self.split][str(event_index)][()]
        frames = torch.from_numpy(frames).float().div_(PIXEL_SCALE)
        # Resize treats the leading time dimension as a batch dimension.
        return self.transform(frames).unsqueeze(1), event_index


def load_vae_checkpoint(path: str | Path, device: torch.device | str):
    """Load the self-contained checkpoints written by ``train_vae_meteo.py``."""
    checkpoint = torch.load(path, map_location=device)
    if "vae_config" not in checkpoint or "vae" not in checkpoint:
        raise ValueError(
            "not a Meteo VAE checkpoint: expected 'vae_config' and 'vae' entries"
        )
    config_values = dict(checkpoint["vae_config"])
    config_values["block_out_channels"] = tuple(config_values["block_out_channels"])
    model = RadarAutoencoder(MeteoVAEConfig(**config_values))
    model.load_state_dict(checkpoint["vae"])
    return model.to(device).eval(), checkpoint
