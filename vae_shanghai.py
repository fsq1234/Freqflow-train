"""Standalone factor-four Shanghai radar VAE and dataset interfaces.

The architecture follows the original Meteo VAE; Shanghai training, checkpoints,
data normalization, and scripts are independent of all Meteo entrypoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

PIXEL_SCALE = 90.0


def shanghai_data_config(image_size=128):
    return {
        "dataset_name": "shanghai",
        "normalization_divisor": 255.0,
        "normalization": f"raw_uint8 / 255.0 then torchvision.Resize({image_size},{image_size})",
        "pixel_scale": PIXEL_SCALE,
        "thresholds": [20, 30, 35, 40],
        "image_size": image_size,
        "val_split": "test",
        "frame_sampling": "one_random_frame_per_event",
    }


def validate_vae_data_config(checkpoint, expected):
    saved = checkpoint.get("data_config")
    if saved is None:
        raise ValueError("Shanghai requires a Shanghai VAE checkpoint with data_config")
    for key in ("dataset_name", "normalization_divisor", "pixel_scale", "thresholds", "image_size"):
        if saved.get(key) != expected[key]:
            raise ValueError(f"VAE checkpoint {key}={saved.get(key)}, requested {expected[key]}")


@dataclass(frozen=True)
class ShanghaiVAEConfig:
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


def build_autoencoder(config: ShanghaiVAEConfig) -> nn.Module:
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

    def __init__(self, config: ShanghaiVAEConfig):
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


class ShanghaiSequenceDataset(Dataset):
    """Read complete events for optional frozen-VAE latent precomputation."""

    def __init__(self, data_path: str | Path, split="train", image_size=128):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported Shanghai split: {split}")
        self.data_path = str(data_path)
        self.split = "test" if split == "val" else split
        self.image_size = image_size
        self.transform = transforms.Resize((image_size, image_size))
        with h5py.File(self.data_path, "r") as handle:
            group = handle[self.split]
            self.num_events = int(group["all_len"][()])
            if self.num_events < 1:
                raise ValueError(f"Shanghai split '{self.split}' is empty")
            self.frames_per_event = int(group["0"].shape[0])
            if self.frames_per_event < 1:
                raise ValueError("Shanghai events must contain at least one frame")

    def __len__(self):
        return self.num_events

    def __getitem__(self, event_index):
        with h5py.File(self.data_path, "r") as handle:
            frames = handle[self.split][str(event_index)][()]
        frames = torch.from_numpy(frames).float().div_(255.0)
        if frames.ndim == 4 and frames.shape[1] == 1:
            frames = frames[:, 0]
        if frames.ndim != 3:
            raise ValueError("Shanghai events must have shape (T,H,W) or (T,1,H,W)")
        return self.transform(frames).unsqueeze(1), event_index


class ShanghaiFrameDataset(ShanghaiSequenceDataset):
    """Read/resize only one random frame, rather than all 25 event frames."""

    def __getitem__(self, event_index):
        with h5py.File(self.data_path, "r") as handle:
            event = handle[self.split][str(event_index)]
            frame_index = int(np.random.randint(0, event.shape[0]))
            frame = event[frame_index]
        frame = torch.from_numpy(frame).float().div_(255.0)
        if frame.ndim == 2:
            frame = frame.unsqueeze(0)
        if frame.ndim != 3 or frame.shape[0] != 1:
            raise ValueError("Shanghai frames must contain exactly one channel")
        return self.transform(frame), event_index, frame_index


def load_vae_checkpoint(path: str | Path, device: torch.device | str):
    """Load the self-contained checkpoints written by ``train_vae_shanghai.py``."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if "vae_config" not in checkpoint or "vae" not in checkpoint:
        raise ValueError(
            "not a Shanghai VAE checkpoint: expected 'vae_config' and 'vae' entries"
        )
    config_values = dict(checkpoint["vae_config"])
    config_values["block_out_channels"] = tuple(config_values["block_out_channels"])
    model = RadarAutoencoder(ShanghaiVAEConfig(**config_values))
    model.load_state_dict(checkpoint["vae"])
    return model.to(device).eval(), checkpoint
