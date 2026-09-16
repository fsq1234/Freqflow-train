"""Meteo latent-HDF5 reader for VAE-space FreqFlow training."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms


PIXEL_SCALE = 90.0
THRESHOLDS = [12, 18, 24, 32]


class MeteoLatent(Dataset):
    """Read one complete latent event and optional raw-pixel targets.

    HDF5 handles are opened inside ``__getitem__`` so DataLoader workers never
    share an h5py handle.  Validation follows the existing AlphaPre Meteo
    contract and uses the source ``test`` split.
    """

    def __init__(
        self,
        latent_path: str | Path,
        split: str,
        raw_data_path: str | Path | None = None,
        image_size: int = 128,
        frames_in: int = 5,
        frames_out: int = 20,
        return_pixels: bool = False,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported Meteo split: {split}")
        self.latent_path = str(latent_path)
        self.split = "test" if split == "val" else split
        self.raw_data_path = None if raw_data_path is None else str(raw_data_path)
        self.image_size = image_size
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.total_frames = frames_in + frames_out
        self.return_pixels = return_pixels
        self.resize = transforms.Resize((image_size, image_size))

        with h5py.File(self.latent_path, "r") as handle:
            schema = handle.attrs.get("schema", "")
            layout = handle.attrs.get("layout", "")
            if schema != "freqflow_meteo_vae_latent_v1" or layout != "TCHW":
                raise ValueError(
                    f"unsupported latent HDF5 contract: schema={schema!r}, layout={layout!r}"
                )
            if self.split not in handle:
                raise KeyError(f"latent HDF5 has no '{self.split}' group")
            group = handle[self.split]
            self.num_events = int(group.attrs.get("num_events", len(group)))
            self.frames_per_event = int(group.attrs.get("frames_per_event", 0))
            if self.num_events < 1:
                raise ValueError(f"latent split '{self.split}' is empty")
            sample_shape = tuple(group["0"].shape)
            if len(sample_shape) != 4:
                raise ValueError(f"expected latent event [T,C,H,W], got {sample_shape}")
            if not self.frames_per_event:
                self.frames_per_event = sample_shape[0]
            self.latent_channels = sample_shape[1]
            self.latent_height = sample_shape[2]
            self.latent_width = sample_shape[3]
            self.latent_mean = float(handle.attrs["train_latent_mean"])
            self.latent_std = float(handle.attrs["train_latent_std"])
            vae_config = json.loads(handle.attrs["vae_config_json"])
            self.vae_image_size = int(vae_config["image_size"])
            self.vae_latent_channels = int(vae_config["latent_channels"])

        if self.frames_per_event < self.total_frames:
            raise ValueError(
                f"need {self.total_frames} frames, latent events contain {self.frames_per_event}"
            )
        if self.latent_std <= 0:
            raise ValueError(f"invalid train latent std: {self.latent_std}")
        if self.return_pixels:
            if self.raw_data_path is None:
                raise ValueError("raw_data_path is required when return_pixels=True")
            with h5py.File(self.raw_data_path, "r") as handle:
                raw_count = int(handle[f"{self.split}_len"][()])
            if raw_count != self.num_events:
                raise ValueError(
                    f"latent/raw event count mismatch for {self.split}: "
                    f"{self.num_events} != {raw_count}"
                )

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, event_index: int):
        with h5py.File(self.latent_path, "r") as handle:
            latent = handle[self.split][str(event_index)][: self.total_frames]
        latent = torch.from_numpy(latent).float()
        if not self.return_pixels:
            return latent

        with h5py.File(self.raw_data_path, "r") as handle:
            future = handle[self.split][str(event_index)][
                self.frames_in : self.total_frames
            ]
        future = torch.from_numpy(future).float().div_(PIXEL_SCALE)
        future = self.resize(future).unsqueeze(1)
        return latent, future
