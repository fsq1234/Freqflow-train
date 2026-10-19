"""Pre-encode Shanghai events with a frozen radar VAE into a latent HDF5 file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vae_shanghai import (
    ShanghaiSequenceDataset, load_vae_checkpoint, shanghai_data_config, validate_vae_data_config,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", choices=("shanghai",), default="shanghai")
    parser.add_argument("--data-path", required=True, help="original radar HDF5")
    parser.add_argument("--vae-checkpoint", required=True, help="best_val_loss.pt")
    parser.add_argument("--output-path", required=True, help="new latent HDF5 path")
    parser.add_argument("--splits", default="train,test", help="comma-separated original HDF5 groups")
    parser.add_argument("--batch-size", type=int, default=2, help="events per loader batch")
    parser.add_argument("--encode-batch-size", type=int, default=32, help="frames per VAE encode call")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


@torch.no_grad()
def encode_split(output, model, dataset, split, batch_size, encode_batch_size, num_workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    group = output.create_group(split)
    group.attrs["num_events"] = dataset.num_events
    group.attrs["frames_per_event"] = dataset.frames_per_event
    latent_sum = 0.0
    latent_square_sum = 0.0
    latent_count = 0
    for sequences, indices in tqdm(loader, desc=f"Encoding {split}", ncols=120):
        # [B,T,1,128,128] -> [B*T,1,128,128], then decode the batch back to events.
        batch, frames, channels, height, width = sequences.shape
        flat = sequences.reshape(batch * frames, channels, height, width).to(device, non_blocking=True)
        encoded = []
        for start in range(0, flat.shape[0], encode_batch_size):
            posterior = model.encode(flat[start : start + encode_batch_size])
            encoded.append(posterior.mode())
        latent = torch.cat(encoded, dim=0).float().cpu().reshape(batch, frames, -1, model.config.image_size // 4, model.config.image_size // 4)
        for item, event_index in zip(latent.numpy().astype(np.float32), indices.tolist()):
            group.create_dataset(str(event_index), data=item, compression="gzip", compression_opts=4)
        if split == "train":
            latent_sum += float(latent.double().sum().item())
            latent_square_sum += float(latent.double().square().sum().item())
            latent_count += int(latent.numel())
    return latent_sum, latent_square_sum, latent_count


def main(argv=None):
    args = parse_args(argv)
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_path}; pass --overwrite to replace it")
    if args.batch_size < 1 or args.encode_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    requested_splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    if not requested_splits or any(value not in {"train", "test"} for value in requested_splits):
        raise ValueError("--splits must contain only train and/or test")
    device = torch.device(args.device)
    model, checkpoint = load_vae_checkpoint(args.vae_checkpoint, device)
    data_config = shanghai_data_config(model.config.image_size)
    validate_vae_data_config(checkpoint, data_config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary_path.exists():
        raise FileExistsError(f"temporary output exists: remove or rename {temporary_path}")
    statistics = (0.0, 0.0, 0)
    try:
        with h5py.File(temporary_path, "w") as output:
            output.attrs["schema"] = f"freqflow_{args.data_name}_vae_latent_v1"
            output.attrs["dataset_name"] = args.data_name
            output.attrs["source_data_path"] = os.path.abspath(args.data_path)
            output.attrs["vae_checkpoint"] = os.path.abspath(args.vae_checkpoint)
            output.attrs["vae_config_json"] = json.dumps(checkpoint["vae_config"])
            output.attrs["layout"] = "TCHW"
            output.attrs["normalization"] = data_config["normalization"]
            output.attrs["data_config_json"] = json.dumps(data_config)
            output.attrs["posterior"] = "mode"
            for split in requested_splits:
                dataset = ShanghaiSequenceDataset(args.data_path, split, model.config.image_size)
                current = encode_split(output, model, dataset, split, args.batch_size, args.encode_batch_size, args.num_workers, device)
                if split == "train":
                    statistics = current
            if statistics[2]:
                mean = statistics[0] / statistics[2]
                variance = max(statistics[1] / statistics[2] - mean * mean, 1.0e-12)
                output.attrs["train_latent_mean"] = mean
                output.attrs["train_latent_std"] = float(variance**0.5)
        os.replace(temporary_path, output_path)
    except Exception:
        # Keep the partial file for diagnosis; never silently overwrite it.
        raise
    print(f"Wrote latent {args.data_name} HDF5: {output_path}")


if __name__ == "__main__":
    main()
