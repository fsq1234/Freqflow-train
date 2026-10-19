"""Synthetic Shanghai data/latent contracts, without Diffusers or a training run."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms

import generate_shanghai_latents
from train_vae_shanghai import parse_args
from vae_meteo import MeteoFrameDataset, MeteoSequenceDataset
from vae_shanghai import (
    ShanghaiFrameDataset, ShanghaiSequenceDataset, shanghai_data_config, validate_vae_data_config,
)


class PoolingCodec:
    """Test double for encoding/chunking/statistics, not a trained VAE."""

    config = SimpleNamespace(image_size=128)

    def encode(self, frames):
        latent = F.avg_pool2d(frames, 4).repeat(1, 4, 1, 1)
        return SimpleNamespace(mode=lambda: latent)


class ShanghaiVAETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        workspace = Path(__file__).resolve().parent
        self.temporary = tempfile.TemporaryDirectory(prefix=".test-shanghai-vae-", dir=workspace)
        self.root = Path(self.temporary.name).resolve()
        if self.root.parent != workspace:
            raise RuntimeError("Test cleanup must remain inside the project directory")
        self.addCleanup(self.temporary.cleanup)
        self.shanghai = self.root / "shanghai.h5"
        self.meteo = self.root / "meteo.h5"
        self.events = {}
        for path, meteo in ((self.shanghai, False), (self.meteo, True)):
            with h5py.File(path, "w") as output:
                for split, count, offset in (("train", 2, 0), ("test", 1, 150)):
                    group = output.create_group(split)
                    if meteo:
                        output.create_dataset(f"{split}_len", data=count)
                    else:
                        group.create_dataset("all_len", data=count)
                    for index in range(count):
                        raw = ((np.arange(25 * 6 * 10).reshape(25, 6, 10)
                                + offset + index * 29) % 256).astype(np.uint8)
                        group.create_dataset(str(index), data=raw)
                        self.events[split, index] = raw

    def test_single_frame_matches_existing_shanghai_preprocessing(self):
        dataset = ShanghaiFrameDataset(self.shanghai, "train")
        with patch("vae_shanghai.np.random.randint", return_value=7):
            frame, event_index, frame_index = dataset[1]
        expected_sequence = transforms.Resize((128, 128))(
            torch.from_numpy(self.events["train", 1]).float() / 255.0
        )
        self.assertEqual(len(dataset), 2)
        self.assertEqual((event_index, frame_index), (1, 7))
        self.assertEqual(frame.shape, (1, 128, 128))
        torch.testing.assert_close(frame, expected_sequence[7:8], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(frame).all())

    def test_sequence_and_validation_follow_existing_split_protocol(self):
        dataset = ShanghaiSequenceDataset(self.shanghai, "val")
        sequence, event_index = dataset[0]
        self.assertEqual(dataset.split, "test")
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.frames_per_event, 25)
        self.assertEqual(sequence.shape, (25, 1, 128, 128))
        expected = transforms.Resize((128, 128))(
            torch.from_numpy(self.events["test", 0]).float() / 255.0
        ).unsqueeze(1)
        torch.testing.assert_close(sequence, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            ShanghaiFrameDataset(self.shanghai, "unknown")

    def test_original_meteo_data_is_preserved_and_shanghai_defaults_match(self):
        sequence, _ = MeteoSequenceDataset(self.meteo, "train")[0]
        expected = transforms.Resize((128, 128))(
            torch.from_numpy(self.events["train", 0]).float() / 90.0
        ).unsqueeze(1)
        torch.testing.assert_close(sequence, expected, rtol=0, atol=0)
        with patch("vae_meteo.np.random.randint", return_value=3):
            frame, _, index = MeteoFrameDataset(self.meteo, "train")[0]
        torch.testing.assert_close(frame, expected[3])
        args = ["--data-path", "source.h5", "--workdir", "runs/test"]
        shanghai = parse_args(args)
        self.assertEqual(shanghai.data_name, "shanghai")
        self.assertEqual(shanghai.wandb_project, "freqflow-shanghai-vae")
        for name, expected in (("epochs", 250), ("batch_size", 3), ("lr", 1e-4),
                               ("kl_weight", 1e-4), ("disc_weight", 0.5),
                               ("disc_start_epoch", 35), ("mixed_precision", "no")):
            self.assertEqual(getattr(shanghai, name), expected)

    def test_checkpoint_dataset_and_normalization_must_match(self):
        shanghai = shanghai_data_config()
        meteo = {**shanghai, "dataset_name": "meteo", "normalization_divisor": 90.0,
                 "thresholds": [12, 18, 24, 32]}
        self.assertEqual(shanghai["normalization_divisor"], 255.0)
        self.assertEqual(shanghai["pixel_scale"], 90.0)
        self.assertEqual(shanghai["thresholds"], [20, 30, 35, 40])
        validate_vae_data_config({"data_config": shanghai}, shanghai)
        for saved in ({}, {"data_config": meteo},
                      {"data_config": {**shanghai, "normalization_divisor": 90.0}}):
            with self.assertRaises(ValueError):
                validate_vae_data_config(saved, shanghai)

    def test_latent_artifact_preserves_events_and_uses_train_statistics_only(self):
        output_path = self.root / "shanghai_latent.h5"
        source_digest = hashlib.sha256(self.shanghai.read_bytes()).digest()
        checkpoint = {"vae_config": {"image_size": 128, "latent_channels": 4},
                      "data_config": shanghai_data_config()}
        arguments = ["--data-path", str(self.shanghai), "--vae-checkpoint", "synthetic.pt",
                     "--output-path", str(output_path), "--num-workers", "0",
                     "--batch-size", "2", "--encode-batch-size", "7", "--device", "cpu"]
        with patch("generate_shanghai_latents.load_vae_checkpoint", return_value=(PoolingCodec(), checkpoint)):
            generate_shanghai_latents.main(arguments)
        self.assertEqual(hashlib.sha256(self.shanghai.read_bytes()).digest(), source_digest)
        with h5py.File(output_path, "r") as output:
            self.assertEqual(output.attrs["schema"], "freqflow_shanghai_vae_latent_v1")
            self.assertEqual(output.attrs["posterior"], "mode")
            self.assertIn("255.0", output.attrs["normalization"])
            self.assertEqual(output["train"].attrs["num_events"], 2)
            self.assertEqual(output["test"].attrs["num_events"], 1)
            training_values = []
            for split, count in (("train", 2), ("test", 1)):
                dataset = ShanghaiSequenceDataset(self.shanghai, split)
                for index in range(count):
                    latent = output[split][str(index)][()]
                    self.assertEqual(latent.shape, (25, 4, 32, 32))
                    self.assertEqual(latent.dtype, np.float32)
                    sequence, _ = dataset[index]
                    expected = PoolingCodec().encode(sequence).mode()
                    torch.testing.assert_close(torch.from_numpy(latent), expected)
                    if split == "train":
                        training_values.append(latent.astype(np.float64).reshape(-1))
            training_values = np.concatenate(training_values)
            self.assertAlmostEqual(output.attrs["train_latent_mean"], training_values.mean(), places=10)
            self.assertAlmostEqual(output.attrs["train_latent_std"], training_values.std(), places=10)
        self.assertFalse(output_path.with_suffix(".h5.partial").exists())
        with self.assertRaises(FileExistsError):
            generate_shanghai_latents.main(arguments)


if __name__ == "__main__":
    unittest.main()
