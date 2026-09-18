"""Diagnose a trained Meteo latent-FreqFlow checkpoint in radar pixel space.

The script compares two paths against the same resized raw-radar truth:

1. future latent -> frozen VAE decoder (the reconstruction upper bound), and
2. history latent -> FreqFlow sampling -> frozen VAE decoder (the forecast).

It does not train or mutate either checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm

from alphapre_compat.dataset_meteo_latent import (
    MeteoLatent,
    PIXEL_SCALE,
    THRESHOLDS,
)
from train_nowcast_alphapre import (
    build_freqflow,
    decode_future,
    sample_future,
    split_batch,
)
from vae_meteo import load_vae_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="precomputed latent HDF5")
    parser.add_argument("--raw-data-path", required=True, help="original Meteo radar HDF5")
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True, help="trained FreqFlow best.pt or last.pt")
    parser.add_argument("--checkpoint-state", choices=("ema", "model"), default="ema")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--frames-in", type=int, default=5)
    parser.add_argument("--frames-out", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--decode-batch-size", type=int, default=20)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--limit-batches", type=int, default=0, help="0 evaluates the complete test split")
    parser.add_argument(
        "--oracle-max-shift",
        type=int,
        default=24,
        help="maximum absolute 128px-grid translation tested by the position oracle",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def _as_list(value):
    return np.asarray(value, dtype=np.float64).tolist()


class PixelDiagnostics:
    """Accumulate per-lead continuous, categorical, and centroid errors."""

    def __init__(self, frames_out, thresholds, pixel_scale):
        self.frames_out = frames_out
        self.thresholds = tuple(int(value) for value in thresholds)
        self.pixel_scale = float(pixel_scale)
        zeros = lambda: np.zeros(frames_out, dtype=np.float64)
        self.num_values = zeros()
        self.num_sequences = zeros()
        self.abs_error = zeros()
        self.squared_error = zeros()
        self.signed_error = zeros()
        self.pred_sum = zeros()
        self.true_sum = zeros()
        self.pred_max_sum = zeros()
        self.true_max_sum = zeros()
        self.binary = {
            threshold: {
                name: zeros()
                for name in (
                    "hits", "misses", "false_alarms", "correct_negatives",
                    "pred_wet", "true_wet", "centroid_distance_sum",
                    "centroid_pairs", "gt_only_frames", "pred_only_frames",
                    "both_dry_frames",
                )
            }
            for threshold in self.thresholds
        }

    def update(self, prediction, truth):
        prediction = prediction.detach().float().cpu().clamp(0.0, 1.0)
        truth = truth.detach().float().cpu().clamp(0.0, 1.0)
        if prediction.shape != truth.shape:
            raise ValueError(f"prediction/truth mismatch: {prediction.shape} != {truth.shape}")
        if prediction.ndim != 5 or prediction.shape[1] != self.frames_out:
            raise ValueError(f"expected [B,{self.frames_out},C,H,W], got {prediction.shape}")

        pred = prediction.numpy()[:, :, 0]
        true = truth.numpy()[:, :, 0]
        batch, frames, height, width = pred.shape
        reduce_axes = (0, 2, 3)
        difference = pred - true
        pixels_per_lead = batch * height * width
        self.num_values += pixels_per_lead
        self.num_sequences += batch
        self.abs_error += np.abs(difference).sum(axis=reduce_axes)
        self.squared_error += np.square(difference).sum(axis=reduce_axes)
        self.signed_error += difference.sum(axis=reduce_axes)
        self.pred_sum += pred.sum(axis=reduce_axes)
        self.true_sum += true.sum(axis=reduce_axes)
        self.pred_max_sum += pred.max(axis=(2, 3)).sum(axis=0)
        self.true_max_sum += true.max(axis=(2, 3)).sum(axis=0)

        y_coords = np.arange(height, dtype=np.float64)[None, None, :, None]
        x_coords = np.arange(width, dtype=np.float64)[None, None, None, :]
        for threshold, values in self.binary.items():
            cutoff = threshold / self.pixel_scale
            pred_wet = pred >= cutoff
            true_wet = true >= cutoff
            values["hits"] += np.logical_and(pred_wet, true_wet).sum(axis=reduce_axes)
            values["misses"] += np.logical_and(~pred_wet, true_wet).sum(axis=reduce_axes)
            values["false_alarms"] += np.logical_and(pred_wet, ~true_wet).sum(axis=reduce_axes)
            values["correct_negatives"] += np.logical_and(~pred_wet, ~true_wet).sum(axis=reduce_axes)
            values["pred_wet"] += pred_wet.sum(axis=reduce_axes)
            values["true_wet"] += true_wet.sum(axis=reduce_axes)

            pred_mass = pred_wet.sum(axis=(2, 3), dtype=np.float64)
            true_mass = true_wet.sum(axis=(2, 3), dtype=np.float64)
            pred_exists = pred_mass > 0
            true_exists = true_mass > 0
            both = pred_exists & true_exists
            pred_cy = _divide((pred_wet * y_coords).sum(axis=(2, 3)), pred_mass)
            pred_cx = _divide((pred_wet * x_coords).sum(axis=(2, 3)), pred_mass)
            true_cy = _divide((true_wet * y_coords).sum(axis=(2, 3)), true_mass)
            true_cx = _divide((true_wet * x_coords).sum(axis=(2, 3)), true_mass)
            centroid_distance = np.sqrt(np.square(pred_cy - true_cy) + np.square(pred_cx - true_cx))
            values["centroid_distance_sum"] += (centroid_distance * both).sum(axis=0)
            values["centroid_pairs"] += both.sum(axis=0)
            values["gt_only_frames"] += np.logical_and(true_exists, ~pred_exists).sum(axis=0)
            values["pred_only_frames"] += np.logical_and(pred_exists, ~true_exists).sum(axis=0)
            values["both_dry_frames"] += np.logical_and(~pred_exists, ~true_exists).sum(axis=0)

    def finalize(self):
        continuous = {
            "mae_normalized": _as_list(_divide(self.abs_error, self.num_values)),
            "rmse_normalized": _as_list(np.sqrt(_divide(self.squared_error, self.num_values))),
            "bias_normalized": _as_list(_divide(self.signed_error, self.num_values)),
            "prediction_mean_normalized": _as_list(_divide(self.pred_sum, self.num_values)),
            "truth_mean_normalized": _as_list(_divide(self.true_sum, self.num_values)),
            "prediction_peak_mean_normalized": _as_list(_divide(self.pred_max_sum, self.num_sequences)),
            "truth_peak_mean_normalized": _as_list(_divide(self.true_max_sum, self.num_sequences)),
        }
        continuous["mae_radar_units"] = _as_list(
            np.asarray(continuous["mae_normalized"]) * self.pixel_scale
        )
        continuous["rmse_radar_units"] = _as_list(
            np.asarray(continuous["rmse_normalized"]) * self.pixel_scale
        )
        continuous["bias_radar_units"] = _as_list(
            np.asarray(continuous["bias_normalized"]) * self.pixel_scale
        )
        for name in (
            "prediction_mean_normalized",
            "truth_mean_normalized",
            "prediction_peak_mean_normalized",
            "truth_peak_mean_normalized",
        ):
            continuous[name.replace("_normalized", "_radar_units")] = _as_list(
                np.asarray(continuous[name]) * self.pixel_scale
            )

        categorical = {}
        spatial = {}
        for threshold, values in self.binary.items():
            hits = values["hits"]
            misses = values["misses"]
            false_alarms = values["false_alarms"]
            correct_negatives = values["correct_negatives"]
            csi = _divide(hits, hits + misses + false_alarms)
            far = _divide(false_alarms, hits + false_alarms)
            pod = _divide(hits, hits + misses)
            hss_numerator = 2.0 * (hits * correct_negatives - misses * false_alarms)
            hss_denominator = (
                (hits + misses) * (misses + correct_negatives)
                + (hits + false_alarms) * (false_alarms + correct_negatives)
            )
            hss = _divide(hss_numerator, hss_denominator)
            categorical[str(threshold)] = {
                "csi_per_lead": _as_list(csi),
                "far_per_lead": _as_list(far),
                "pod_per_lead": _as_list(pod),
                "hss_per_lead": _as_list(hss),
                "mean_csi_over_leads": float(csi.mean()),
                "prediction_wet_fraction_per_lead": _as_list(
                    _divide(values["pred_wet"], self.num_values)
                ),
                "truth_wet_fraction_per_lead": _as_list(
                    _divide(values["true_wet"], self.num_values)
                ),
            }
            spatial[str(threshold)] = {
                "centroid_distance_pixels_per_lead": _as_list(
                    _divide(values["centroid_distance_sum"], values["centroid_pairs"])
                ),
                "centroid_valid_pairs_per_lead": _as_list(values["centroid_pairs"]),
                "truth_wet_prediction_dry_frames_per_lead": _as_list(values["gt_only_frames"]),
                "truth_dry_prediction_wet_frames_per_lead": _as_list(values["pred_only_frames"]),
                "both_dry_frames_per_lead": _as_list(values["both_dry_frames"]),
            }
        return {
            "num_sequences": int(self.num_sequences[0]),
            "frames_out": self.frames_out,
            "lead_indices": list(range(1, self.frames_out + 1)),
            "continuous": continuous,
            "categorical": categorical,
            "spatial": spatial,
        }


class SpectralDiagnostics:
    """Accumulate radial pixel-domain power spectra for each forecast lead."""

    def __init__(self, frames_out, height, width):
        self.frames_out = frames_out
        self.height = height
        self.width = width
        # Frequencies are cycles per pixel.  The final edge includes diagonal
        # frequencies up to sqrt(0.5^2 + 0.5^2).
        self.band_edges = np.asarray(
            [0.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, math.sqrt(0.5)],
            dtype=np.float64,
        )
        fy = torch.fft.fftfreq(height)
        fx = torch.fft.fftfreq(width)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        self.masks = []
        for index in range(len(self.band_edges) - 1):
            lower, upper = self.band_edges[index : index + 2]
            if index == len(self.band_edges) - 2:
                mask = (radius >= lower) & (radius <= upper)
            else:
                mask = (radius >= lower) & (radius < upper)
            self.masks.append(mask)
        bands = len(self.masks)
        self.prediction_energy = np.zeros((frames_out, bands), dtype=np.float64)
        self.truth_energy = np.zeros((frames_out, bands), dtype=np.float64)
        self.num_sequences = 0

    def update(self, prediction, truth):
        pred = prediction.detach().float().cpu()[:, :, 0]
        true = truth.detach().float().cpu()[:, :, 0]
        # Remove the per-frame mean so DC intensity bias (reported separately)
        # does not hide structural low/mid/high-frequency errors.
        pred = pred - pred.mean(dim=(-2, -1), keepdim=True)
        true = true - true.mean(dim=(-2, -1), keepdim=True)
        pred_power = torch.fft.fft2(pred, norm="ortho").abs().square()
        true_power = torch.fft.fft2(true, norm="ortho").abs().square()
        for band_index, mask in enumerate(self.masks):
            self.prediction_energy[:, band_index] += (
                pred_power[..., mask].sum(dim=(0, 2)).double().numpy()
            )
            self.truth_energy[:, band_index] += (
                true_power[..., mask].sum(dim=(0, 2)).double().numpy()
            )
        self.num_sequences += pred.shape[0]

    def finalize(self):
        prediction_mean = self.prediction_energy / max(self.num_sequences, 1)
        truth_mean = self.truth_energy / max(self.num_sequences, 1)
        prediction_fraction = _divide(prediction_mean, prediction_mean.sum(axis=1, keepdims=True))
        truth_fraction = _divide(truth_mean, truth_mean.sum(axis=1, keepdims=True))
        labels = [
            f"{self.band_edges[index]:.5f}-{self.band_edges[index + 1]:.5f}"
            for index in range(len(self.band_edges) - 1)
        ]
        return {
            "frequency_unit": "cycles_per_pixel",
            "input_preprocessing": "per-frame spatial mean removed before FFT",
            "band_edges": self.band_edges.tolist(),
            "band_labels": labels,
            "mean_prediction_power_per_lead": prediction_mean.tolist(),
            "mean_truth_power_per_lead": truth_mean.tolist(),
            "relative_power_error_per_lead": _divide(
                prediction_mean - truth_mean, truth_mean
            ).tolist(),
            "prediction_power_fraction_per_lead": prediction_fraction.tolist(),
            "truth_power_fraction_per_lead": truth_fraction.tolist(),
        }


class OracleDiagnostics:
    """Measure truth-assisted position/intensity correction upper bounds.

    These are diagnostic oracles, not deployable forecast metrics.  Each frame
    receives at most one global integer translation and/or one global positive
    intensity multiplier selected using its ground truth.
    """

    def __init__(self, frames_out, thresholds, pixel_scale, max_shift):
        self.frames_out = frames_out
        self.max_shift = int(max_shift)
        self.position = PixelDiagnostics(frames_out, thresholds, pixel_scale)
        self.intensity = PixelDiagnostics(frames_out, thresholds, pixel_scale)
        self.position_intensity = PixelDiagnostics(frames_out, thresholds, pixel_scale)
        self.shift_dy_sum = np.zeros(frames_out, dtype=np.float64)
        self.shift_dx_sum = np.zeros(frames_out, dtype=np.float64)
        self.shift_distance_sum = np.zeros(frames_out, dtype=np.float64)
        self.shift_count = np.zeros(frames_out, dtype=np.float64)
        self.gain_sum = np.zeros(frames_out, dtype=np.float64)
        self.aligned_gain_sum = np.zeros(frames_out, dtype=np.float64)
        self.gain_count = np.zeros(frames_out, dtype=np.float64)

    @staticmethod
    def _zero_filled_shift(frame, dy, dx):
        """Translate one [H,W] frame without wrapping pixels across edges."""
        height, width = frame.shape
        shifted = torch.zeros_like(frame)
        source_y0 = max(0, -dy)
        source_y1 = min(height, height - dy)
        source_x0 = max(0, -dx)
        source_x1 = min(width, width - dx)
        target_y0 = max(0, dy)
        target_y1 = min(height, height + dy)
        target_x0 = max(0, dx)
        target_x1 = min(width, width + dx)
        if source_y1 > source_y0 and source_x1 > source_x0:
            shifted[target_y0:target_y1, target_x0:target_x1] = frame[
                source_y0:source_y1, source_x0:source_x1
            ]
        return shifted

    def _position_align(self, prediction, truth):
        pred = prediction[:, :, 0].float().cpu().clamp(0.0, 1.0)
        true = truth[:, :, 0].float().cpu().clamp(0.0, 1.0)
        batch, frames, height, width = pred.shape

        # Smooth sqrt-intensity fields emphasize precipitation placement while
        # reducing domination by a single extreme pixel.
        pred_signal = F.avg_pool2d(
            pred.sqrt().reshape(batch * frames, 1, height, width), 5, stride=1, padding=2
        ).reshape(batch, frames, height, width)
        true_signal = F.avg_pool2d(
            true.sqrt().reshape(batch * frames, 1, height, width), 5, stride=1, padding=2
        ).reshape(batch, frames, height, width)
        correlation = torch.fft.ifft2(
            torch.fft.fft2(true_signal) * torch.conj(torch.fft.fft2(pred_signal))
        ).real
        offsets = torch.arange(-self.max_shift, self.max_shift + 1)
        y_indices = offsets.remainder(height)
        x_indices = offsets.remainder(width)
        candidates = correlation[..., y_indices[:, None], x_indices[None, :]]
        best = candidates.reshape(batch, frames, -1).argmax(dim=-1)
        side = offsets.numel()
        dy = offsets[torch.div(best, side, rounding_mode="floor")]
        dx = offsets[best % side]
        valid = (pred_signal.square().sum(dim=(-2, -1)) > 1e-8) & (
            true_signal.square().sum(dim=(-2, -1)) > 1e-8
        )
        dy = torch.where(valid, dy, torch.zeros_like(dy))
        dx = torch.where(valid, dx, torch.zeros_like(dx))

        aligned = torch.zeros_like(pred)
        for batch_index in range(batch):
            for lead_index in range(frames):
                aligned[batch_index, lead_index] = self._zero_filled_shift(
                    pred[batch_index, lead_index],
                    int(dy[batch_index, lead_index]),
                    int(dx[batch_index, lead_index]),
                )
        return aligned.unsqueeze(2), dy, dx, valid

    @staticmethod
    def _intensity_scale(prediction, truth):
        pred = prediction.float().cpu().clamp(0.0, 1.0)
        true = truth.float().cpu().clamp(0.0, 1.0)
        true_energy = true.square().sum(dim=(-3, -2, -1))
        pred_energy = pred.square().sum(dim=(-3, -2, -1))
        gain = torch.where(
            pred_energy > 1e-8,
            torch.sqrt(true_energy / pred_energy.clamp_min(1e-8)),
            torch.ones_like(pred_energy),
        )
        # Keep this an interpretable calibration oracle rather than allowing an
        # unbounded multiplier on nearly empty predictions.
        gain = gain.clamp(0.25, 4.0)
        return (pred * gain[:, :, None, None, None]).clamp(0.0, 1.0), gain

    def update(self, prediction, truth):
        prediction = prediction.detach().float().cpu()
        truth = truth.detach().float().cpu()
        aligned, dy, dx, valid = self._position_align(prediction, truth)
        scaled, gain = self._intensity_scale(prediction, truth)
        aligned_scaled, aligned_gain = self._intensity_scale(aligned, truth)
        self.position.update(aligned, truth)
        self.intensity.update(scaled, truth)
        self.position_intensity.update(aligned_scaled, truth)

        valid_np = valid.numpy().astype(np.float64)
        dy_np = dy.numpy().astype(np.float64)
        dx_np = dx.numpy().astype(np.float64)
        self.shift_dy_sum += (dy_np * valid_np).sum(axis=0)
        self.shift_dx_sum += (dx_np * valid_np).sum(axis=0)
        self.shift_distance_sum += (
            np.sqrt(np.square(dy_np) + np.square(dx_np)) * valid_np
        ).sum(axis=0)
        self.shift_count += valid_np.sum(axis=0)
        self.gain_sum += gain.numpy().sum(axis=0)
        self.aligned_gain_sum += aligned_gain.numpy().sum(axis=0)
        self.gain_count += gain.shape[0]

    def finalize(self):
        return {
            "warning": (
                "Truth-assisted diagnostic upper bounds only; do not report these as deployable model scores."
            ),
            "position_oracle": {
                "definition": "one ground-truth-selected global integer translation per sample and lead",
                "max_absolute_shift_pixels": self.max_shift,
                "mean_dy_per_lead": _as_list(_divide(self.shift_dy_sum, self.shift_count)),
                "mean_dx_per_lead": _as_list(_divide(self.shift_dx_sum, self.shift_count)),
                "mean_shift_distance_pixels_per_lead": _as_list(
                    _divide(self.shift_distance_sum, self.shift_count)
                ),
                "metrics": self.position.finalize(),
            },
            "intensity_oracle": {
                "definition": "one ground-truth RMS-energy matching multiplier in [0.25,4] per sample and lead",
                "mean_gain_per_lead": _as_list(_divide(self.gain_sum, self.gain_count)),
                "metrics": self.intensity.finalize(),
            },
            "position_then_intensity_oracle": {
                "definition": "position oracle followed by the same bounded intensity calibration",
                "mean_gain_per_lead": _as_list(_divide(self.aligned_gain_sum, self.gain_count)),
                "metrics": self.position_intensity.finalize(),
            },
        }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def intensity_view(result):
    return {
        "num_sequences": result["num_sequences"],
        "lead_indices": result["lead_indices"],
        "continuous": result["continuous"],
        "wet_area_by_threshold": {
            threshold: {
                "prediction_wet_fraction_per_lead": values["prediction_wet_fraction_per_lead"],
                "truth_wet_fraction_per_lead": values["truth_wet_fraction_per_lead"],
            }
            for threshold, values in result["categorical"].items()
        },
    }


def main():
    args = parse_args()
    if args.frames_in != 5 or args.frames_out != 20:
        raise ValueError("the current Meteo latent model requires 5 input and 20 output frames")
    if args.batch_size < 1 or args.decode_batch_size < 1 or args.sample_steps < 1:
        raise ValueError("batch sizes and sample steps must be positive")
    if args.limit_batches < 0:
        raise ValueError("--limit-batches must be zero or positive")
    if args.oracle_max_shift < 0 or args.oracle_max_shift >= args.img_size // 2:
        raise ValueError("--oracle-max-shift must be in [0, img_size/2)")

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    process_seed = args.seed + accelerator.process_index
    torch.manual_seed(process_seed)
    np.random.seed(process_seed)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    dataset = MeteoLatent(
        latent_path=args.data_path,
        split="test",
        raw_data_path=args.raw_data_path,
        image_size=args.img_size,
        frames_in=args.frames_in,
        frames_out=args.frames_out,
        return_pixels=True,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    contract = {
        "latent_channels": dataset.latent_channels,
        "latent_height": dataset.latent_height,
        "latent_width": dataset.latent_width,
        "latent_mean": dataset.latent_mean,
        "latent_std": dataset.latent_std,
    }

    vae, _ = load_vae_checkpoint(args.vae_checkpoint, "cpu")
    vae = vae.to(accelerator.device).eval().requires_grad_(False)
    model = build_freqflow(
        args.frames_in,
        args.frames_out,
        args.img_size,
        dataset.latent_channels,
        dataset.latent_height,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.checkpoint_state not in checkpoint:
        raise KeyError(f"checkpoint has no state named {args.checkpoint_state!r}")
    model.load_state_dict(checkpoint[args.checkpoint_state], strict=True)
    model.eval()
    model, loader = accelerator.prepare(model, loader)

    forecast_metrics = PixelDiagnostics(args.frames_out, THRESHOLDS, PIXEL_SCALE)
    vae_metrics = PixelDiagnostics(args.frames_out, THRESHOLDS, PIXEL_SCALE)
    forecast_spectrum = SpectralDiagnostics(args.frames_out, args.img_size, args.img_size)
    vae_spectrum = SpectralDiagnostics(args.frames_out, args.img_size, args.img_size)
    oracle_metrics = OracleDiagnostics(
        args.frames_out, THRESHOLDS, PIXEL_SCALE, args.oracle_max_shift
    )

    total_batches = len(loader)
    if args.limit_batches:
        total_batches = min(total_batches, args.limit_batches)
    progress = tqdm(
        loader,
        total=total_batches,
        desc="FreqFlow diagnostics",
        disable=not accelerator.is_main_process,
        ncols=120,
    )
    processed_batches = 0
    with torch.inference_mode():
        for batch_index, (latent, future_pixels) in enumerate(progress):
            if args.limit_batches and batch_index >= args.limit_batches:
                break
            history, future = split_batch(
                latent,
                args.frames_in,
                args.frames_out,
                dataset.latent_channels,
                dataset.latent_mean,
                dataset.latent_std,
                accelerator.device,
            )
            with accelerator.autocast():
                prediction = sample_future(
                    model,
                    history,
                    args.frames_out * dataset.latent_channels,
                    args.sample_steps,
                )
            prediction_pixels = decode_future(
                vae,
                prediction,
                args.frames_out,
                dataset.latent_channels,
                dataset.latent_mean,
                dataset.latent_std,
                args.decode_batch_size,
                accelerator,
            )
            reconstruction_pixels = decode_future(
                vae,
                future,
                args.frames_out,
                dataset.latent_channels,
                dataset.latent_mean,
                dataset.latent_std,
                args.decode_batch_size,
                accelerator,
            )
            truth = future_pixels.to(accelerator.device, non_blocking=True)
            gathered_truth = accelerator.gather_for_metrics(truth).float().cpu()
            gathered_prediction = accelerator.gather_for_metrics(prediction_pixels).float().cpu()
            gathered_reconstruction = accelerator.gather_for_metrics(reconstruction_pixels).float().cpu()
            if accelerator.is_main_process:
                forecast_metrics.update(gathered_prediction, gathered_truth)
                vae_metrics.update(gathered_reconstruction, gathered_truth)
                forecast_spectrum.update(gathered_prediction, gathered_truth)
                vae_spectrum.update(gathered_reconstruction, gathered_truth)
                oracle_metrics.update(gathered_prediction, gathered_truth)
            processed_batches += 1
    progress.close()

    if accelerator.is_main_process:
        forecast_result = forecast_metrics.finalize()
        vae_result = vae_metrics.finalize()
        write_json(output_dir / "metrics_per_lead.json", forecast_result)
        write_json(output_dir / "vae_upper_bound.json", vae_result)
        write_json(
            output_dir / "intensity_bias.json",
            {
                "forecast": intensity_view(forecast_result),
                "vae_reconstruction": intensity_view(vae_result),
            },
        )
        write_json(
            output_dir / "spatial_error.json",
            {
                "distance_unit": "pixels_at_128x128",
                "forecast": forecast_result["spatial"],
                "vae_reconstruction": vae_result["spatial"],
            },
        )
        write_json(
            output_dir / "spectral_error.json",
            {
                "forecast": forecast_spectrum.finalize(),
                "vae_reconstruction": vae_spectrum.finalize(),
            },
        )
        oracle_result = oracle_metrics.finalize()
        write_json(output_dir / "oracle_diagnostics.json", oracle_result)
        write_json(
            output_dir / "summary.json",
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_state": args.checkpoint_state,
                "vae_checkpoint": str(Path(args.vae_checkpoint).resolve()),
                "latent_data": str(Path(args.data_path).resolve()),
                "raw_data": str(Path(args.raw_data_path).resolve()),
                "sample_steps": args.sample_steps,
                "mixed_precision": args.mixed_precision,
                "seed": args.seed,
                "limit_batches": args.limit_batches,
                "processed_batches_per_process": processed_batches,
                "num_sequences": forecast_result["num_sequences"],
                "thresholds": list(THRESHOLDS),
                "oracle_max_shift": args.oracle_max_shift,
                "output_files": [
                    "metrics_per_lead.json",
                    "vae_upper_bound.json",
                    "intensity_bias.json",
                    "spatial_error.json",
                    "spectral_error.json",
                    "oracle_diagnostics.json",
                ],
            },
        )
        print(f"Diagnostics complete: {output_dir}", flush=True)
        print(f"Evaluated sequences: {forecast_result['num_sequences']}", flush=True)
        for threshold in THRESHOLDS:
            forecast_csi = forecast_result["categorical"][str(threshold)]["mean_csi_over_leads"]
            vae_csi = vae_result["categorical"][str(threshold)]["mean_csi_over_leads"]
            position_csi = oracle_result["position_oracle"]["metrics"]["categorical"][str(threshold)]["mean_csi_over_leads"]
            intensity_csi = oracle_result["intensity_oracle"]["metrics"]["categorical"][str(threshold)]["mean_csi_over_leads"]
            combined_csi = oracle_result["position_then_intensity_oracle"]["metrics"]["categorical"][str(threshold)]["mean_csi_over_leads"]
            print(
                f"Threshold {threshold}: forecast mean CSI={forecast_csi:.6f}; "
                f"VAE reconstruction mean CSI={vae_csi:.6f}; "
                f"position oracle={position_csi:.6f}; intensity oracle={intensity_csi:.6f}; "
                f"combined oracle={combined_csi:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
