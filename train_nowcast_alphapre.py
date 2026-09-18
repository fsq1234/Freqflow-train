"""Train conditional FreqFlow on frozen Meteo VAE latents."""

import argparse
import copy
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
import utils
from alphapre_compat import metrics, metrics_valid
from alphapre_compat.dataset_meteo_latent import MeteoLatent, PIXEL_SCALE, THRESHOLDS
from configs.freqflow_nowcast_alphapre import get_config as get_nowcast_config
from fft2d import Fourier_filter
from sde import multi_scale_targets
from vae_meteo import load_vae_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("meteo",), default="meteo")
    parser.add_argument(
        "--data-path", required=True,
        help="precomputed meteo_latent_vae32.h5",
    )
    parser.add_argument("--raw-data-path", required=True, help="original meteo_radar.h5 used as metric truth")
    parser.add_argument("--vae-checkpoint", required=True, help="frozen Meteo VAE best_val_loss.pt")
    parser.add_argument("--workdir", default="workdir/freqflow_meteo_vae32_5to20")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=13, help="legacy launcher option; unused for MeteoNet")
    parser.add_argument("--frames-in", type=int, default=5)
    parser.add_argument("--frames-out", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--ema-rate", type=float, default=0.95)
    parser.add_argument("--ema-update-every", type=int, default=20)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--decode-batch-size", type=int, default=20, help="frames decoded per frozen-VAE call")
    parser.add_argument("--log-every", type=int, default=20, help="write terminal/W&B training progress every N steps")
    parser.add_argument("--use-wandb", action="store_true", help="enable Weights & Biases experiment tracking")
    parser.add_argument("--wandb-project", default="freqflow-nowcast", help="Weights & Biases project name")
    parser.add_argument("--wandb-name", default=None, help="Weights & Biases run display name")
    parser.add_argument("--wandb-group", default=None, help="Weights & Biases run group")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--valid", action="store_true", help="run AlphaPre validation and choose best.pt by mean CSI")
    parser.add_argument("--valid-limit", action="store_true")
    parser.add_argument("--vlnum", type=int, default=10)
    parser.add_argument(
        "--final-test-checkpoint", choices=("last", "best"), default="last",
        help="EMA checkpoint evaluated automatically on the full test set after training",
    )
    parser.add_argument("--skip-final-test", action="store_true", help="skip only the automatic post-training full test")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--horizon-frequency-router",
        action="store_true",
        help="enable lead-time/history-conditioned latent frequency routing",
    )
    parser.add_argument(
        "--frequency-router-mode",
        choices=("time", "time_lead", "time_lead_history", "time_lead_history_energy"),
        default="time_lead_history",
        help="conditioning used by the horizon frequency router",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="initialize model weights from a checkpoint without restoring optimizer/epoch state",
    )
    parser.add_argument(
        "--init-checkpoint-state",
        choices=("model", "ema"),
        default="ema",
        help="checkpoint state used with --init-checkpoint",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_freqflow(
    frames_in, frames_out, img_size, latent_channels, latent_size,
    horizon_frequency_router=False, frequency_router_mode="time_lead_history",
):
    """Build the full-capacity original FreqFlow backbone for radar latents."""
    config = get_nowcast_config()
    actual = (img_size, frames_in, frames_out, latent_channels, latent_size)
    expected = (
        config.radar_img_size, config.frames_in, config.frames_out,
        config.latent_channels, config.latent_size,
    )
    if actual != expected:
        raise ValueError(
            "the full-capacity latent config is fixed to 128px radar, 5 input frames, "
            "20 output frames, and 4x32x32 latents"
        )
    model_args = copy.deepcopy(config.nnet.model_args)
    if model_args.channels != frames_out * latent_channels:
        raise ValueError("model output channels do not match frames_out * latent_channels")
    if model_args.history_channels != frames_in * latent_channels:
        raise ValueError("model history channels do not match frames_in * latent_channels")
    model_args.horizon_frequency_router = horizon_frequency_router
    model_args.horizon_frequency_router_mode = frequency_router_mode
    return utils.get_nnet(name=config.nnet.name, model_args=model_args)


def as_plain_config(value):
    """Convert the lightweight config objects into checkpoint-safe primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [as_plain_config(item) for item in value]
    if isinstance(value, dict):
        return {key: as_plain_config(item) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        return {key: as_plain_config(item) for key, item in vars(value).items()}
    return repr(value)


def make_run_config(args):
    """Persist the task/backbone contract alongside each checkpoint."""
    config = get_nowcast_config()
    return {
        "model": as_plain_config(config.nnet.model_args),
        "task": {
            "radar_img_size": config.radar_img_size,
            "latent_size": config.latent_size,
            "latent_channels": config.latent_channels,
            "frames_in": config.frames_in,
            "frames_out": config.frames_out,
        },
        "arguments": vars(args).copy(),
    }


def init_wandb(args, workdir, run_config, accelerator):
    """Mirror PercpCast's resumable, main-process-only WandB bookkeeping."""
    if not args.use_wandb or not accelerator.is_main_process:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "WandB is enabled but unavailable. Install it in the training environment with "
            "`python -m pip install wandb`, then run `wandb login`."
        ) from exc

    run_id_path = workdir / "wandb_run_id.txt"
    if run_id_path.is_file():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    else:
        run_id = wandb.util.generate_id()
        run_id_path.write_text(run_id, encoding="utf-8")

    run = wandb.init(
        id=run_id,
        project=args.wandb_project,
        name=args.wandb_name or workdir.name,
        group=args.wandb_group or None,
        config=run_config,
        dir=str(workdir),
        resume="allow",
        allow_val_change=True,
    )
    if getattr(run, "url", None):
        print(f"W&B run: {run.url}", flush=True)
    return run


def reduce_losses(accelerator, losses):
    """Return per-step loss means across all Accelerate processes."""
    return {
        name: accelerator.gather(value.detach().float().reshape(1)).mean().item()
        for name, value in losses.items()
    }


def mean_epoch_losses(accelerator, loss_sums, loss_count):
    """Compute global epoch means without assuming a single GPU."""
    count = torch.tensor([float(loss_count)], device=accelerator.device)
    global_count = accelerator.gather(count).sum().item()
    return {
        name: accelerator.gather(
            torch.tensor([value], device=accelerator.device, dtype=torch.float32)
        ).sum().item() / max(global_count, 1.0)
        for name, value in loss_sums.items()
    }


def build_loaders(data_path, raw_data_path, img_size, frames_in, frames_out, batch_size, num_workers):
    """Build worker-safe latent loaders and raw-target validation loaders."""
    common = dict(
        latent_path=data_path, raw_data_path=raw_data_path, image_size=img_size,
        frames_in=frames_in, frames_out=frames_out,
    )
    train = MeteoLatent(split="train", return_pixels=False, **common)
    valid = MeteoLatent(split="val", return_pixels=True, **common)
    test = MeteoLatent(split="test", return_pixels=True, **common)
    contract = {
        "latent_channels": train.latent_channels,
        "latent_height": train.latent_height,
        "latent_width": train.latent_width,
        "latent_mean": train.latent_mean,
        "latent_std": train.latent_std,
        "vae_image_size": train.vae_image_size,
        "vae_latent_channels": train.vae_latent_channels,
    }
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, valid_loader, test_loader, PIXEL_SCALE, THRESHOLDS, contract


def reduce_frequency_gate(accelerator, model):
    """Return the process-averaged low-frequency gate for each forecast lead."""
    gate = getattr(accelerator.unwrap_model(model), "last_frequency_gate", None)
    if gate is None:
        return None
    local_by_lead = gate.detach().float().mean(dim=(0, 2, 3, 4)).unsqueeze(0)
    gathered = accelerator.gather(local_by_lead)
    return gathered.mean(dim=0).cpu()


def load_initial_model_state(module, checkpoint_state, router_enabled):
    """Load a baseline state while allowing only newly added router parameters."""
    incompatible = module.load_state_dict(checkpoint_state, strict=False)
    allowed_prefix = "horizon_frequency_router."
    bad_missing = [
        key for key in incompatible.missing_keys
        if not (router_enabled and key.startswith(allowed_prefix))
    ]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "incompatible initialization checkpoint: "
            f"missing={bad_missing}, unexpected={incompatible.unexpected_keys}"
        )
    return incompatible.missing_keys


def scale_corrected_target(value, output_height):
    """Match FreqFlow's existing target amplitude convention at a branch size."""
    full_height = value.shape[-2]
    if full_height == output_height:
        return value
    ratio = full_height // output_height
    if ratio < 1 or ratio & (ratio - 1) or full_height % output_height:
        raise ValueError(f"unsupported target resize {full_height} -> {output_height}")
    levels = int(math.log2(ratio)) + 1
    return multi_scale_targets(value, levels=levels, scale_correction=True)[output_height]


def conditional_freqflow_loss(model, history, future):
    """FlowCast-style CFM path with FreqFlow's spatial/frequency supervision."""
    noise = torch.randn_like(future)
    t = torch.rand(future.shape[0], device=future.device, dtype=future.dtype)
    t_like = t[:, None, None, None]
    flow_state = (1.0 - t_like) * noise + t_like * future
    velocity = future - noise

    predictions = model(flow_state, t=t, log_snr=t * 8.0 - 4.0, history=history)
    frequency_prediction, spatial_prediction = predictions
    low, high = Fourier_filter(velocity, D=8)
    frequency_target = scale_corrected_target(
        torch.cat([low, high, velocity], dim=1), frequency_prediction.shape[-1]
    )
    spatial_target = scale_corrected_target(velocity, spatial_prediction.shape[-1])
    spatial_loss = F.mse_loss(spatial_prediction, spatial_target)
    frequency_loss = F.mse_loss(frequency_prediction, frequency_target)
    total_loss = spatial_loss + frequency_loss / 4.0
    return total_loss, {"total_loss": total_loss.detach(), "spatial_loss": spatial_loss.detach(), "frequency_loss": frequency_loss.detach()}


@torch.no_grad()
def sample_future(model, history, future_channels, sample_steps):
    state = torch.randn(
        history.shape[0], future_channels, *history.shape[-2:],
        device=history.device, dtype=history.dtype,
    )
    dt = 1.0 / sample_steps
    for step in range(sample_steps):
        t = torch.full((state.shape[0],), step * dt, device=state.device, dtype=state.dtype)
        velocity = model(state, t=t, log_snr=t * 8.0 - 4.0, history=history)[-1]
        state = state + dt * velocity
    return state


def split_batch(batch, frames_in, frames_out, latent_channels, latent_mean, latent_std, device):
    """Normalize [B,T,C,H,W] and flatten time/channel for FreqFlow."""
    sequence = batch[:, : frames_in + frames_out].to(device, non_blocking=True)
    if sequence.ndim != 5 or sequence.shape[2] != latent_channels:
        raise ValueError(
            f"expected latent [B,T,{latent_channels},H,W], got {tuple(sequence.shape)}"
        )
    sequence = (sequence - latent_mean) / latent_std
    batch_size, _, _, height, width = sequence.shape
    history = sequence[:, :frames_in].reshape(
        batch_size, frames_in * latent_channels, height, width
    )
    future = sequence[:, frames_in : frames_in + frames_out].reshape(
        batch_size, frames_out * latent_channels, height, width
    )
    return history, future


@torch.no_grad()
def decode_future(
    vae, normalized_latent, frames_out, latent_channels,
    latent_mean, latent_std, decode_batch_size, accelerator,
):
    """Return decoded radar frames in AlphaPre's [B,T,1,H,W] contract."""
    batch_size, channels, height, width = normalized_latent.shape
    expected_channels = frames_out * latent_channels
    if channels != expected_channels:
        raise ValueError(f"expected {expected_channels} generated channels, got {channels}")
    latent = normalized_latent.float().reshape(
        batch_size * frames_out, latent_channels, height, width
    )
    latent = latent * latent_std + latent_mean
    decoded = []
    for start in range(0, latent.shape[0], decode_batch_size):
        with accelerator.autocast():
            decoded.append(vae.decode(latent[start : start + decode_batch_size]))
    radar = torch.cat(decoded, dim=0).float().clamp_(0.0, 1.0)
    return radar.reshape(batch_size, frames_out, 1, *radar.shape[-2:])


def cosine_schedule(optimizer, warmup_steps, total_steps):
    def factor(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def update_ema(ema_model, model, rate):
    utils.ema(ema_model, model, rate)


@torch.no_grad()
def evaluate(
    accelerator,
    model,
    vae,
    loader,
    args,
    evaluator_cls,
    pixel_scale,
    thresholds,
    latent_contract,
    limit=None,
    description="Evaluation",
):
    evaluator = evaluator_cls(seq_len=args.frames_out, value_scale=pixel_scale, thresholds=thresholds)
    model.eval()
    total = min(len(loader), limit) if limit is not None else len(loader)
    progress = tqdm(
        loader,
        total=total,
        desc=description,
        disable=not accelerator.is_main_process,
        ncols=120,
        leave=False,
        mininterval=1.0,
    )
    for index, (latent, future_pixels) in enumerate(progress):
        history, _ = split_batch(
            latent, args.frames_in, args.frames_out,
            latent_contract["latent_channels"], latent_contract["latent_mean"],
            latent_contract["latent_std"], accelerator.device,
        )
        with accelerator.autocast():
            prediction = sample_future(
                model, history, args.frames_out * latent_contract["latent_channels"],
                args.sample_steps,
            )
        prediction_pixels = decode_future(
            vae, prediction, args.frames_out, latent_contract["latent_channels"],
            latent_contract["latent_mean"], latent_contract["latent_std"],
            args.decode_batch_size, accelerator,
        )
        gathered_future = accelerator.gather_for_metrics(
            future_pixels.to(accelerator.device, non_blocking=True)
        ).float().cpu()
        gathered_prediction = accelerator.gather_for_metrics(prediction_pixels).float().cpu()
        if accelerator.is_main_process:
            evaluator.evaluate(gathered_future, gathered_prediction)
        if limit is not None and index + 1 >= limit:
            break
    progress.close()
    return evaluator.done(is_main_process=True) if accelerator.is_main_process else None


def save_checkpoint(path, accelerator, model, ema_model, optimizer, scheduler, epoch, step, best_csi, run_config):
    if not accelerator.is_main_process:
        return
    os.makedirs(path.parent, exist_ok=True)
    torch.save({
        "model": accelerator.unwrap_model(model).state_dict(),
        "ema": ema_model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "epoch": epoch, "step": step,
        "best_csi": float(best_csi), "run_config": run_config,
    }, path)


def append_epoch_metrics(workdir, epoch, step, epoch_losses, lr, validation, best_csi, elapsed_seconds):
    """Keep a small append-only epoch record next to the existing train.log."""
    validation_csi = "disabled" if validation is None else f"{float(validation['csi']):.8f}"
    with (workdir / "metrics.txt").open("a", encoding="utf-8") as handle:
        handle.write(
            f"epoch={epoch + 1} global_step={step} "
            f"train_total={epoch_losses['total_loss']:.8f} "
            f"train_spatial={epoch_losses['spatial_loss']:.8f} "
            f"train_frequency={epoch_losses['frequency_loss']:.8f} "
            f"lr={lr:.10g} validation_csi={validation_csi} "
            f"best_csi={best_csi:.8f} elapsed_seconds={elapsed_seconds:.2f}\n"
        )


def main():
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.frames_in != 5 or args.frames_out != 20:
        raise ValueError("AlphaPre-compatible runs require --frames-in 5 --frames-out 20")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1")
    if args.decode_batch_size < 1:
        raise ValueError("--decode-batch-size must be at least 1")
    torch.manual_seed(args.seed)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    workdir = Path(args.workdir)
    if accelerator.is_main_process:
        workdir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    train_loader, valid_loader, test_loader, pixel_scale, thresholds, latent_contract = build_loaders(
        args.data_path, args.raw_data_path, args.img_size, args.frames_in, args.frames_out,
        args.batch_size, args.num_workers,
    )
    latent_channels = latent_contract["latent_channels"]
    latent_size = latent_contract["latent_height"]
    if latent_contract["latent_width"] != latent_size:
        raise ValueError("FreqFlow requires square latent frames")
    if latent_contract["vae_latent_channels"] != latent_channels:
        raise ValueError("latent HDF5 shape disagrees with its stored VAE config")
    if latent_contract["vae_image_size"] != args.img_size:
        raise ValueError("VAE image size disagrees with --img-size")

    vae, _ = load_vae_checkpoint(args.vae_checkpoint, "cpu")
    vae = vae.to(accelerator.device).eval()
    if vae.config.latent_channels != latent_channels or vae.config.image_size != args.img_size:
        raise ValueError("selected VAE checkpoint is incompatible with the latent HDF5")
    vae.requires_grad_(False)

    run_config = make_run_config(args)
    run_config["latent_contract"] = as_plain_config(latent_contract)
    wandb_run = init_wandb(args, workdir, run_config, accelerator)
    model = build_freqflow(
        args.frames_in, args.frames_out, args.img_size, latent_channels, latent_size,
        horizon_frequency_router=args.horizon_frequency_router,
        frequency_router_mode=args.frequency_router_mode,
    )
    ema_model = copy.deepcopy(model).eval().to(accelerator.device)
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = cosine_schedule(optimizer, args.warmup_steps, total_steps)
    model, optimizer, scheduler, train_loader, valid_loader, test_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, valid_loader, test_loader
    )

    start_epoch = step = 0
    best_csi = float("-inf")
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if args.init_checkpoint_state not in checkpoint:
            raise KeyError(
                f"checkpoint has no {args.init_checkpoint_state!r} state: {args.init_checkpoint}"
            )
        initial_state = checkpoint[args.init_checkpoint_state]
        missing = load_initial_model_state(
            accelerator.unwrap_model(model), initial_state, args.horizon_frequency_router
        )
        load_initial_model_state(ema_model, initial_state, args.horizon_frequency_router)
        if accelerator.is_main_process:
            print(
                f"Initialized model and EMA from {args.init_checkpoint_state} state: "
                f"{args.init_checkpoint}",
                flush=True,
            )
            if missing:
                print(
                    f"Fresh horizon frequency router parameters: {len(missing)} tensors",
                    flush=True,
                )
        del initial_state, checkpoint
    elif args.resume:
        # These are self-produced, trusted training checkpoints containing
        # optimizer state and metadata.  PyTorch 2.6+ defaults to
        # ``weights_only=True``, which rejects the NumPy scalar stored by older
        # runs even though the checkpoint itself is valid.
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, step, best_csi = checkpoint["epoch"] + 1, checkpoint["step"], checkpoint["best_csi"]

    if accelerator.is_main_process:
        print("============================================================", flush=True)
        print("                 FreqFlow Training Start", flush=True)
        print("============================================================", flush=True)
        print(f"Device: {accelerator.device}", flush=True)
        print(f"GPU Count: {torch.cuda.device_count()}", flush=True)
        print(f"Processes: {accelerator.num_processes}", flush=True)
        print(
            "Horizon Frequency Router: "
            f"{args.horizon_frequency_router} ({args.frequency_router_mode})",
            flush=True,
        )
        print(f"Train Batch Size (per process): {args.batch_size}", flush=True)
        print(f"Train Data: {len(train_loader)} batches per process", flush=True)
        print(f"Val Data: {len(valid_loader)} batches per process", flush=True)
        print(f"Test Data: {len(test_loader)} batches per process", flush=True)
        print(f"Total Epochs: {args.epochs}", flush=True)
        print("Optimizer: AdamW (betas=(0.9, 0.95))", flush=True)
        print(f"Learning Rate: {args.lr}", flush=True)
        print(f"Mixed Precision: {args.mixed_precision}", flush=True)
        print(f"Pixel Scale: {pixel_scale}", flush=True)
        print(f"Thresholds: {thresholds}", flush=True)
        print(
            f"Latent contract: {args.frames_in}x{latent_channels} -> "
            f"{args.frames_out}x{latent_channels} at {latent_size}x{latent_size}",
            flush=True,
        )
        print(
            f"Latent normalization: mean={latent_contract['latent_mean']:.8f}, "
            f"std={latent_contract['latent_std']:.8f}", flush=True,
        )
        print(f"Frozen VAE: {args.vae_checkpoint}", flush=True)
        print("Model: original full-capacity FreqFlow backbone in VAE space", flush=True)
        print(f"Workdir: {workdir}", flush=True)
        if args.resume:
            print(f"Resume checkpoint: {args.resume}", flush=True)
        if args.use_wandb:
            print(
                f"W&B: enabled (project={args.wandb_project}, "
                f"group={args.wandb_group or 'none'})",
                flush=True,
            )
        else:
            print("W&B: disabled", flush=True)
        print("============================================================", flush=True)

    global_pbar = tqdm(
        initial=step,
        total=args.epochs * len(train_loader),
        disable=not accelerator.is_main_process,
        desc="Overall Progress",
        mininterval=1.0,
    )
    lr = scheduler.get_last_lr()[0]
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.monotonic()
        if accelerator.is_main_process:
            print(f"\n========== Epoch {epoch + 1}/{args.epochs} start ==========", flush=True)
        model.train()
        epoch_loss_sums = {"total_loss": 0.0, "spatial_loss": 0.0, "frequency_loss": 0.0}
        epoch_loss_count = 0
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_main_process,
            ncols=120,
            mininterval=1.0,
        )
        for batch_index, batch in enumerate(train_pbar, start=1):
            history, future = split_batch(
                batch, args.frames_in, args.frames_out, latent_channels,
                latent_contract["latent_mean"], latent_contract["latent_std"],
                accelerator.device,
            )
            with accelerator.autocast():
                loss, losses = conditional_freqflow_loss(model, history, future)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if step % args.ema_update_every == 0:
                update_ema(ema_model, accelerator.unwrap_model(model), args.ema_rate)
            lr = scheduler.get_last_lr()[0]
            local_losses = {name: value.float().item() for name, value in losses.items()}
            for name, value in local_losses.items():
                epoch_loss_sums[name] += value
            epoch_loss_count += 1

            if accelerator.is_main_process:
                train_pbar.set_postfix(
                    loss=f"{local_losses['total_loss']:.4f}",
                    spatial=f"{local_losses['spatial_loss']:.4f}",
                    frequency=f"{local_losses['frequency_loss']:.4f}",
                    lr=f"{lr:.2e}",
                )
                global_pbar.update(1)

            should_log = batch_index == 1 or batch_index % args.log_every == 0 or batch_index == len(train_loader)
            if should_log:
                logged_losses = reduce_losses(accelerator, losses)
                gate_by_lead = reduce_frequency_gate(accelerator, model)
                if accelerator.is_main_process:
                    gate_message = ""
                    if gate_by_lead is not None:
                        gate_message = (
                            f" low_gate={gate_by_lead.mean().item():.4f} "
                            f"early={gate_by_lead[:5].mean().item():.4f} "
                            f"late={gate_by_lead[-5:].mean().item():.4f}"
                        )
                    print(
                        f"Epoch {epoch + 1}/{args.epochs}, Step {batch_index}/{len(train_loader)}, "
                        f"Global step {step} | total={logged_losses['total_loss']:.6f} "
                        f"spatial={logged_losses['spatial_loss']:.6f} "
                        f"frequency={logged_losses['frequency_loss']:.6f} lr={lr:.3e}"
                        f"{gate_message}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_metrics = {
                                "Train/total_loss": logged_losses["total_loss"],
                                "Train/spatial_loss": logged_losses["spatial_loss"],
                                "Train/frequency_loss": logged_losses["frequency_loss"],
                                "Learning_Rate": lr,
                                "Epoch": epoch + 1,
                                "Batch": batch_index,
                        }
                        if gate_by_lead is not None:
                            wandb_metrics["FrequencyGate/low_mean"] = gate_by_lead.mean().item()
                            wandb_metrics["FrequencyGate/low_early_1_5"] = gate_by_lead[:5].mean().item()
                            wandb_metrics["FrequencyGate/low_late_16_20"] = gate_by_lead[-5:].mean().item()
                            for lead_index, value in enumerate(gate_by_lead.tolist(), start=1):
                                wandb_metrics[f"FrequencyGate/lead_{lead_index:02d}"] = value
                        wandb_run.log(wandb_metrics, step=step)
        train_pbar.close()
        epoch_losses = mean_epoch_losses(accelerator, epoch_loss_sums, epoch_loss_count)

        validation = None
        if args.valid:
            validation = evaluate(
                accelerator, ema_model, vae, valid_loader, args, metrics_valid.Evaluator,
                pixel_scale, thresholds, latent_contract,
                args.vlnum if args.valid_limit else None,
                description=f"Epoch {epoch + 1}/{args.epochs} [Val]",
            )
            if accelerator.is_main_process and validation["csi"] > best_csi:
                best_csi = validation["csi"]
                save_checkpoint(
                    workdir / "checkpoints" / "best.pt", accelerator, model, ema_model,
                    optimizer, scheduler, epoch, step, best_csi, run_config,
                )
                print(f"Save Best Model: validation CSI={best_csi:.8f} at epoch {epoch + 1}", flush=True)
        save_checkpoint(
            workdir / "checkpoints" / "last.pt", accelerator, model, ema_model,
            optimizer, scheduler, epoch, step, best_csi, run_config,
        )
        if accelerator.is_main_process:
            elapsed_seconds = time.monotonic() - epoch_started
            validation_csi = "disabled" if validation is None else f"{float(validation['csi']):.8f}"
            print(f"Save Last Model: epoch {epoch + 1}", flush=True)
            print(f"========== Epoch {epoch + 1}/{args.epochs} summary ==========", flush=True)
            print(
                f"Train: time={elapsed_seconds:.1f}s total={epoch_losses['total_loss']:.6f} "
                f"spatial={epoch_losses['spatial_loss']:.6f} "
                f"frequency={epoch_losses['frequency_loss']:.6f} lr={lr:.3e}",
                flush=True,
            )
            print(f"Validation CSI: {validation_csi}", flush=True)
            print(f"Best validation CSI: {best_csi:.8f}", flush=True)
            append_epoch_metrics(
                workdir, epoch, step, epoch_losses, lr, validation, best_csi, elapsed_seconds
            )
            if wandb_run is not None:
                epoch_log = {
                    "Loss/train": epoch_losses["total_loss"],
                    "Loss/spatial": epoch_losses["spatial_loss"],
                    "Loss/frequency": epoch_losses["frequency_loss"],
                    "Learning_Rate": lr,
                    "Epoch": epoch + 1,
                    "Time/epoch_seconds": elapsed_seconds,
                }
                if validation is not None:
                    epoch_log["Metrics/Validation_CSI"] = float(validation["csi"])
                    epoch_log["Metrics/Best_Validation_CSI"] = float(best_csi)
                wandb_run.log(epoch_log, step=step)
        accelerator.wait_for_everyone()

    # AlphaPre's full evaluator is intentionally used only after training; its
    # additional LPIPS/SSIM/CRPS reporting is too expensive for every epoch.
    if args.valid and not args.skip_final_test:
        accelerator.wait_for_everyone()
        final_checkpoint_path = workdir / "checkpoints" / f"{args.final_test_checkpoint}.pt"
        if not final_checkpoint_path.is_file():
            raise FileNotFoundError(f"final test checkpoint not found: {final_checkpoint_path}")
        final_checkpoint = torch.load(
            final_checkpoint_path, map_location="cpu", weights_only=False
        )
        ema_model.load_state_dict(final_checkpoint["ema"])
        results = evaluate(
            accelerator,
            ema_model,
            vae,
            test_loader,
            args,
            metrics.Evaluator,
            pixel_scale,
            thresholds,
            latent_contract,
            description=f"Final test ({args.final_test_checkpoint}.pt EMA)",
        )
        if accelerator.is_main_process:
            print(f"AlphaPre test metrics ({args.final_test_checkpoint}.pt EMA): {results}", flush=True)
            if wandb_run is not None:
                wandb_run.log({"Metrics/Test_CSI": float(results["csi"])}, step=step)

    global_pbar.close()
    if accelerator.is_main_process:
        if wandb_run is not None:
            wandb_run.finish()
        print("FreqFlow training complete.", flush=True)


if __name__ == "__main__":
    main()
