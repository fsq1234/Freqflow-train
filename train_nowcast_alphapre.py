"""Conditional FreqFlow training with vendored AlphaPre radar data/metrics."""

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
from alphapre_compat import get_datasets, metrics, metrics_valid
from configs.freqflow_nowcast_alphapre import get_config as get_nowcast_config
from fft2d import Fourier_filter
from sde import multi_scale_targets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("sevir", "meteo"), default="sevir")
    parser.add_argument(
        "--data-path", required=True,
        help="SEVIR root (CATALOG.csv plus data/) or MeteoNet meteo_radar.h5 file",
    )
    parser.add_argument("--workdir", default="workdir/freqflow_cond_fs_sevir_5to20")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=13, help="SEVIR sequence stride; unused for MeteoNet")
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
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_freqflow(frames_in, frames_out, img_size):
    """Build the full-capacity original FreqFlow backbone for radar frames."""
    config = get_nowcast_config()
    if (img_size, frames_in, frames_out) != (config.img_size, config.frames_in, config.frames_out):
        raise ValueError(
            "the full-capacity nowcast config is fixed to 128px, 5 input frames, and 20 output frames"
        )
    model_args = copy.deepcopy(config.nnet.model_args)
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
            "img_size": config.img_size,
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


def build_loaders(dataset_name, data_path, img_size, stride, batch_size, num_workers):
    """Use AlphaPre's original reader and loader contract for each radar set."""
    get_datasets.DATAPATH[dataset_name] = data_path
    train, valid, test, _, pixel_scale, thresholds = get_datasets.get_dataset(
        data_name=dataset_name, img_size=img_size, seq_len=25,
        batch_size=batch_size, stride=stride,
    )
    if dataset_name == "sevir":
        # SEVIR already batches internally, exactly as AlphaPre does.
        train_loader = train.get_torch_dataloader(num_workers=num_workers)
        valid_loader = valid.get_torch_dataloader(num_workers=num_workers)
        test_loader = test.get_torch_dataloader(num_workers=num_workers)
    elif dataset_name == "meteo":
        # This is AlphaPre's non-SEVIR loader behavior (grad accumulation is
        # one in this entrypoint, so its train batch is batch_size directly).
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )
    else:
        raise ValueError(f"unsupported dataset: {dataset_name}")
    return (
        train_loader,
        valid_loader,
        test_loader,
        pixel_scale,
        thresholds,
    )


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

    predictions = model(flow_state, log_snr=t * 8.0 - 4.0, history=history)
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
def sample_future(model, history, frames_out, sample_steps):
    state = torch.randn(history.shape[0], frames_out, *history.shape[-2:], device=history.device, dtype=history.dtype)
    dt = 1.0 / sample_steps
    for step in range(sample_steps):
        t = torch.full((state.shape[0],), step * dt, device=state.device, dtype=state.dtype)
        velocity = model(state, log_snr=t * 8.0 - 4.0, history=history)[-1]
        state = state + dt * velocity
    return state.clamp_(0.0, 1.0)


def split_batch(batch, frames_in, frames_out, device):
    # AlphaPre returns [B, 25, 1, H, W].  FreqFlow uses time as the channel
    # axis, so the singleton VIL channel is removed after validating it.
    sequence = batch[:, : frames_in + frames_out].to(device, non_blocking=True)
    if sequence.ndim != 5 or sequence.shape[2] != 1:
        raise ValueError(f"expected AlphaPre VIL [B,T,1,H,W], got {tuple(sequence.shape)}")
    history = sequence[:, :frames_in, 0]
    future = sequence[:, frames_in : frames_in + frames_out, 0]
    return history, future


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
    loader,
    args,
    evaluator_cls,
    pixel_scale,
    thresholds,
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
    for index, batch in enumerate(progress):
        history, future = split_batch(batch, args.frames_in, args.frames_out, accelerator.device)
        prediction = sample_future(model, history, args.frames_out, args.sample_steps)
        # Restore AlphaPre's original [B, T, 1, H, W] metric interface.  The
        # singleton VIL channel is removed only inside the FreqFlow backbone.
        gathered_future = accelerator.gather_for_metrics(future).unsqueeze(2).cpu()
        gathered_prediction = accelerator.gather_for_metrics(prediction).unsqueeze(2).cpu()
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
        "best_csi": best_csi, "run_config": run_config,
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
    if args.frames_in != 5 or args.frames_out != 20:
        raise ValueError("AlphaPre-compatible runs require --frames-in 5 --frames-out 20")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1")
    torch.manual_seed(args.seed)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    run_config = make_run_config(args)
    workdir = Path(args.workdir)
    if accelerator.is_main_process:
        workdir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    wandb_run = init_wandb(args, workdir, run_config, accelerator)

    train_loader, valid_loader, test_loader, pixel_scale, thresholds = build_loaders(
        args.dataset, args.data_path, args.img_size, args.stride, args.batch_size, args.num_workers
    )
    model = build_freqflow(args.frames_in, args.frames_out, args.img_size)
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
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
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
        print(f"Train Batch Size (per process): {args.batch_size}", flush=True)
        print(f"Train Data: {len(train_loader)} batches per process", flush=True)
        print(f"Val Data: {len(valid_loader)} batches per process", flush=True)
        print(f"Test Data: {len(test_loader)} batches per process", flush=True)
        print(f"Total Epochs: {args.epochs}", flush=True)
        print("Optimizer: AdamW (betas=(0.9, 0.95))", flush=True)
        print(f"Learning Rate: {args.lr}", flush=True)
        print(f"Pixel Scale: {pixel_scale}", flush=True)
        print(f"Thresholds: {thresholds}", flush=True)
        print("Model: original full-capacity FreqFlow nowcast backbone", flush=True)
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
            history, future = split_batch(batch, args.frames_in, args.frames_out, accelerator.device)
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
                if accelerator.is_main_process:
                    print(
                        f"Epoch {epoch + 1}/{args.epochs}, Step {batch_index}/{len(train_loader)}, "
                        f"Global step {step} | total={logged_losses['total_loss']:.6f} "
                        f"spatial={logged_losses['spatial_loss']:.6f} "
                        f"frequency={logged_losses['frequency_loss']:.6f} lr={lr:.3e}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "Train/total_loss": logged_losses["total_loss"],
                                "Train/spatial_loss": logged_losses["spatial_loss"],
                                "Train/frequency_loss": logged_losses["frequency_loss"],
                                "Learning_Rate": lr,
                                "Epoch": epoch + 1,
                                "Batch": batch_index,
                            },
                            step=step,
                        )
        train_pbar.close()
        epoch_losses = mean_epoch_losses(accelerator, epoch_loss_sums, epoch_loss_count)

        validation = None
        if args.valid:
            validation = evaluate(
                accelerator, ema_model, valid_loader, args, metrics_valid.Evaluator,
                pixel_scale, thresholds, args.vlnum if args.valid_limit else None,
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
    if args.valid:
        accelerator.wait_for_everyone()
        final_checkpoint_path = workdir / "checkpoints" / f"{args.final_test_checkpoint}.pt"
        if not final_checkpoint_path.is_file():
            raise FileNotFoundError(f"final test checkpoint not found: {final_checkpoint_path}")
        final_checkpoint = torch.load(final_checkpoint_path, map_location="cpu")
        ema_model.load_state_dict(final_checkpoint["ema"])
        results = evaluate(
            accelerator,
            ema_model,
            test_loader,
            args,
            metrics.Evaluator,
            pixel_scale,
            thresholds,
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
