"""Conditional FreqFlow training with vendored AlphaPre radar data/metrics."""

import argparse
import copy
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
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
def evaluate(accelerator, model, loader, args, evaluator_cls, pixel_scale, thresholds, limit=None):
    evaluator = evaluator_cls(seq_len=args.frames_out, value_scale=pixel_scale, thresholds=thresholds)
    model.eval()
    for index, batch in enumerate(loader):
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


def main():
    args = parse_args()
    if args.frames_in != 5 or args.frames_out != 20:
        raise ValueError("AlphaPre-compatible runs require --frames-in 5 --frames-out 20")
    torch.manual_seed(args.seed)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    run_config = make_run_config(args)
    workdir = Path(args.workdir)
    if accelerator.is_main_process:
        workdir.mkdir(parents=True, exist_ok=True)

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

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch in train_loader:
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
            accelerator.log({key: value.item() for key, value in losses.items()} | {"lr": scheduler.get_last_lr()[0]}, step=step)

        validation = None
        if args.valid:
            validation = evaluate(
                accelerator, ema_model, valid_loader, args, metrics_valid.Evaluator,
                pixel_scale, thresholds, args.vlnum if args.valid_limit else None,
            )
            if accelerator.is_main_process and validation["csi"] > best_csi:
                best_csi = validation["csi"]
                save_checkpoint(
                    workdir / "checkpoints" / "best.pt", accelerator, model, ema_model,
                    optimizer, scheduler, epoch, step, best_csi, run_config,
                )
        save_checkpoint(
            workdir / "checkpoints" / "last.pt", accelerator, model, ema_model,
            optimizer, scheduler, epoch, step, best_csi, run_config,
        )
        if accelerator.is_main_process:
            print(f"epoch={epoch} validation={validation} best_csi={best_csi}")

    # AlphaPre's full evaluator is intentionally used only after training; its
    # additional LPIPS/SSIM/CRPS reporting is too expensive for every epoch.
    if args.valid:
        accelerator.wait_for_everyone()
        final_checkpoint_path = workdir / "checkpoints" / f"{args.final_test_checkpoint}.pt"
        if not final_checkpoint_path.is_file():
            raise FileNotFoundError(f"final test checkpoint not found: {final_checkpoint_path}")
        final_checkpoint = torch.load(final_checkpoint_path, map_location="cpu")
        ema_model.load_state_dict(final_checkpoint["ema"])
        results = evaluate(accelerator, ema_model, test_loader, args, metrics.Evaluator, pixel_scale, thresholds)
        if accelerator.is_main_process:
            print(f"AlphaPre test metrics ({args.final_test_checkpoint}.pt EMA): {results}")


if __name__ == "__main__":
    main()
