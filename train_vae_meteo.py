"""Train the factor-four radar AutoencoderKL used before latent FreqFlow.

This is intentionally a separate stage from ``train_nowcast_alphapre.py``.
It follows FlowCast's KL-VAE, PatchGAN, and random-one-frame-per-event
sampling, while retaining the factor-four latent interface required by FreqFlow.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from alphapre_compat import metrics_valid
from vae_meteo import MeteoFrameDataset, MeteoVAEConfig, PIXEL_SCALE, RadarAutoencoder


class PatchDiscriminator(nn.Module):
    """The three-layer PatchGAN used by FlowCast's VAE loss."""

    def __init__(self, channels: int = 1, width: int = 64, layers: int = 3):
        super().__init__()
        blocks = [nn.Conv2d(channels, width, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        previous = width
        for layer in range(1, layers):
            current = min(width * (2**layer), width * 8)
            blocks += [
                nn.Conv2d(previous, current, 4, 2, 1, bias=False),
                nn.BatchNorm2d(current),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            previous = current
        current = min(width * (2**layers), width * 8)
        blocks += [
            nn.Conv2d(previous, current, 4, 1, 1, bias=False),
            nn.BatchNorm2d(current),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(current, 1, 4, 1, 1),
        ]
        self.main = nn.Sequential(*blocks)
        self.apply(self._flowcast_weights_init)

    @staticmethod
    def _flowcast_weights_init(module: nn.Module) -> None:
        """Match FlowCast's PatchGAN Conv/BatchNorm initialization."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.main(value)


class FlowCastVAELoss(nn.Module):
    """FlowCast-style pixel, KL, adaptive-GAN loss without RGB LPIPS."""

    def __init__(self, kl_weight: float, disc_weight: float):
        super().__init__()
        self.kl_weight = float(kl_weight)
        self.disc_weight = float(disc_weight)
        self.logvar = nn.Parameter(torch.zeros(()))
        self.discriminator = PatchDiscriminator()

    @staticmethod
    def kl(posterior) -> torch.Tensor:
        return -0.5 * torch.sum(
            1 + posterior.logvar - posterior.mean.square() - posterior.logvar.exp()
        ) / posterior.mean.shape[0]

    def reconstruction_terms(self, source, reconstruction, posterior):
        rec = torch.abs(source.contiguous() - reconstruction.contiguous())
        nll = rec / self.logvar.exp() + self.logvar
        return nll.sum() / source.shape[0], rec.mean(), self.kl(posterior)

    def generator_loss(self, source, reconstruction, posterior, last_layer, use_discriminator):
        nll, rec, kl = self.reconstruction_terms(source, reconstruction, posterior)
        # FlowCast and PreDiff run the PatchGAN computation during warmup too,
        # then multiply its contribution by zero.  Keeping this graph makes the
        # corresponding discriminator backward pass valid while its gradients
        # remain identically zero before activation.
        logits_fake = self.discriminator(reconstruction)
        gan = -logits_fake.mean()
        nll_grad = torch.autograd.grad(nll, last_layer, retain_graph=True)[0]
        gan_grad = torch.autograd.grad(gan, last_layer, retain_graph=True)[0]
        adaptive = (nll_grad.norm() / (gan_grad.norm() + 1.0e-4)).clamp(0.0, 1.0e4).detach()
        adaptive = adaptive * self.disc_weight
        disc_factor = 1.0 if use_discriminator else 0.0
        total = nll + self.kl_weight * kl + adaptive * disc_factor * gan
        return total, {"reconstruction": rec.detach(), "kl": kl.detach(), "gan": gan.detach(), "adaptive": adaptive.detach()}

    def discriminator_loss(self, source, reconstruction, use_discriminator):
        real = self.discriminator(source.detach())
        fake = self.discriminator(reconstruction.detach())
        hinge = 0.5 * (F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean())
        disc_factor = 1.0 if use_discriminator else 0.0
        return disc_factor * hinge

    def forward(self, mode, *args):
        """Route updates through ``forward`` so DDP synchronizes this module."""
        if mode == "generator":
            return self.generator_loss(*args)
        if mode == "discriminator":
            return self.discriminator_loss(*args)
        raise ValueError(f"unknown VAE loss mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.2)
    parser.add_argument("--warmup-min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=1.0e-3)
    parser.add_argument("--kl-weight", type=float, default=1.0e-4)
    parser.add_argument("--disc-weight", type=float, default=0.5)
    parser.add_argument("--disc-start-epoch", type=int, default=35)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--val-limit", type=int, default=0, help="0 validates every sampled event")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="freqflow-meteo-vae")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def flowcast_cosine_schedule(
    optimizer, total_steps: int, warmup_ratio: float,
    warmup_min_lr_ratio: float, min_lr_ratio: float,
):
    """FlowCast's linear warmup followed by cosine decay, stepped per update."""
    warmup_steps = int(round(warmup_ratio * total_steps))
    warmup_steps = min(max(warmup_steps, 1), max(total_steps - 1, 1))

    def warmup_multiplier(step: int) -> float:
        if step <= warmup_steps:
            return warmup_min_lr_ratio + (1.0 - warmup_min_lr_ratio) * step / warmup_steps
        return 1.0

    warmup = LambdaLR(optimizer, lr_lambda=warmup_multiplier)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=min_lr_ratio * optimizer.param_groups[0]["lr"],
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def init_wandb(args, workdir, accelerator):
    if not args.use_wandb or not accelerator.is_main_process:
        return None
    import wandb
    run_id_path = workdir / "wandb_run_id.txt"
    run_id = run_id_path.read_text().strip() if run_id_path.exists() else wandb.util.generate_id()
    if not run_id_path.exists():
        run_id_path.write_text(run_id, encoding="utf-8")
    return wandb.init(
        id=run_id, project=args.wandb_project, name=args.wandb_name or workdir.name,
        group=args.wandb_group or None, config=vars(args), dir=str(workdir), resume="allow",
    )


@torch.no_grad()
def validate(accelerator, model, criterion, loader, thresholds, limit):
    evaluator = metrics_valid.Evaluator(seq_len=1, value_scale=PIXEL_SCALE, thresholds=thresholds)
    model.eval()
    criterion.eval()
    total_l1 = total_mse = pixels = 0.0
    total_generator_loss = generator_loss_batches = 0.0
    for batch_index, (frames, _, _) in enumerate(loader):
        frames = frames.to(accelerator.device, non_blocking=True)
        reconstruction, posterior = model(frames, sample_posterior=False)
        # FlowCast selects its VAE by validation generator loss.  During eval
        # its adaptive GAN term is zero (there is no autograd graph), leaving
        # the source-faithful NLL/reconstruction plus KL objective.
        raw_criterion = accelerator.unwrap_model(criterion)
        nll, _, kl = raw_criterion.reconstruction_terms(frames, reconstruction, posterior)
        generator_loss = nll + raw_criterion.kl_weight * kl
        gathered_generator_loss = accelerator.gather(generator_loss.detach().float().reshape(1))
        gathered_source = accelerator.gather_for_metrics(frames)
        gathered_reconstruction = accelerator.gather_for_metrics(reconstruction)
        if accelerator.is_main_process:
            source_cpu = gathered_source.float().cpu()
            reconstruction_cpu = gathered_reconstruction.float().cpu().clamp(0.0, 1.0)
            evaluator.evaluate(source_cpu.unsqueeze(1), reconstruction_cpu.unsqueeze(1))
            total_l1 += F.l1_loss(reconstruction_cpu, source_cpu, reduction="sum").item()
            total_mse += F.mse_loss(reconstruction_cpu, source_cpu, reduction="sum").item()
            pixels += float(source_cpu.numel())
            total_generator_loss += gathered_generator_loss.sum().item()
            generator_loss_batches += float(gathered_generator_loss.numel())
        if limit and batch_index + 1 >= limit:
            break
    result = evaluator.done(is_main_process=True) if accelerator.is_main_process else None
    if accelerator.is_main_process:
        result.update({
            "generator_loss": total_generator_loss / max(generator_loss_batches, 1.0),
            "l1": total_l1 / max(pixels, 1.0),
            "mse": total_mse / max(pixels, 1.0),
        })
    model.train()
    criterion.train()
    return result


def save_checkpoint(path, accelerator, model, criterion, generator_optimizer, discriminator_optimizer, generator_scheduler, discriminator_scheduler, epoch, step, best_val_loss, best_csi, config):
    if not accelerator.is_main_process:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "vae": accelerator.unwrap_model(model).state_dict(),
        "loss": accelerator.unwrap_model(criterion).state_dict(),
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "generator_scheduler": generator_scheduler.state_dict(),
        "discriminator_scheduler": discriminator_scheduler.state_dict(),
        "epoch": epoch, "step": step,
        "best_val_loss": best_val_loss, "best_csi": best_csi,
        "vae_config": config.as_dict(),
    }, path)


def main():
    args = parse_args()
    if args.img_size != 128:
        raise ValueError("the initial Meteo VAE contract is fixed to 128x128")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")
    if args.disc_start_epoch < 0:
        raise ValueError("--disc-start-epoch must be non-negative")
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    set_seed(args.seed, device_specific=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = MeteoVAEConfig(image_size=args.img_size, latent_channels=args.latent_channels)
    train_dataset = MeteoFrameDataset(args.data_path, "train", args.img_size)
    val_dataset = MeteoFrameDataset(args.data_path, "val", args.img_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = RadarAutoencoder(config)
    criterion = FlowCastVAELoss(args.kl_weight, args.disc_weight)
    generator_optimizer = torch.optim.AdamW(list(model.parameters()) + [criterion.logvar], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    discriminator_optimizer = torch.optim.AdamW(criterion.discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    updates_per_epoch = math.ceil(len(train_loader) / accelerator.num_processes)
    total_steps = args.epochs * max(updates_per_epoch, 1)
    generator_scheduler = flowcast_cosine_schedule(generator_optimizer, total_steps, args.warmup_ratio, args.warmup_min_lr_ratio, args.min_lr_ratio)
    discriminator_scheduler = flowcast_cosine_schedule(discriminator_optimizer, total_steps, args.warmup_ratio, args.warmup_min_lr_ratio, args.min_lr_ratio)
    model, criterion, generator_optimizer, discriminator_optimizer, train_loader, val_loader, generator_scheduler, discriminator_scheduler = accelerator.prepare(
        model, criterion, generator_optimizer, discriminator_optimizer, train_loader, val_loader, generator_scheduler, discriminator_scheduler
    )
    start_epoch = step = 0
    best_val_loss = float("inf")
    best_csi = float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(checkpoint["vae"])
        accelerator.unwrap_model(criterion).load_state_dict(checkpoint["loss"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
        generator_scheduler.load_state_dict(checkpoint["generator_scheduler"])
        discriminator_scheduler.load_state_dict(checkpoint["discriminator_scheduler"])
        start_epoch, step = checkpoint["epoch"] + 1, checkpoint["step"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_csi = checkpoint.get("best_csi", float("-inf"))
    if accelerator.is_main_process:
        (workdir / "vae_config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
        print(f"Meteo VAE: 1x{args.img_size}x{args.img_size} -> {args.latent_channels}x32x32", flush=True)
        print(f"Train events / epoch: {len(train_dataset)}; validation events / epoch: {len(val_dataset)}", flush=True)
        print("VAE sampling: one random frame per event per epoch (FlowCast style)", flush=True)
        print(f"FlowCast schedule: epochs={args.epochs}, warmup={args.warmup_ratio:.0%}, GAN after epoch {args.disc_start_epoch}", flush=True)
        print(f"Mixed Precision: {args.mixed_precision}", flush=True)
    wandb_run = init_wandb(args, workdir, accelerator)
    thresholds = [12, 18, 24, 32]

    for epoch in range(start_epoch, args.epochs):
        model.train(); criterion.train()
        sums = {name: 0.0 for name in ("total", "reconstruction", "kl", "gan", "adaptive", "discriminator")}
        batches = 0
        progress = tqdm(train_loader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=120)
        for frames, _, _ in progress:
            frames = frames.to(accelerator.device, non_blocking=True)
            # FlowCast activates PatchGAN after completing its 35th warmup epoch.
            use_discriminator = epoch > args.disc_start_epoch
            raw_criterion = accelerator.unwrap_model(criterion)
            for parameter in raw_criterion.discriminator.parameters():
                parameter.requires_grad_(False)
            generator_optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                reconstruction, posterior = model(frames, sample_posterior=False)
                generator_loss, details = criterion("generator", frames, reconstruction, posterior, accelerator.unwrap_model(model).last_layer, use_discriminator)
            accelerator.backward(generator_loss)
            accelerator.clip_grad_norm_(list(model.parameters()) + [raw_criterion.logvar], 1.0)
            generator_optimizer.step(); generator_scheduler.step()

            for parameter in raw_criterion.discriminator.parameters():
                parameter.requires_grad_(True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                discriminator_loss = criterion("discriminator", frames, reconstruction, use_discriminator)
            accelerator.backward(discriminator_loss)
            accelerator.clip_grad_norm_(raw_criterion.discriminator.parameters(), 1.0)
            discriminator_optimizer.step(); discriminator_scheduler.step()

            values = {"total": generator_loss.detach(), "discriminator": discriminator_loss.detach(), **details}
            reduced = {key: accelerator.gather(value.float().reshape(1)).mean().item() for key, value in values.items()}
            for key, value in reduced.items(): sums[key] += value
            batches += 1; step += 1
            if accelerator.is_main_process and step % args.log_every == 0:
                current_lr = generator_optimizer.param_groups[0]["lr"]
                progress.set_postfix(loss=f"{reduced['total']:.4f}", rec=f"{reduced['reconstruction']:.5f}", lr=f"{current_lr:.2e}")
                if wandb_run is not None:
                    wandb_run.log({f"Train/{key}": value for key, value in reduced.items()} | {"Train/lr": current_lr}, step=step)

        epoch_losses = {key: value / max(batches, 1) for key, value in sums.items()}
        validation = validate(accelerator, model, criterion, val_loader, thresholds, args.val_limit)
        if accelerator.is_main_process:
            val_loss = float(validation["generator_loss"])
            csi = float(validation["csi"])
            # Keep the formal best checkpoint comparable to FlowCast's post-GAN stage.
            if epoch > args.disc_start_epoch and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(workdir / "checkpoints" / "best_val_loss.pt", accelerator, model, criterion, generator_optimizer, discriminator_optimizer, generator_scheduler, discriminator_scheduler, epoch, step, best_val_loss, best_csi, config)
            if csi > best_csi:
                best_csi = csi
                save_checkpoint(workdir / "checkpoints" / "best_csi.pt", accelerator, model, criterion, generator_optimizer, discriminator_optimizer, generator_scheduler, discriminator_scheduler, epoch, step, best_val_loss, best_csi, config)
            save_checkpoint(workdir / "checkpoints" / "last.pt", accelerator, model, criterion, generator_optimizer, discriminator_optimizer, generator_scheduler, discriminator_scheduler, epoch, step, best_val_loss, best_csi, config)
            record = {"epoch": epoch + 1, "step": step, "best_val_loss": best_val_loss, "best_csi": best_csi, "validation": validation, "training": epoch_losses, "seconds": time.time()}
            with (workdir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"VAE epoch {epoch + 1}: train={epoch_losses['total']:.6f} val_generator_loss={val_loss:.6f} best_val_loss={best_val_loss:.6f} val_csi={csi:.6f} best_csi={best_csi:.6f}", flush=True)
            if wandb_run is not None:
                wandb_run.log({f"Epoch/{key}": value for key, value in epoch_losses.items()} | {"Validation/Generator_Loss": val_loss, "Validation/Best_Generator_Loss": best_val_loss, "Validation/CSI": csi, "Validation/L1": validation["l1"], "Validation/MSE": validation["mse"], "Validation/Best_CSI": best_csi}, step=step)
        accelerator.wait_for_everyone()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
