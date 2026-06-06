#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the NN noise model.

Features
--------
- YAML config driven
- TensorBoard logging  (loss curves, KL divergence, noise-std curves)
- KL divergence between predicted and GT noise distributions (printed & logged)
- Python logging to file + console
- Periodic model checkpoint saving (.pth)

Usage
-----
  python train_noise_model.py --config ../configs/train_noise_model.yml
  python train_noise_model.py --config ../configs/train_noise_model.yml --cuda
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from noise_model import NoiseModel
from noise_dataset import NoiseProfileDataset, create_noise_loaders

# Optional: TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("[WARN] torch.utils.tensorboard not found. "
          "Install tensorboard for logging:  pip install tensorboard")


# ======================================================================
# KL divergence between two univariate Gaussians
# ======================================================================

def kl_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    """KL( P || Q ) where P = N(mu_p, sigma_p^2), Q = N(mu_q, sigma_q^2).

    Returns element-wise KL in nats.
    """
    sigma_p = torch.clamp(sigma_p, min=1e-8)
    sigma_q = torch.clamp(sigma_q, min=1e-8)
    return (
        torch.log(sigma_q / sigma_p)
        + (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
        - 0.5
    )


# ======================================================================
# Loss functions
# ======================================================================

class GaussianNLLLoss(nn.Module):
    """Negative log-likelihood assuming Gaussian noise.

    Given predicted (mu, sigma) and target (mu_gt, sigma_gt),
    we evaluate NLL of observing sigma_gt under N(mu, sigma^2).
    """

    def forward(self, pred, target):
        """
        pred:   (B, 2)  [predicted_mean, predicted_std]
        target: (B, 2)  [gt_mean=0,      gt_std]
        """
        pred_mean  = pred[:, 0]
        pred_std   = torch.clamp(pred[:, 1], min=1e-8)
        gt_mean    = target[:, 0]
        gt_std     = target[:, 1]

        # NLL of gt_std under N(pred_mean, pred_std^2) is not quite right;
        # instead we use a combination loss:
        #   L = MSE(pred_std, gt_std) + lambda * MSE(pred_mean, gt_mean)
        loss_std  = torch.mean((pred_std - gt_std) ** 2)
        loss_mean = torch.mean((pred_mean - gt_mean) ** 2)
        return loss_std + 0.1 * loss_mean


class CombinedLoss(nn.Module):
    """MSE on std + MSE on mean + KL regulariser."""

    def __init__(self, kl_weight=0.01):
        super().__init__()
        self.kl_weight = kl_weight

    def forward(self, pred, target):
        pred_mean = pred[:, 0]
        pred_std  = torch.clamp(pred[:, 1], min=1e-8)
        gt_mean   = target[:, 0]
        gt_std    = torch.clamp(target[:, 1], min=1e-8)

        loss_std   = torch.mean((pred_std - gt_std) ** 2)
        loss_mean  = torch.mean((pred_mean - gt_mean) ** 2)
        kl         = torch.mean(kl_gaussian(gt_mean, gt_std, pred_mean, pred_std))

        return loss_std + 0.1 * loss_mean + self.kl_weight * kl, {
            'loss_std': loss_std.item(),
            'loss_mean': loss_mean.item(),
            'kl': kl.item(),
        }


# ======================================================================
# Setup helpers
# ======================================================================

def setup_logging(log_dir, log_file='train.log'):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger('noise_model')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def build_model(cfg):
    hidden = cfg['model'].get('hidden_dims', [128, 256, 128, 64])
    inp_dim = cfg['model'].get('input_dim', 2)
    out_dim = cfg['model'].get('output_dim', 2)
    dropout = cfg['model'].get('dropout', 0.1)
    use_res = cfg['model'].get('use_res_blocks', True)
    return NoiseModel(
        input_dim=inp_dim, output_dim=out_dim,
        hidden_dims=hidden, use_res_blocks=use_res, dropout=dropout)


def build_optimizer(model, cfg):
    tc = cfg['train']
    return Adam(
        model.parameters(),
        lr=tc.get('learning_rate', 1e-3),
        weight_decay=tc.get('weight_decay', 1e-5))


def build_scheduler(optimizer, cfg):
    tc = cfg['train']
    sched_cfg = tc.get('lr_scheduler', {})
    stype = sched_cfg.get('type', 'cosine')
    if stype == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get('T_max', tc.get('num_epochs', 200)),
            eta_min=sched_cfg.get('eta_min', 1e-6))
    else:
        return ReduceLROnPlateau(
            optimizer, patience=sched_cfg.get('patience', 20),
            factor=sched_cfg.get('factor', 0.5), verbose=True)


def build_loss(cfg):
    loss_name = cfg['train'].get('loss', 'combined')
    kl_w = cfg['train'].get('kl_weight', 0.01)
    if loss_name == 'nll':
        return GaussianNLLLoss(), False
    else:
        return CombinedLoss(kl_weight=kl_w), True


# ======================================================================
# Evaluation helpers
# ======================================================================

@torch.no_grad()
def evaluate(model, loader, loss_fn, has_detail, device):
    model.eval()
    total_loss = 0.0
    total_kl = 0.0
    n = 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        pred = model(inp)
        if has_detail:
            loss, detail = loss_fn(pred, tgt)
            total_kl += detail['kl'] * inp.size(0)
        else:
            loss = loss_fn(pred, tgt)
        total_loss += loss.item() * inp.size(0)
        n += inp.size(0)
    avg_loss = total_loss / max(n, 1)
    avg_kl   = total_kl / max(n, 1)
    model.train()
    return avg_loss, avg_kl


@torch.no_grad()
def compute_dataset_kl(model, dataset, device):
    """Compute average KL divergence over the whole dataset.

    KL( GT || NN ):  GT = N(0, sigma_gt^2),  NN = N(pred_mu, pred_sigma^2)
    """
    model.eval()
    raw = dataset.get_raw_data()
    # Build normalised inputs
    iso_n = (raw[:, 0] - dataset.iso_min) / max(dataset.iso_max - dataset.iso_min, 1e-8)
    sig_n = (raw[:, 2] - dataset.sig_min) / max(dataset.sig_max - dataset.sig_min, 1e-8)
    inp = torch.tensor(np.stack([iso_n, sig_n], axis=1), dtype=torch.float32).to(device)

    pred = model(inp).cpu()
    pred_mean = pred[:, 0]
    pred_std  = pred[:, 1]

    gt_mean = torch.zeros(len(raw))
    gt_std  = torch.tensor(raw[:, 4], dtype=torch.float32)  # noise_std column

    kl_vals = kl_gaussian(gt_mean, gt_std, pred_mean, pred_std)
    model.train()

    return float(kl_vals.mean()), float(kl_vals.std())


# ======================================================================
# Training loop
# ======================================================================

def train(config, use_cuda=False):
    # --- Paths ---
    oc = config['output']
    ckpt_dir = oc.get('ckpt_dir', './ckpts/noise_model')
    log_dir  = oc.get('log_dir',  './logs/noise_model')
    tb_dir   = oc.get('tensorboard_dir', './runs/noise_model')
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = setup_logging(log_dir)
    logger.info("=" * 60)
    logger.info("Noise Model Training")
    logger.info("=" * 60)
    logger.info(f"Config:\n{yaml.dump(config, default_flow_style=False)}")

    # --- Device ---
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # --- Data ---
    tc = config['train']
    dc = config['data']
    batch_size  = tc.get('batch_size', 256)
    num_workers = tc.get('num_workers', 4)

    train_loader, val_loader, train_ds, val_ds = create_noise_loaders(
        dc['dataset_dir'], batch_size=batch_size, num_workers=num_workers)
    logger.info(f"Train samples: {len(train_ds)},  Val samples: {len(val_ds)}")
    logger.info(f"ISO range:    [{train_ds.iso_min}, {train_ds.iso_max}]")
    logger.info(f"Signal range: [{train_ds.sig_min:.1f}, {train_ds.sig_max:.1f}]")

    # --- Model / Optimizer / Scheduler / Loss ---
    model     = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    loss_fn, has_detail = build_loss(config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # --- TensorBoard ---
    writer = None
    if HAS_TB:
        writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"TensorBoard log dir: {tb_dir}")

    # --- Training params ---
    num_epochs     = tc.get('num_epochs', 200)
    log_interval   = tc.get('log_interval', 50)
    save_interval  = tc.get('save_interval', 10)
    kl_interval    = tc.get('kl_eval_interval', 5)

    # --- Main loop ---
    best_val_loss = float('inf')
    global_step = 0
    t_start = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_n = 0
        t_epoch = time.time()

        for batch_idx, (inp, tgt) in enumerate(train_loader):
            inp, tgt = inp.to(device), tgt.to(device)

            pred = model(inp)
            if has_detail:
                loss, detail = loss_fn(pred, tgt)
                epoch_kl += detail['kl'] * inp.size(0)
            else:
                loss = loss_fn(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inp.size(0)
            epoch_n += inp.size(0)
            global_step += 1

            # Periodic batch log
            if (batch_idx + 1) % log_interval == 0:
                avg = epoch_loss / epoch_n
                logger.debug(
                    f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}]  "
                    f"batch_loss={loss.item():.6f}  running_avg={avg:.6f}")

            # TensorBoard per-step
            if writer:
                writer.add_scalar('train/batch_loss', loss.item(), global_step)

        # --- End of epoch ---
        train_avg = epoch_loss / max(epoch_n, 1)
        train_kl_avg = epoch_kl / max(epoch_n, 1)

        # Validation
        val_loss, val_kl = evaluate(model, val_loader, loss_fn, has_detail, device)

        # LR scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        elapsed = time.time() - t_epoch
        logger.info(
            f"Epoch {epoch:>4d}/{num_epochs}  "
            f"train_loss={train_avg:.6f}  val_loss={val_loss:.6f}  "
            f"train_kl={train_kl_avg:.6f}  val_kl={val_kl:.6f}  "
            f"lr={current_lr:.2e}  time={elapsed:.1f}s")

        # TensorBoard epoch-level
        if writer:
            writer.add_scalar('train/epoch_loss', train_avg, epoch)
            writer.add_scalar('val/epoch_loss', val_loss, epoch)
            writer.add_scalar('train/kl_divergence', train_kl_avg, epoch)
            writer.add_scalar('val/kl_divergence', val_kl, epoch)
            writer.add_scalar('train/learning_rate', current_lr, epoch)

        # Full dataset KL divergence evaluation
        if epoch % kl_interval == 0 or epoch == num_epochs:
            kl_mean, kl_std = compute_dataset_kl(model, val_ds, device)
            logger.info(
                f"  >> Dataset KL(GT || NN):  mean={kl_mean:.6f}  std={kl_std:.6f}")
            if writer:
                writer.add_scalar('val/dataset_kl_mean', kl_mean, epoch)
                writer.add_scalar('val/dataset_kl_std', kl_std, epoch)

        # Save checkpoint
        if epoch % save_interval == 0 or epoch == num_epochs:
            ckpt_path = os.path.join(ckpt_dir, f'noise_model_epoch{epoch:04d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_avg,
                'val_loss': val_loss,
                'config': config,
                'iso_range': (train_ds.iso_min, train_ds.iso_max),
                'signal_range': (train_ds.sig_min, train_ds.sig_max),
                'std_range': (train_ds.std_min, train_ds.std_max),
            }, ckpt_path)
            logger.info(f"  >> Checkpoint saved: {ckpt_path}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(ckpt_dir, 'noise_model_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'iso_range': (train_ds.iso_min, train_ds.iso_max),
                'signal_range': (train_ds.sig_min, train_ds.sig_max),
                'std_range': (train_ds.std_min, train_ds.std_max),
            }, best_path)
            logger.info(f"  >> Best model updated (val_loss={val_loss:.6f})")

    total_time = time.time() - t_start
    logger.info(f"\nTraining complete.  Total time: {total_time:.1f}s")
    logger.info(f"Best val loss: {best_val_loss:.6f}")

    if writer:
        writer.close()

    return model


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Train NN noise model')
    parser.add_argument('--config', type=str, required=True,
                        help='YAML config path')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint .pth')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model = train(config, use_cuda=args.cuda)


if __name__ == '__main__':
    main()
