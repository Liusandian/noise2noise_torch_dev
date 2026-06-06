#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noise model inference and visualisation.

Generates three-way comparison:
  1. GT (ground truth)          -- real noise statistics from captured data
  2. NN model + GT noise        -- noise synthesised by the trained NN model
  3. Linear model + GT noise    -- noise synthesised by the classical linear model

Outputs
-------
- Per-ISO noise-std vs signal curves  (GT / NN / Linear on the same plot)
- Image-level comparison: clean | GT-noisy | NN-noisy | Linear-noisy
- KL divergence table  (NN vs GT, Linear vs GT)
- All figures saved to vis_output_dir

Usage
-----
  python infer_noise.py \
      --config ../configs/train_noise_model.yml \
      --checkpoint ../ckpts/noise_model/noise_model_best.pth

  python infer_noise.py \
      --config ../configs/train_noise_model.yml \
      --checkpoint ../ckpts/noise_model/noise_model_best.pth \
      --ref-image ../data/test/example.png
"""

import os
import sys
import json
import argparse

import yaml
import numpy as np
import torch

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from noise_model import NoiseModel, LinearNoiseModel
from noise_dataset import NoiseProfileDataset


# ======================================================================
# KL divergence (same as training)
# ======================================================================

def kl_gaussian(mu_p, sigma_p, mu_q, sigma_q):
    """KL(P || Q) for univariate Gaussians."""
    sigma_p = np.maximum(sigma_p, 1e-8)
    sigma_q = np.maximum(sigma_q, 1e-8)
    return (
        np.log(sigma_q / sigma_p)
        + (sigma_p**2 + (mu_p - mu_q)**2) / (2.0 * sigma_q**2)
        - 0.5
    )


# ======================================================================
# Load trained NN model from checkpoint
# ======================================================================

def load_nn_model(ckpt_path, device='cpu'):
    """Load trained NoiseModel from a .pth checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt['config']

    model = NoiseModel(
        input_dim=cfg['model'].get('input_dim', 2),
        output_dim=cfg['model'].get('output_dim', 2),
        hidden_dims=cfg['model'].get('hidden_dims', [128, 256, 128, 64]),
        use_res_blocks=cfg['model'].get('use_res_blocks', True),
        dropout=cfg['model'].get('dropout', 0.1),  # must match training arch
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    iso_range    = ckpt.get('iso_range',    (100, 1600))
    signal_range = ckpt.get('signal_range', (0, 1023))

    return model, cfg, iso_range, signal_range


# ======================================================================
# Fit linear model from dataset
# ======================================================================

def fit_linear_model(dataset_dir):
    """Fit per-ISO linear model from the benchmark dataset."""
    data = np.load(os.path.join(dataset_dir, 'noise_dataset.npy'))
    lm = LinearNoiseModel()
    lm.fit(data[:, 0], data[:, 2], data[:, 3])  # iso, signal, variance
    return lm


# ======================================================================
# Predict noise-std curve with NN model
# ======================================================================

@torch.no_grad()
def nn_predict_curve(model, iso_val, signal_arr, iso_range, signal_range, device):
    """Predict noise std for a range of signal values at a given ISO."""
    iso_norm = (iso_val - iso_range[0]) / max(iso_range[1] - iso_range[0], 1e-8)
    sig_norm = (signal_arr - signal_range[0]) / max(signal_range[1] - signal_range[0], 1e-8)

    iso_t = np.full_like(sig_norm, iso_norm)
    inp = torch.tensor(np.stack([iso_t, sig_norm], axis=1), dtype=torch.float32).to(device)

    pred = model(inp).cpu().numpy()
    pred_mean = pred[:, 0]
    pred_std  = pred[:, 1]
    return pred_mean, pred_std


# ======================================================================
# Plot 1:  Noise-std vs Signal curves (per ISO)
# ======================================================================

def plot_noise_curves(gt_data, nn_model, linear_model,
                      iso_levels, iso_range, signal_range,
                      num_points, output_dir, device):
    """Plot noise_std vs signal for GT, NN, Linear on same axes."""
    os.makedirs(output_dir, exist_ok=True)

    sig_min, sig_max = signal_range
    signal_arr = np.linspace(sig_min, sig_max, num_points)

    n_iso = len(iso_levels)
    fig, axes = plt.subplots(1, n_iso, figsize=(5 * n_iso, 4), squeeze=False)

    for idx, iso_val in enumerate(iso_levels):
        ax = axes[0, idx]

        # --- GT scatter ---
        mask = gt_data[:, 0] == iso_val
        if mask.sum() > 0:
            gt_sig = gt_data[mask, 2]
            gt_std = gt_data[mask, 4]
            ax.scatter(gt_sig, gt_std, s=1, alpha=0.15, c='gray', label='GT data')

        # --- NN curve ---
        _, nn_std = nn_predict_curve(
            nn_model, iso_val, signal_arr, iso_range, signal_range, device)
        ax.plot(signal_arr, nn_std, 'r-', linewidth=2, label='NN model')

        # --- Linear curve ---
        lin_std = linear_model.predict_std(iso_val, signal_arr)
        ax.plot(signal_arr, lin_std, 'b--', linewidth=2, label='Linear model')

        ax.set_title(f'ISO {iso_val}')
        ax.set_xlabel('Signal level')
        ax.set_ylabel('Noise std')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Noise Std vs Signal Level', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'noise_std_curves.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # --- Also plot all ISOs on single axes ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, n_iso))
    for idx, iso_val in enumerate(iso_levels):
        _, nn_std = nn_predict_curve(
            nn_model, iso_val, signal_arr, iso_range, signal_range, device)
        lin_std = linear_model.predict_std(iso_val, signal_arr)
        ax2.plot(signal_arr, nn_std, '-',  color=colors[idx], linewidth=2,
                 label=f'NN  ISO{iso_val}')
        ax2.plot(signal_arr, lin_std, '--', color=colors[idx], linewidth=1.5,
                 label=f'Lin ISO{iso_val}')
    ax2.set_xlabel('Signal level')
    ax2.set_ylabel('Noise std')
    ax2.set_title('All ISOs: NN (solid) vs Linear (dashed)')
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'noise_std_all_iso.png')
    fig2.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {path2}")


# ======================================================================
# Plot 2:  Noise variance vs Signal (mean-variance plot)
# ======================================================================

def plot_mean_variance(gt_data, nn_model, linear_model,
                       iso_levels, iso_range, signal_range,
                       num_points, output_dir, device):
    """Plot noise variance vs signal (classical photon transfer curve)."""
    sig_min, sig_max = signal_range
    signal_arr = np.linspace(sig_min, sig_max, num_points)

    n_iso = len(iso_levels)
    fig, axes = plt.subplots(1, n_iso, figsize=(5 * n_iso, 4), squeeze=False)

    for idx, iso_val in enumerate(iso_levels):
        ax = axes[0, idx]

        # GT
        mask = gt_data[:, 0] == iso_val
        if mask.sum() > 0:
            ax.scatter(gt_data[mask, 2], gt_data[mask, 3],
                       s=1, alpha=0.15, c='gray', label='GT')

        # NN
        _, nn_std = nn_predict_curve(
            nn_model, iso_val, signal_arr, iso_range, signal_range, device)
        ax.plot(signal_arr, nn_std**2, 'r-', linewidth=2, label='NN')

        # Linear
        lin_var = linear_model.predict_variance(iso_val, signal_arr)
        ax.plot(signal_arr, lin_var, 'b--', linewidth=2, label='Linear')

        ax.set_title(f'ISO {iso_val}')
        ax.set_xlabel('Signal')
        ax.set_ylabel('Variance')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Mean-Variance (Photon Transfer)', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'mean_variance_curves.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 3:  Image-level comparison (GT vs NN-noisy vs Linear-noisy)
# ======================================================================

def _load_reference_image(ref_path):
    """Load an 8-bit reference image as float [0, 1]."""
    from PIL import Image
    img = Image.open(ref_path).convert('L')     # grayscale
    return np.array(img).astype(np.float64) / 255.0


def _make_synthetic_gradient(height=256, width=256):
    """Create a smooth 0-to-1 gradient image for demo."""
    x = np.linspace(0, 1, width)
    img = np.tile(x, (height, 1))
    return img


def _add_noise_from_model(clean_img, iso_val, nn_model, iso_range, signal_range,
                           bit_depth, device):
    """Add noise to clean image using NN model predictions."""
    max_val = 2**bit_depth - 1
    signal = clean_img * max_val  # scale to raw range

    flat_sig = signal.flatten()
    _, pred_std = nn_predict_curve(
        nn_model, iso_val, flat_sig, iso_range, signal_range, device)
    noise = np.random.randn(len(flat_sig)) * pred_std
    noisy = (flat_sig + noise).reshape(clean_img.shape)
    return np.clip(noisy / max_val, 0, 1)


def _add_noise_linear(clean_img, iso_val, linear_model, bit_depth):
    """Add noise to clean image using linear model."""
    max_val = 2**bit_depth - 1
    signal = clean_img * max_val
    flat_sig = signal.flatten()
    pred_std = linear_model.predict_std(iso_val, flat_sig)
    noise = np.random.randn(len(flat_sig)) * pred_std
    noisy = (flat_sig + noise).reshape(clean_img.shape)
    return np.clip(noisy / max_val, 0, 1)


def _add_noise_gt(clean_img, iso_val, gt_data, bit_depth):
    """Add noise to clean image using GT statistics (interpolated)."""
    max_val = 2**bit_depth - 1
    signal = clean_img * max_val

    mask = gt_data[:, 0] == iso_val
    if mask.sum() < 5:
        # fallback: find nearest ISO
        available = np.unique(gt_data[:, 0])
        iso_val = float(available[np.argmin(np.abs(available - iso_val))])
        mask = gt_data[:, 0] == iso_val

    gt_sig = gt_data[mask, 2]
    gt_std = gt_data[mask, 4]

    # Sort and interpolate
    order = np.argsort(gt_sig)
    gt_sig_sorted = gt_sig[order]
    gt_std_sorted = gt_std[order]

    flat_sig = signal.flatten()
    interp_std = np.interp(flat_sig, gt_sig_sorted, gt_std_sorted)

    noise = np.random.randn(len(flat_sig)) * interp_std
    noisy = (flat_sig + noise).reshape(clean_img.shape)
    return np.clip(noisy / max_val, 0, 1)


def plot_image_comparison(gt_data, nn_model, linear_model,
                          iso_levels, iso_range, signal_range,
                          bit_depth, ref_image_path, output_dir, device):
    """Create GT vs NN vs Linear noisy image comparison figure."""
    # Load or synthesise reference
    if ref_image_path and os.path.isfile(ref_image_path):
        clean = _load_reference_image(ref_image_path)
        print(f"  Reference image: {ref_image_path}  shape={clean.shape}")
    else:
        clean = _make_synthetic_gradient(256, 256)
        print("  Using synthetic gradient (no reference image provided)")

    n_iso = len(iso_levels)
    fig, axes = plt.subplots(n_iso, 4, figsize=(16, 4 * n_iso))
    if n_iso == 1:
        axes = axes[np.newaxis, :]

    for row, iso_val in enumerate(iso_levels):
        # Use a fixed random seed per ISO for fair comparison
        np.random.seed(42 + int(iso_val))

        noisy_gt  = _add_noise_gt(clean, iso_val, gt_data, bit_depth)
        np.random.seed(42 + int(iso_val))
        noisy_nn  = _add_noise_from_model(
            clean, iso_val, nn_model, iso_range, signal_range, bit_depth, device)
        np.random.seed(42 + int(iso_val))
        noisy_lin = _add_noise_linear(clean, iso_val, linear_model, bit_depth)

        titles = [f'Clean (ISO {iso_val})',
                  'GT noise',
                  'NN model noise',
                  'Linear model noise']
        images = [clean, noisy_gt, noisy_nn, noisy_lin]

        for col in range(4):
            ax = axes[row, col]
            ax.imshow(images[col], cmap='gray', vmin=0, vmax=1)
            ax.set_title(titles[col], fontsize=10)
            ax.axis('off')

    plt.suptitle('Image-level Noise Comparison: GT vs NN vs Linear',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'image_comparison.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 4:  Noise residual / difference visualisation
# ======================================================================

def plot_noise_residual(gt_data, nn_model, linear_model,
                        iso_levels, iso_range, signal_range,
                        bit_depth, output_dir, device):
    """Visualise the noise residual (difference between models)."""
    clean = _make_synthetic_gradient(256, 256)

    n_iso = len(iso_levels)
    fig, axes = plt.subplots(n_iso, 3, figsize=(12, 4 * n_iso))
    if n_iso == 1:
        axes = axes[np.newaxis, :]

    for row, iso_val in enumerate(iso_levels):
        np.random.seed(42 + int(iso_val))
        noisy_gt = _add_noise_gt(clean, iso_val, gt_data, bit_depth)
        np.random.seed(42 + int(iso_val))
        noisy_nn = _add_noise_from_model(
            clean, iso_val, nn_model, iso_range, signal_range, bit_depth, device)
        np.random.seed(42 + int(iso_val))
        noisy_lin = _add_noise_linear(clean, iso_val, linear_model, bit_depth)

        # Residuals
        res_gt  = noisy_gt - clean
        res_nn  = noisy_nn - clean
        res_lin = noisy_lin - clean

        vmax = max(np.abs(res_gt).max(), np.abs(res_nn).max(),
                   np.abs(res_lin).max()) * 0.8

        for col, (title, res) in enumerate([
            ('GT noise residual', res_gt),
            ('NN noise residual', res_nn),
            ('Linear noise residual', res_lin),
        ]):
            ax = axes[row, col]
            im = ax.imshow(res, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'{title} (ISO {iso_val})', fontsize=9)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Noise Residuals', fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'noise_residuals.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 5:  KL divergence heatmap
# ======================================================================

def plot_kl_table(gt_data, nn_model, linear_model,
                  iso_levels, iso_range, signal_range,
                  num_points, output_dir, device):
    """Compute and visualise per-ISO KL divergence for NN and Linear models."""
    sig_min, sig_max = signal_range
    signal_arr = np.linspace(sig_min, sig_max, num_points)

    kl_nn_list = []
    kl_lin_list = []

    print("\n  KL Divergence Summary (lower is better)")
    print("  " + "-" * 50)
    print(f"  {'ISO':>6s}  {'KL(GT||NN)':>12s}  {'KL(GT||Linear)':>14s}")
    print("  " + "-" * 50)

    for iso_val in iso_levels:
        mask = gt_data[:, 0] == iso_val
        if mask.sum() < 5:
            kl_nn_list.append(float('nan'))
            kl_lin_list.append(float('nan'))
            continue

        gt_sig = gt_data[mask, 2]
        gt_std = gt_data[mask, 4]
        gt_mean = np.zeros_like(gt_std)

        # NN predictions at GT signal levels
        _, nn_std = nn_predict_curve(
            nn_model, iso_val, gt_sig, iso_range, signal_range, device)
        nn_mean = np.zeros_like(nn_std)

        # Linear predictions at GT signal levels
        lin_std = linear_model.predict_std(iso_val, gt_sig)
        lin_mean = np.zeros_like(lin_std)

        kl_nn  = float(np.mean(kl_gaussian(gt_mean, gt_std, nn_mean, nn_std)))
        kl_lin = float(np.mean(kl_gaussian(gt_mean, gt_std, lin_mean, lin_std)))

        kl_nn_list.append(kl_nn)
        kl_lin_list.append(kl_lin)

        print(f"  {iso_val:>6d}  {kl_nn:>12.6f}  {kl_lin:>14.6f}")

    avg_nn  = np.nanmean(kl_nn_list)
    avg_lin = np.nanmean(kl_lin_list)
    print("  " + "-" * 50)
    print(f"  {'AVG':>6s}  {avg_nn:>12.6f}  {avg_lin:>14.6f}")
    print()

    # Bar chart
    x = np.arange(len(iso_levels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, kl_nn_list,  width, label='KL(GT||NN)',     color='#e74c3c')
    ax.bar(x + width/2, kl_lin_list, width, label='KL(GT||Linear)', color='#3498db')
    ax.set_xlabel('ISO')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence: NN vs Linear (lower = better)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in iso_levels])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'kl_divergence.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    return kl_nn_list, kl_lin_list


# ======================================================================
# Plot 6:  Noise histogram comparison at a specific signal level
# ======================================================================

def plot_noise_histogram(gt_data, nn_model, linear_model,
                         iso_val, target_signal, iso_range, signal_range,
                         bit_depth, output_dir, device, tol_ratio=0.1):
    """Compare noise histograms at a specific (ISO, signal) point."""
    max_val = 2 ** bit_depth - 1
    mask = gt_data[:, 0] == iso_val
    if mask.sum() == 0:
        return

    gt_sig = gt_data[mask, 2]
    gt_std = gt_data[mask, 4]

    # Find GT samples near target_signal
    tol = tol_ratio * (signal_range[1] - signal_range[0])
    near_mask = np.abs(gt_sig - target_signal) < tol
    if near_mask.sum() < 10:
        return

    gt_std_local = gt_std[near_mask]
    gt_mean_std = np.mean(gt_std_local)

    # NN prediction
    _, nn_std = nn_predict_curve(
        nn_model, iso_val, np.array([target_signal]),
        iso_range, signal_range, device)
    nn_std_val = float(nn_std[0])

    # Linear prediction
    lin_std_val = float(linear_model.predict_std(iso_val, target_signal))

    # Generate samples from each distribution
    n_samples = 50000
    gt_noise  = np.random.normal(0, gt_mean_std, n_samples)
    nn_noise  = np.random.normal(0, nn_std_val, n_samples)
    lin_noise = np.random.normal(0, lin_std_val, n_samples)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-4 * max(gt_mean_std, nn_std_val, lin_std_val),
                        4 * max(gt_mean_std, nn_std_val, lin_std_val), 100)

    ax.hist(gt_noise,  bins=bins, alpha=0.4, density=True,
            color='gray',  label=f'GT (std={gt_mean_std:.2f})')
    ax.hist(nn_noise,  bins=bins, alpha=0.4, density=True,
            color='red',   label=f'NN (std={nn_std_val:.2f})')
    ax.hist(lin_noise, bins=bins, alpha=0.4, density=True,
            color='blue',  label=f'Linear (std={lin_std_val:.2f})')

    ax.set_xlabel('Noise value')
    ax.set_ylabel('Density')
    ax.set_title(f'Noise Distribution @ ISO {iso_val}, Signal={target_signal:.0f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir,
                        f'histogram_ISO{iso_val}_sig{int(target_signal)}.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Noise model inference & visualisation')
    parser.add_argument('--config', type=str, required=True,
                        help='YAML config path')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Trained NN model checkpoint (.pth)')
    parser.add_argument('--ref-image', type=str, default=None,
                        help='Reference clean image for noise overlay')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(
        'cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # --- Load NN model ---
    print("Loading NN model ...")
    nn_model, model_cfg, iso_range, signal_range = load_nn_model(
        args.checkpoint, device)
    print(f"  ISO range:    {iso_range}")
    print(f"  Signal range: {signal_range}")

    # --- Load GT dataset ---
    dataset_dir = config['data']['dataset_dir']
    print(f"Loading GT dataset from {dataset_dir} ...")
    val_ds = NoiseProfileDataset(dataset_dir, split='val')
    gt_data = val_ds.get_all_raw_data()    # use all data for GT reference
    print(f"  Total GT samples: {len(gt_data)}")

    # --- Fit linear model ---
    print("Fitting linear noise model ...")
    linear_model = fit_linear_model(dataset_dir)
    for iso_val, (a, b) in sorted(linear_model.params.items()):
        print(f"  ISO {int(iso_val):>5d}:  var = {a:.6f} * signal + {b:.4f}")

    # --- Config ---
    ic = config.get('infer', {})
    iso_levels  = ic.get('vis_iso_levels', [100, 400, 800, 1600])
    num_points  = ic.get('signal_points', 256)
    output_dir  = ic.get('vis_output_dir', './results/noise_vis')
    bit_depth   = config['data'].get('bit_depth', 10)
    ref_image   = args.ref_image or ic.get('reference_image', '')

    os.makedirs(output_dir, exist_ok=True)

    # --- Generate all plots ---
    print("\n" + "=" * 60)
    print("Generating visualisations ...")
    print("=" * 60)

    # 1. Noise-std curves
    print("\n[1/6] Noise-std vs Signal curves ...")
    plot_noise_curves(gt_data, nn_model, linear_model,
                      iso_levels, iso_range, signal_range,
                      num_points, output_dir, device)

    # 2. Mean-variance curves
    print("\n[2/6] Mean-Variance (photon transfer) curves ...")
    plot_mean_variance(gt_data, nn_model, linear_model,
                       iso_levels, iso_range, signal_range,
                       num_points, output_dir, device)

    # 3. Image comparison
    print("\n[3/6] Image-level noise comparison ...")
    plot_image_comparison(gt_data, nn_model, linear_model,
                          iso_levels, iso_range, signal_range,
                          bit_depth, ref_image, output_dir, device)

    # 4. Noise residuals
    print("\n[4/6] Noise residual maps ...")
    plot_noise_residual(gt_data, nn_model, linear_model,
                        iso_levels, iso_range, signal_range,
                        bit_depth, output_dir, device)

    # 5. KL divergence table & bar chart
    print("\n[5/6] KL divergence comparison ...")
    kl_nn, kl_lin = plot_kl_table(
        gt_data, nn_model, linear_model,
        iso_levels, iso_range, signal_range,
        num_points, output_dir, device)

    # 6. Noise histograms at representative signal levels
    print("\n[6/6] Noise histograms ...")
    sig_min, sig_max = signal_range
    for iso_val in iso_levels:
        for frac in [0.25, 0.50, 0.75]:
            target_sig = sig_min + frac * (sig_max - sig_min)
            plot_noise_histogram(
                gt_data, nn_model, linear_model,
                iso_val, target_sig, iso_range, signal_range,
                bit_depth, output_dir, device)

    # --- Save summary JSON ---
    summary = {
        'iso_levels': iso_levels,
        'kl_nn': kl_nn,
        'kl_linear': kl_lin,
        'iso_range': list(iso_range),
        'signal_range': list(signal_range),
        'linear_params': {
            int(k): list(v) for k, v in linear_model.params.items()
        },
    }
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    print(f"\nAll results saved to: {output_dir}")
    print("Done.")


if __name__ == '__main__':
    main()
