#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Network Noise Model.

Learns camera noise distribution parameters as a function of (ISO, signal_level).
Camera noise model: pixel = signal + noise
  - noise ~ N(mu, sigma)
  - Linear model:  sigma^2 = a * signal + b  (shot noise + read noise)
  - NN model:      (mu, sigma) = f_nn(ISO, signal)   (captures nonlinearities)

Architecture: MLP with residual connections and BatchNorm.
Input:  (normalized_iso, normalized_signal)  -- 2D
Output: (noise_mean, noise_std)              -- 2D
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with BatchNorm."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class NoiseModel(nn.Module):
    """MLP-based noise model.

    Predicts noise distribution parameters (mean, std) given (ISO, signal_level).
    Uses separate heads for mean and std to allow different learning dynamics.
    """

    def __init__(self, input_dim=2, output_dim=2,
                 hidden_dims=None, use_res_blocks=True, dropout=0.1):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256, 128, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim

        # --- Shared feature extractor ---
        layers = []
        prev_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if use_res_blocks and i > 0 and prev_dim == h_dim:
                layers.append(ResBlock(h_dim))
            prev_dim = h_dim

        self.features = nn.Sequential(*layers)

        # --- Mean head ---
        self.mean_head = nn.Linear(prev_dim, 1)

        # --- Std head (softplus ensures positive output) ---
        self.std_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 2) tensor  -- [normalized_iso, normalized_signal]
        Returns:
            (B, 2) tensor     -- [noise_mean, noise_std]
        """
        feat = self.features(x)
        noise_mean = self.mean_head(feat)               # (B, 1)
        noise_std = self.std_head(feat) + 1e-6           # (B, 1), always > 0
        return torch.cat([noise_mean, noise_std], dim=-1)  # (B, 2)

    def predict_noise_params(self, iso, signal,
                             iso_range=(100, 1600),
                             signal_range=(0, 1023)):
        """Convenience method: raw ISO + signal values -> noise params.

        Args:
            iso:    scalar or 1-D tensor / ndarray
            signal: scalar or 1-D tensor / ndarray
        Returns:
            (noise_mean, noise_std) as numpy arrays
        """
        import numpy as np
        iso_arr = np.atleast_1d(np.asarray(iso, dtype=np.float32))
        sig_arr = np.atleast_1d(np.asarray(signal, dtype=np.float32))

        # Normalize
        iso_norm = (iso_arr - iso_range[0]) / (iso_range[1] - iso_range[0])
        sig_norm = (sig_arr - signal_range[0]) / (signal_range[1] - signal_range[0])

        inp = torch.from_numpy(
            np.stack([iso_norm, sig_norm], axis=-1)
        ).float()

        self.eval()
        with torch.no_grad():
            out = self.forward(inp).cpu().numpy()

        return out[:, 0], out[:, 1]   # mean, std


class LinearNoiseModel:
    """Classical linear noise model: var = a * signal + b  (per ISO).

    This is the Poisson-Gaussian noise model baseline.
    """

    def __init__(self):
        self.params = {}   # {iso: (a, b)}

    def fit(self, iso_arr, signal_arr, var_arr):
        """Fit linear model per ISO level."""
        import numpy as np
        unique_isos = np.unique(iso_arr)
        for iso_val in unique_isos:
            mask = iso_arr == iso_val
            signals = signal_arr[mask]
            variances = var_arr[mask]
            if len(signals) < 2:
                continue
            A = np.vstack([signals, np.ones_like(signals)]).T
            result = np.linalg.lstsq(A, variances, rcond=None)
            a, b = result[0]
            self.params[float(iso_val)] = (float(a), float(b))

    def predict_variance(self, iso, signal):
        """Predict noise variance."""
        import numpy as np
        iso = float(iso)
        if iso not in self.params:
            # Find nearest ISO
            available = sorted(self.params.keys())
            iso = min(available, key=lambda x: abs(x - iso))
        a, b = self.params[iso]
        return np.maximum(a * np.asarray(signal) + b, 0.0)

    def predict_std(self, iso, signal):
        import numpy as np
        return np.sqrt(self.predict_variance(iso, signal))

    def save(self, filepath):
        import json
        with open(filepath, 'w') as f:
            json.dump(self.params, f, indent=2)

    def load(self, filepath):
        import json
        with open(filepath, 'r') as f:
            raw = json.load(f)
        self.params = {float(k): tuple(v) for k, v in raw.items()}
