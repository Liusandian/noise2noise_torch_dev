#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Dataset for noise model training.

Loads the benchmark dataset produced by build_dataset.py and provides
normalised (input, target) pairs for the NN noise model.

Input:  (normalised_iso, normalised_signal)   -- 2-D
Target: (noise_mean, noise_std)               -- 2-D

For the training target we treat noise_mean as 0 (after dark-frame subtraction)
and noise_std = sqrt(noise_var).  The model learns to predict these.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class NoiseProfileDataset(Dataset):
    """Dataset of (ISO, signal) -> (noise_mean, noise_std) samples."""

    # column layout in noise_dataset.npy
    COL_ISO       = 0
    COL_CHANNEL   = 1
    COL_SIGNAL    = 2
    COL_VAR       = 3
    COL_STD       = 4
    COL_BRIGHT    = 5

    def __init__(self, dataset_dir, split='train', normalize=True):
        """
        Args:
            dataset_dir: path produced by build_dataset.py
            split:       'train' or 'val'
            normalize:   normalise ISO and signal to [0, 1]
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.normalize = normalize

        # Load full data and metadata
        data_path = os.path.join(dataset_dir, 'noise_dataset.npy')
        self.all_data = np.load(data_path)

        with open(os.path.join(dataset_dir, 'metadata.json'), 'r') as f:
            self.meta = json.load(f)

        # Load split indices
        idx_file = os.path.join(dataset_dir, f'{split}_indices.npy')
        indices = np.load(idx_file)
        self.data = self.all_data[indices]

        # Compute normalisation ranges from *all* data (train+val)
        self.iso_min   = float(self.all_data[:, self.COL_ISO].min())
        self.iso_max   = float(self.all_data[:, self.COL_ISO].max())
        self.sig_min   = float(self.all_data[:, self.COL_SIGNAL].min())
        self.sig_max   = float(self.all_data[:, self.COL_SIGNAL].max())
        self.var_min   = float(self.all_data[:, self.COL_VAR].min())
        self.var_max   = float(self.all_data[:, self.COL_VAR].max())
        self.std_min   = float(self.all_data[:, self.COL_STD].min())
        self.std_max   = float(self.all_data[:, self.COL_STD].max())

        # Also expose raw ranges for de-normalisation later
        self.iso_range    = (self.iso_min, self.iso_max)
        self.signal_range = (self.sig_min, self.sig_max)
        self.std_range    = (self.std_min, self.std_max)

    # ------------------------------------------------------------------
    def _norm(self, val, vmin, vmax):
        span = vmax - vmin
        if span < 1e-12:
            return 0.0
        return (val - vmin) / span

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        iso    = row[self.COL_ISO]
        signal = row[self.COL_SIGNAL]
        n_var  = row[self.COL_VAR]
        n_std  = row[self.COL_STD]

        if self.normalize:
            iso_n = self._norm(iso, self.iso_min, self.iso_max)
            sig_n = self._norm(signal, self.sig_min, self.sig_max)
        else:
            iso_n = iso
            sig_n = signal

        # Input:  [normalised_iso, normalised_signal]
        inp = torch.tensor([iso_n, sig_n], dtype=torch.float32)
        # Target: [noise_mean=0, noise_std]  (mean ~ 0 after dark subtraction)
        tgt = torch.tensor([0.0, n_std], dtype=torch.float32)

        return inp, tgt

    # ------------------------------------------------------------------
    def get_raw_data(self):
        """Return the raw numpy data for this split (for analysis)."""
        return self.data.copy()

    def get_all_raw_data(self):
        """Return the full raw numpy data (train + val)."""
        return self.all_data.copy()


# -----------------------------------------------------------------------
# Convenience loader builder
# -----------------------------------------------------------------------

def create_noise_loaders(dataset_dir, batch_size=256, num_workers=4):
    """Create train and val DataLoaders."""
    train_ds = NoiseProfileDataset(dataset_dir, split='train')
    val_ds   = NoiseProfileDataset(dataset_dir, split='val')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader, train_ds, val_ds
