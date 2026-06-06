#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build noise model training dataset from raw camera data.

Reads raw frames captured at various (ISO, brightness) combinations,
computes per-patch noise statistics (mean, variance), and saves them
as a structured numpy dataset for training the NN noise model.

Expected data directory layout
==============================
  <raw_dir>/
  +-- ISO100/           (or simply 100/)
  |   +-- 12.5/         brightness percentage
  |   |   +-- frame_0001.raw
  |   |   +-- frame_0002.raw
  |   |   +-- ...
  |   +-- 25/
  |   +-- 37.5/
  |   +-- 50/
  |   +-- 62.5/
  |   +-- 75/
  |   +-- 87.5/
  |   +-- 100/
  |   +-- dark/         dark frame (lens cap on)
  +-- ISO200/
  |   +-- ...
  +-- ISO400/
  +-- ISO800/
  +-- ISO1600/

Usage
=====
  python build_dataset.py --config ../configs/train_noise_model.yml
  python build_dataset.py --config ../configs/train_noise_model.yml \
                          --raw-dir /path/to/raw --output-dir ./my_dataset
"""

import os
import sys
import glob
import re
import json
import argparse
import numpy as np
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Raw file readers
# ---------------------------------------------------------------------------

def read_raw_plain(filepath, width, height, bit_depth=10):
    """Read a plain (unpacked) binary raw file (uint8 / uint16).

    读取未打包的二进制原始图像文件（uint8 或 uint16）。
    返回形状为 (height, width) 的 float64 数组。如果文件像素数超过
    预期，则截断到期望大小；如果不足则抛出 ValueError。
    """
    dtype = np.uint8 if bit_depth <= 8 else np.uint16
    raw = np.fromfile(filepath, dtype=dtype)
    expected = width * height
    if raw.size >= expected:
        return raw[:expected].reshape(height, width).astype(np.float64)
    raise ValueError(
        f"File {filepath}: expected >= {expected} pixels, got {raw.size}"
    )


def _unpack_mipi_raw10(data, width, height):
    """Unpack MIPI RAW10 packed format (5 bytes per 4 pixels).

    将 MIPI RAW10 打包格式（每 4 像素 5 字节）展开为每像素 10 位的整数数组。
    返回 shape 为 (height, width) 的 float64 数组。
    """
    n_pixels = width * height
    n_groups = n_pixels // 4
    expected_bytes = n_groups * 5
    data = data[:expected_bytes]

    b0 = data[0::5].astype(np.uint16)
    b1 = data[1::5].astype(np.uint16)
    b2 = data[2::5].astype(np.uint16)
    b3 = data[3::5].astype(np.uint16)
    b4 = data[4::5].astype(np.uint16)

    p0 = (b0 << 2) | ((b4 >> 0) & 0x03)
    p1 = (b1 << 2) | ((b4 >> 2) & 0x03)
    p2 = (b2 << 2) | ((b4 >> 4) & 0x03)
    p3 = (b3 << 2) | ((b4 >> 6) & 0x03)

    out = np.empty(n_groups * 4, dtype=np.uint16)
    out[0::4] = p0
    out[1::4] = p1
    out[2::4] = p2
    out[3::4] = p3
    return out[:n_pixels].reshape(height, width).astype(np.float64)


def _unpack_mipi_raw12(data, width, height):
    """Unpack MIPI RAW12 packed format (3 bytes per 2 pixels).

    将 MIPI RAW12 打包格式（每 2 像素 3 字节）展开为每像素 12 位的整数数组。
    返回 shape 为 (height, width) 的 float64 数组。
    """
    n_pixels = width * height
    n_groups = n_pixels // 2
    expected_bytes = n_groups * 3
    data = data[:expected_bytes]

    b0 = data[0::3].astype(np.uint16)
    b1 = data[1::3].astype(np.uint16)
    b2 = data[2::3].astype(np.uint16)

    p0 = (b0 << 4) | ((b2 >> 0) & 0x0F)
    p1 = (b1 << 4) | ((b2 >> 4) & 0x0F)

    out = np.empty(n_groups * 2, dtype=np.uint16)
    out[0::2] = p0
    out[1::2] = p1
    return out[:n_pixels].reshape(height, width).astype(np.float64)


def read_raw(filepath, width, height, bit_depth=10, packed=False):
    """Read a raw file with auto-detection of packing format.

    Args:
        filepath:  path to .raw file
        width:     sensor width  in pixels
        height:    sensor height in pixels
        bit_depth: bits per pixel (8, 10, 12, 14, 16)
        packed:    if True, assume MIPI packed format
    """
    """
    自动检测并读取原始 .raw 文件。

    会基于文件大小和 bit_depth 判断是未打包（plain）还是 MIPI 打包格式，
    并调用相应的解包函数。返回 float64 的图像数组，shape 为 (height, width)。
    """
    file_size = os.path.getsize(filepath)
    expected_plain = width * height * (2 if bit_depth > 8 else 1)

    if not packed and file_size >= expected_plain:
        return read_raw_plain(filepath, width, height, bit_depth)

    # Try packed formats
    raw_bytes = np.fromfile(filepath, dtype=np.uint8)

    if bit_depth == 10:
        expected_packed = (width * height * 10) // 8
        if file_size >= expected_packed:
            return _unpack_mipi_raw10(raw_bytes, width, height)

    if bit_depth == 12:
        expected_packed = (width * height * 12) // 8
        if file_size >= expected_packed:
            return _unpack_mipi_raw12(raw_bytes, width, height)

    # Fallback: try plain uint16
    return read_raw_plain(filepath, width, height, bit_depth)


# ---------------------------------------------------------------------------
# Bayer helpers
# ---------------------------------------------------------------------------

BAYER_OFFSETS = {
    'RGGB': {'R': (0, 0), 'Gr': (0, 1), 'Gb': (1, 0), 'B': (1, 1)},
    'BGGR': {'B': (0, 0), 'Gb': (0, 1), 'Gr': (1, 0), 'R': (1, 1)},
    'GRBG': {'Gr': (0, 0), 'R': (0, 1), 'B': (1, 0), 'Gb': (1, 1)},
    'GBRG': {'Gb': (0, 0), 'B': (0, 1), 'R': (1, 0), 'Gr': (1, 1)},
}

CHANNEL_NAMES = ['R', 'Gr', 'Gb', 'B']


def split_bayer(raw_img, pattern='RGGB'):
    """Split Bayer mosaic into 4 half-resolution channels.

    将 Bayer 马赛克图像分解为 4 个通道（R, Gr, Gb, B），每个通道为原图一半分辨率。
    `pattern` 指定 Bayer 排列（例如 'RGGB'）。返回字典，键为通道名。
    """
    offsets = BAYER_OFFSETS[pattern]
    channels = {}
    for name, (r_off, c_off) in offsets.items():
        channels[name] = raw_img[r_off::2, c_off::2]
    return channels


# ---------------------------------------------------------------------------
# Directory parser
# ---------------------------------------------------------------------------

def parse_data_dirs(raw_dir):
    """Parse directory tree and return {iso: {brightness: [file_list]}}.

    Supports directory names like:
        ISO100, ISO_100, 100  (ISO level)
        12.5, 25, dark, black, 黑帧  (brightness)
    """
    """
    解析给定的原始数据目录，按 ISO 和亮度层级收集 .raw 文件。

    返回结构为 {iso_level: {brightness_value: [filepaths...]}}。
    将名称为 dark/black/黑帧 等视作亮度 0.0（暗帧）。
    """
    iso_re = re.compile(r'(?:ISO)?_?(\d+)', re.IGNORECASE)
    dark_re = re.compile(r'(?:dark|black|黑帧|darkframe|bf)', re.IGNORECASE)

    result = {}
    for iso_name in sorted(os.listdir(raw_dir)):
        iso_path = os.path.join(raw_dir, iso_name)
        if not os.path.isdir(iso_path):
            continue
        m = iso_re.match(iso_name)
        if m is None:
            continue
        iso_val = int(m.group(1))
        result[iso_val] = {}

        for br_name in sorted(os.listdir(iso_path)):
            br_path = os.path.join(iso_path, br_name)
            if not os.path.isdir(br_path):
                continue
            if dark_re.match(br_name):
                brightness = 0.0
            else:
                try:
                    brightness = float(br_name)
                except ValueError:
                    continue

            raws = sorted(
                glob.glob(os.path.join(br_path, '*.raw'))
                + glob.glob(os.path.join(br_path, '*.RAW'))
                + glob.glob(os.path.join(br_path, '*.Raw'))
            )
            if raws:
                result[iso_val][brightness] = raws

    return result


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def compute_temporal_stats(raw_files, width, height, bit_depth, packed,
                           dark_frame, bayer_pattern, patch_size):
    """When >= 2 frames are available, use temporal mean/var as GT.

    使用多帧（>=2）计算每像素的时间平均值与方差，作为噪声模型的目标统计量。
    计算完成后按 Bayer 通道切分并按给定 patch_size 提取小块，返回列表：
    (channel_index, signal_mean, noise_var)。方差使用无偏估计（ddof=1）。
    """
    raws = []
    for fp in raw_files:
        img = read_raw(fp, width, height, bit_depth, packed)
        if dark_frame is not None:
            img = img - dark_frame
            img = np.clip(img, 0, 2**bit_depth - 1)
        raws.append(img)

    stacked = np.stack(raws, axis=0)                     # (N, H, W)
    temporal_mean = np.mean(stacked, axis=0)              # (H, W)
    temporal_var  = np.var(stacked, axis=0, ddof=1)       # (H, W)  unbiased

    mean_ch = split_bayer(temporal_mean, bayer_pattern)
    var_ch  = split_bayer(temporal_var, bayer_pattern)

    samples = []
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        mc = mean_ch[ch_name]
        vc = var_ch[ch_name]
        ch_h, ch_w = mc.shape
        for ri in range(0, ch_h - patch_size + 1, patch_size):
            for ci in range(0, ch_w - patch_size + 1, patch_size):
                pm = mc[ri:ri + patch_size, ci:ci + patch_size]
                pv = vc[ri:ri + patch_size, ci:ci + patch_size]
                sig = float(np.mean(pm))
                var = float(np.mean(pv))
                samples.append((ch_idx, sig, var))
    return samples


def compute_spatial_stats(raw_file, width, height, bit_depth, packed,
                          dark_frame, bayer_pattern, patch_size):
    """When only 1 frame is available, use spatial statistics.

    对单帧图像在空间上计算每个 patch 的均值和方差，作为噪声统计量近似值。
    将图像按 Bayer 通道分离后在每个通道上提取 patch，返回列表：
    (channel_index, signal_mean, noise_var)。
    """
    img = read_raw(raw_file, width, height, bit_depth, packed)
    if dark_frame is not None:
        img = img - dark_frame
        img = np.clip(img, 0, 2**bit_depth - 1)

    channels = split_bayer(img, bayer_pattern)
    samples = []
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        ch = channels[ch_name]
        ch_h, ch_w = ch.shape
        for ri in range(0, ch_h - patch_size + 1, patch_size):
            for ci in range(0, ch_w - patch_size + 1, patch_size):
                patch = ch[ri:ri + patch_size, ci:ci + patch_size]
                sig = float(np.mean(patch))
                var = float(np.var(patch))
                samples.append((ch_idx, sig, var))
    return samples


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dataset(config):
    """Build the benchmark noise dataset."""
    """
    构建噪声模型训练数据集的主流程函数。

    从配置中读取原始数据路径、图像尺寸、Bayer 模式等信息，解析目录，
    估计暗帧（dark frame），对每个 ISO/亮度组合提取噪声统计样本，
    将样本保存为 numpy 数组并写入元数据（metadata.json）。
    同时为每个 ISO 拟合一个线性基线模型 var = a * signal + b 并保存参数。
    返回 (data, meta)。
    """
    dc = config['data']
    raw_dir      = dc['raw_dir']
    output_dir   = dc['dataset_dir']
    width        = dc['raw_width']
    height       = dc['raw_height']
    bit_depth    = dc.get('bit_depth', 10)
    packed       = dc.get('packed', False)
    bayer        = dc.get('bayer_pattern', 'RGGB')
    patch_size   = dc.get('patch_size', 32)

    os.makedirs(output_dir, exist_ok=True)

    data_dict = parse_data_dirs(raw_dir)
    if not data_dict:
        print(f"[ERROR] No valid ISO/brightness directories found in {raw_dir}")
        print(f"  Expected layout:  <raw_dir>/ISO100/12.5/*.raw")
        sys.exit(1)

    print("=" * 60)
    print("Building noise model benchmark dataset")
    print("=" * 60)
    for iso_val in sorted(data_dict):
        brs = sorted(data_dict[iso_val].keys())
        counts = [len(data_dict[iso_val][b]) for b in brs]
        print(f"  ISO {iso_val:>5d} : brightness {brs}  frames {counts}")

    # ---- dark frame estimation ----
    dark_frames = {}
    for iso_val in sorted(data_dict):
        if 0.0 not in data_dict[iso_val]:
            continue
        darks = []
        for fp in data_dict[iso_val][0.0]:
            darks.append(read_raw(fp, width, height, bit_depth, packed))
        dark_mean = np.mean(darks, axis=0)
        dark_frames[iso_val] = dark_mean
        print(f"  ISO {iso_val}: dark frame black level = {np.mean(dark_mean):.2f}")

    # ---- extract noise statistics ----
    # columns: iso, channel, signal_mean, noise_var, noise_std, brightness
    all_rows = []

    for iso_val in sorted(data_dict):
        dark = dark_frames.get(iso_val)

        for brightness in sorted(data_dict[iso_val]):
            if brightness == 0.0:
                continue
            files = data_dict[iso_val][brightness]
            print(f"\n  Processing ISO {iso_val}, brightness {brightness}% "
                  f"({len(files)} frames) ...", end='')

            if len(files) >= 2:
                samples = compute_temporal_stats(
                    files, width, height, bit_depth, packed,
                    dark, bayer, patch_size)
            else:
                samples = compute_spatial_stats(
                    files[0], width, height, bit_depth, packed,
                    dark, bayer, patch_size)

            for ch_idx, sig, var in samples:
                all_rows.append([
                    float(iso_val), float(ch_idx),
                    sig, var, np.sqrt(max(var, 0.0)),
                    float(brightness),
                ])
            print(f"  {len(samples)} patches")

    data = np.array(all_rows, dtype=np.float64)
    print(f"\n{'=' * 60}")
    print(f"Total samples : {len(data)}")

    # ---- fit per-ISO linear model as baseline ----
    linear_params = {}
    for iso_val in sorted(data_dict):
        mask = data[:, 0] == iso_val
        if mask.sum() < 10:
            continue
        signals   = data[mask, 2]
        variances = data[mask, 3]
        A = np.vstack([signals, np.ones_like(signals)]).T
        (a, b), *_ = np.linalg.lstsq(A, variances, rcond=None)
        linear_params[int(iso_val)] = {'a': float(a), 'b': float(b)}
        print(f"  ISO {iso_val} linear fit:  var = {a:.6f} * signal + {b:.4f}")

    # ---- save ----
    np.save(os.path.join(output_dir, 'noise_dataset.npy'), data)

    n = len(data)
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    np.save(os.path.join(output_dir, 'train_indices.npy'), idx[:split])
    np.save(os.path.join(output_dir, 'val_indices.npy'),   idx[split:])

    meta = {
        'columns': ['iso', 'channel', 'signal_mean',
                     'noise_var', 'noise_std', 'brightness'],
        'iso_levels': sorted([int(k) for k in data_dict.keys()]),
        'brightness_levels': sorted(list({
            b for brights in data_dict.values()
            for b in brights if b > 0
        })),
        'bayer_pattern': bayer,
        'patch_size': patch_size,
        'bit_depth': bit_depth,
        'num_samples': n,
        'train_size': split,
        'val_size': n - split,
        'channel_names': CHANNEL_NAMES,
        'linear_params': linear_params,
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nDataset saved to {output_dir}")
    print(f"  train / val = {split} / {n - split}")
    print(f"  Files: noise_dataset.npy, train_indices.npy, "
          f"val_indices.npy, metadata.json")
    return data, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI 入口：解析命令行参数并调用 `build_dataset`。

    该函数读取 YAML 配置文件，可通过 `--raw-dir` 和 `--output-dir` 覆盖配置中的路径。
    """
    parser = argparse.ArgumentParser(
        description='Build noise model training dataset from raw camera data')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config (e.g. ../configs/train_noise_model.yml)')
    parser.add_argument('--raw-dir', type=str, default=None,
                        help='Override raw data directory in config')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output dataset directory in config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.raw_dir:
        config['data']['raw_dir'] = args.raw_dir
    if args.output_dir:
        config['data']['dataset_dir'] = args.output_dir

    build_dataset(config)


if __name__ == '__main__':
    main()
