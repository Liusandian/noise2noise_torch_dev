#!/usr/bin/env python3
"""
Batch crop color chart region from RAW images.

Folder structure expected:
    rootFolder/
        ISO_xxx/
            light1/
                *.raw
            light2/
                *.raw
            ...

The color chart sits at the center of the image (center cell of a 3x3 grid).
This script crops the center 1/9 region and saves:
    - Cropped RAW (.raw)  : original Bayer data preserved (uint16 LE)
    - Visualization PNG    : full image with red crop rectangle + cropped preview

Two sensor modes are supported (auto-detected by file size):
    - DCG   (14-bit) : 4096 x 2304, stored as uint16
    - LOFIC (12-bit) : 4096 x 3968, stored as uint16

Output goes to:  <rootFolder>_cropped/  (same sub-folder structure)

Usage:
    python crop_colorchecker.py /path/to/rootFolder
    python crop_colorchecker.py /path/to/rootFolder --mode lofic
    python crop_colorchecker.py /path/to/rootFolder --crop-ratio 0.25
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ──────────────────────────────────────────────
# Sensor configurations
# ──────────────────────────────────────────────
SENSOR_CONFIGS = {
    "dcg": {"width": 4096, "height": 2304, "bit_depth": 14},
    "lofic": {"width": 4096, "height": 3968, "bit_depth": 12},
}

# Expected file sizes (uint16 = 2 bytes per pixel)
FILE_SIZE_TO_MODE = {
    4096 * 2304 * 2: "dcg",
    4096 * 3968 * 2: "lofic",
}


# ──────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────
def detect_mode_by_filesize(filepath: Path) -> str:
    """Detect sensor mode from .raw file size."""
    fsize = filepath.stat().st_size
    mode = FILE_SIZE_TO_MODE.get(fsize)
    if mode is None:
        raise ValueError(
            f"Cannot auto-detect mode for {filepath.name}: "
            f"file size {fsize} bytes does not match any known resolution.\n"
            f"  Expected: DCG={4096*2304*2} bytes, LOFIC={4096*3968*2} bytes.\n"
            f"  Use --mode to specify manually."
        )
    return mode


def detect_mode_by_folder(filepath: Path) -> str:
    """Fallback: detect mode from folder name containing 'lofic'."""
    if "lofic" in str(filepath).lower():
        return "lofic"
    return "dcg"


def read_raw(filepath: Path, width: int, height: int) -> np.ndarray:
    """Read a flat Bayer .raw file (uint16 LE) into a 2D array."""
    data = np.fromfile(str(filepath), dtype="<u2")  # little-endian uint16
    expected = width * height
    if data.size != expected:
        raise ValueError(
            f"Pixel count mismatch: file has {data.size} pixels, "
            f"expected {expected} ({width}x{height})"
        )
    return data.reshape(height, width)


def crop_center_ninth(bayer: np.ndarray, crop_ratio: float = 1.0 / 3.0):
    """
    Crop the center region of a Bayer image.

    Parameters
    ----------
    bayer : 2D ndarray  (H, W)
    crop_ratio : float
        Fraction of each dimension to crop (default 1/3 → center 1/9).

    Returns
    -------
    cropped : 2D ndarray
    crop_rect : tuple (x_start, y_start, crop_w, crop_h) in Bayer coordinates
    """
    h, w = bayer.shape
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)

    x_start = (w - crop_w) // 2
    y_start = (h - crop_h) // 2

    # Align to 2-pixel boundary to preserve BGGR Bayer pattern
    x_start = x_start - (x_start % 2)
    y_start = y_start - (y_start % 2)
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    cropped = bayer[y_start : y_start + crop_h, x_start : x_start + crop_w].copy()
    return cropped, (x_start, y_start, crop_w, crop_h)


def simple_demosaic_bggr(bayer: np.ndarray) -> np.ndarray:
    """
    Minimal 2x2-binning demosaic (BGGR) for visualization only.
    Returns float32 RGB array of shape (H/2, W/2, 3).
    """
    h, w = bayer.shape
    # Ensure even dimensions
    h2, w2 = h - h % 2, w - w % 2
    b = bayer[:h2, :w2].astype(np.float32)

    # BGGR layout:
    #   row 0: B G B G ...
    #   row 1: G R G R ...
    rgb = np.empty((h2 // 2, w2 // 2, 3), dtype=np.float32)
    rgb[:, :, 0] = b[1::2, 1::2]                                       # R
    rgb[:, :, 1] = (b[0::2, 1::2] + b[1::2, 0::2]) * 0.5              # G
    rgb[:, :, 2] = b[0::2, 0::2]                                       # B
    return rgb


def visualize_and_save(
    bayer: np.ndarray,
    crop_rect: tuple,
    bit_depth: int,
    output_png: Path,
):
    """
    Save a side-by-side PNG:
      Left  – full image (down-sampled) with red crop rectangle
      Right – cropped region preview
    """
    rgb_full = simple_demosaic_bggr(bayer)
    max_val = float((1 << bit_depth) - 1)
    rgb_full = np.clip(rgb_full / max_val, 0.0, 1.0)
    rgb_full = np.power(rgb_full, 1.0 / 2.2)  # simple gamma

    x, y, cw, ch = crop_rect

    # Crop preview (in demosaiced / half-res coordinates)
    x2, y2, cw2, ch2 = x // 2, y // 2, cw // 2, ch // 2
    rgb_crop = rgb_full[y2 : y2 + ch2, x2 : x2 + cw2]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ── Left: full image + rectangle ──
    axes[0].imshow(rgb_crop)
    axes[0].set_title(
        f"Cropped Center 1/9\n"
        f"Bayer region: x={x}, y={y}, {cw}x{ch}",
        fontsize=11,
    )
    axes[0].axis("off")

    # ── Right: cropped preview ──
    axes[1].imshow(rgb_full)
    rect = patches.Rectangle(
        (x2, y2), cw2, ch2,
        linewidth=2, edgecolor="red", facecolor="none",
    )
    axes[1].add_patch(rect)
    axes[1].set_title("Full Image  (red = crop region)", fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(str(output_png), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
# File processing
# ──────────────────────────────────────────────
def process_one(
    raw_path: Path,
    output_dir: Path,
    mode: str,
    crop_ratio: float,
):
    """Process a single .raw file: crop + save raw & png."""
    cfg = SENSOR_CONFIGS[mode]
    width, height, bit_depth = cfg["width"], cfg["height"], cfg["bit_depth"]

    bayer = read_raw(raw_path, width, height)
    cropped, crop_rect = crop_center_ninth(bayer, crop_ratio)

    stem = raw_path.stem
    out_raw = output_dir / f"{stem}_cropped.raw"
    out_png = output_dir / f"{stem}_cropped.png"

    # Save cropped Bayer data (same uint16 LE format)
    cropped.astype("<u2").tofile(str(out_raw))

    # Save visualization
    visualize_and_save(bayer, crop_rect, bit_depth, out_png)

    cx, cy, cw, ch = crop_rect
    print(f"    -> {out_raw.name}  ({cw}x{ch}, crop@({cx},{cy}))")
    print(f"    -> {out_png.name}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Batch crop color chart from center of RAW images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "root_folder",
        type=str,
        help="Root folder (contains ISO_xxx sub-folders)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "dcg", "lofic"],
        default="auto",
        help="Sensor mode. 'auto' detects from file size (default: auto)",
    )
    parser.add_argument(
        "--crop-ratio",
        type=float,
        default=1.0 / 3.0,
        help="Fraction of each dimension to crop from center "
             "(default: 0.333 → center 1/9)",
    )
    args = parser.parse_args()

    root = Path(args.root_folder).resolve()
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.")
        sys.exit(1)

    output_root = root.parent / f"{root.name}_cropped"

    # Collect all .raw files
    raw_files = sorted(root.rglob("*.raw"))
    if not raw_files:
        print(f"No .raw files found under {root}")
        sys.exit(1)

    print(f"Root folder : {root}")
    print(f"Output root : {output_root}")
    print(f"Crop ratio  : {args.crop_ratio:.4f}  "
          f"(center {args.crop_ratio**2*100:.1f}% of image)")
    print(f"RAW files   : {len(raw_files)}")
    print("=" * 60)

    success, fail = 0, 0
    for raw_path in raw_files:
        rel = raw_path.relative_to(root)
        out_dir = output_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # Detect mode
        if args.mode == "auto":
            try:
                mode = detect_mode_by_filesize(raw_path)
            except ValueError:
                mode = detect_mode_by_folder(raw_path)
                print(f"  [warn] file-size detection failed, "
                      f"falling back to folder-name detection → {mode}")
        else:
            mode = args.mode

        cfg = SENSOR_CONFIGS[mode]
        print(f"[{rel}]  mode={mode}  "
              f"{cfg['width']}x{cfg['height']}  {cfg['bit_depth']}-bit")

        try:
            process_one(raw_path, out_dir, mode, args.crop_ratio)
            success += 1
        except Exception as exc:
            print(f"    ERROR: {exc}")
            fail += 1

    print("=" * 60)
    print(f"Done. success={success}, failed={fail}")
    print(f"Results saved to: {output_root}")


if __name__ == "__main__":
    main()
