"""Pack split CSVs + processed PNGs into per-split uint8 .npy arrays for GPU training.

The GPU VM loads the full .npy into RAM at startup so there is zero SFS disk I/O
during training. norm_stats.json contains train-only mean/std in [0, 1] scale for
use with albumentations Normalize(max_pixel_value=255.0).

Usage:
    python src/data/pack_dataset.py \
        --splits-dir /mnt/sfs/lung_cancer_detection/data/splits \
        --out-dir    /mnt/sfs/lung_cancer_detection/data/packed/protB \
        --protocol   B \
        [--n-workers 4]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from PIL import Image
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.make_luna_splits import assert_no_scan_overlap

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
OUTPUT_SIZE = 224


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_single(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    assert img.size == (OUTPUT_SIZE, OUTPUT_SIZE), (
        f"Expected {OUTPUT_SIZE}×{OUTPUT_SIZE}, got {img.size} for {path}"
    )
    return np.array(img, dtype=np.uint8)


def load_images_to_array(
    df: pd.DataFrame,
    desc: str = "",
    n_workers: int = 4,
) -> np.ndarray:
    """Load all PNGs in df['filepath'] in order, stack into uint8 [N, H, W].

    Uses executor.map which preserves input order (unlike as_completed).
    """
    paths = df["filepath"].tolist()
    n = len(paths)

    # Memory pre-flight: N × 224 × 224 × 1 byte
    needed_bytes = n * OUTPUT_SIZE * OUTPUT_SIZE
    available = psutil.virtual_memory().available
    if needed_bytes > available * 0.8:
        log.warning(
            "Loading %d images needs ~%.1f GB; only %.1f GB available (80%% threshold).",
            n,
            needed_bytes / 1e9,
            available * 0.8 / 1e9,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        # map() preserves order — critical for label alignment
        results = list(tqdm(exe.map(_load_single, paths), total=n, desc=desc))

    array = np.stack(results, axis=0)  # [N, H, W] uint8
    assert array.shape == (n, OUTPUT_SIZE, OUTPUT_SIZE), f"Unexpected shape: {array.shape}"
    assert array.dtype == np.uint8
    return array


# ── Norm stats ────────────────────────────────────────────────────────────────

def compute_norm_stats(train_array: np.ndarray) -> dict[str, list[float]]:
    """Compute pixel-level mean and std from the training array only.

    Stats are in [0, 1] scale (divided by 255) for use with:
        A.Normalize(mean=stats['mean'], std=stats['std'], max_pixel_value=255.0)
    """
    arr_float = train_array.astype(np.float32) / 255.0
    mean = float(arr_float.mean())
    std = float(arr_float.std())
    # Grayscale repeated to 3 channels — same stats for all channels
    return {
        "mean": [mean, mean, mean],
        "std": [std, std, std],
        "n_samples": int(train_array.shape[0]),
        "pixel_range": "0-255",
        "scale": "[0,1] — use with Normalize(max_pixel_value=255.0)",
    }


# ── Saving ────────────────────────────────────────────────────────────────────

def save_split(
    array: np.ndarray,
    labels: pd.DataFrame,
    out_dir: Path,
    split_name: str,
) -> None:
    assert array.shape[0] == len(labels), (
        f"Array/label length mismatch for {split_name}: {array.shape[0]} vs {len(labels)}"
    )
    npy_path = out_dir / f"{split_name}.npy"
    csv_path = out_dir / f"labels_{split_name}.csv"
    np.save(npy_path, array)
    labels.to_csv(csv_path, index=False)
    log.info("Saved %s: shape=%s, labels=%d", split_name, array.shape, len(labels))


def verify_packed(out_dir: Path, protocol: str) -> None:
    """Post-save sanity checks: shapes, dtypes, label alignment, Protocol B overlap."""
    log.info("── Verification ──")
    dfs: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        arr = np.load(out_dir / f"{split}.npy")
        lbl = pd.read_csv(out_dir / f"labels_{split}.csv")
        assert arr.dtype == np.uint8, f"{split}.npy dtype is {arr.dtype}, expected uint8"
        assert arr.shape[1:] == (OUTPUT_SIZE, OUTPUT_SIZE), f"Unexpected shape: {arr.shape}"
        assert arr.shape[0] == len(lbl), f"{split}: array/label length mismatch"
        dist = lbl.groupby("class_name")["label"].count().to_dict()
        log.info("  %s: shape=%s  classes=%s", split, arr.shape, dist)
        dfs[split] = lbl

    if protocol == "B":
        assert_no_scan_overlap(dfs["train"], dfs["val"], dfs["test"], label="packed Protocol B")
        log.info("  Protocol B leakage guard: PASS")

    stats = json.loads((out_dir / "norm_stats.json").read_text())
    assert all(0.0 < v < 1.0 for v in stats["mean"]), "norm_stats mean not in (0,1)"
    assert all(0.0 < v < 1.0 for v in stats["std"]), "norm_stats std not in (0,1)"
    log.info("  norm_stats: mean=%s std=%s", stats["mean"], stats["std"])


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pack split CSVs into .npy arrays.")
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--protocol", choices=["A", "B"], required=True)
    p.add_argument("--n-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    proto = args.protocol

    dfs: dict[str, pd.DataFrame] = {}
    arrays: dict[str, np.ndarray] = {}

    for split in SPLITS:
        csv_path = args.splits_dir / f"prot{proto}_{split}.csv"
        assert csv_path.exists(), f"Missing split CSV: {csv_path}"
        dfs[split] = pd.read_csv(csv_path)
        arrays[split] = load_images_to_array(
            dfs[split],
            desc=f"Loading {split}",
            n_workers=args.n_workers,
        )

    # Norm stats from training array only
    stats = compute_norm_stats(arrays["train"])
    (out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("norm_stats.json written (train-only, [0,1] scale)")

    for split in SPLITS:
        save_split(arrays[split], dfs[split], out_dir, split)

    verify_packed(out_dir, proto)


if __name__ == "__main__":
    main()

# git commit -m "feat(data): add pack_dataset – order-preserving parallel PNG loading, norm_stats in [0,1] scale, post-save verification"
