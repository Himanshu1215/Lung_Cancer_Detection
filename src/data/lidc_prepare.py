"""LUNA16 preprocessing: HU windowing → nodule patch extraction → PNG + manifest CSV.

Usage (run once per subset, delete zip after):
    python src/data/lidc_prepare.py \
        --subset /mnt/sfs/lung_cancer_detection/data/luna16/raw/subset0 \
        --out    /mnt/sfs/lung_cancer_detection/data/luna16/processed \
        --config-root /mnt/sfs/lung_cancer_detection

After all subsets:
    python src/data/lidc_prepare.py --merge-only \
        --out /mnt/sfs/lung_cancer_detection/data/luna16/processed
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

# Allow running as a script without installing the package
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import ExperimentConfig, cfg_from_root
from src.data import LABEL_MAP

log = logging.getLogger(__name__)


# ── Types ─────────────────────────────────────────────────────────────────────

class ManifestRow(TypedDict):
    filepath: str
    scan_id: str
    subset: int
    class_name: str
    label: int
    slice_z: int
    nodule_uid: str  # '{scan_id}_{nodule_idx}' or '' for normal slices


# ── Volume loading ─────────────────────────────────────────────────────────────

def find_mhd_files(subset_dir: Path) -> list[Path]:
    """Handle both nested (subset0/subset0/*.mhd) and flat (subset3/*.mhd) layouts."""
    return sorted(subset_dir.rglob("*.mhd"))


def load_volume(mhd_path: Path) -> tuple[np.ndarray, sitk.Image]:
    """Load int16 CT volume. Returns (array [Z, Y, X], sitk image for coord transforms)."""
    img = sitk.ReadImage(str(mhd_path))
    vol = sitk.GetArrayFromImage(img)  # shape [Z, Y, X], dtype int16
    assert vol.ndim == 3, f"Expected 3D volume, got shape {vol.shape}"
    return vol, img


def apply_hu_window(
    volume: np.ndarray,
    hu_lower: float,
    hu_upper: float,
) -> np.ndarray:
    """Clip HU range and scale to uint8 [0, 255]."""
    clipped = np.clip(volume.astype(np.float32), hu_lower, hu_upper)
    scaled = (clipped - hu_lower) / (hu_upper - hu_lower) * 255.0
    return scaled.astype(np.uint8)


# ── Coordinate transforms ─────────────────────────────────────────────────────

def world_to_voxel(
    img: sitk.Image,
    coord_world: tuple[float, float, float],
) -> tuple[int, int, int]:
    """World mm (x, y, z) → voxel indices (ix, iy, iz).

    SimpleITK returns (ix, iy, iz); array indexing is vol[iz, iy, ix].
    """
    vox = img.TransformPhysicalPointToIndex(coord_world)
    return int(vox[0]), int(vox[1]), int(vox[2])


def mm_to_pixels(mm: float, spacing_xy: float) -> int:
    """Convert a physical size in mm to pixels using the scan's in-plane spacing."""
    return max(1, round(mm / spacing_xy))


# ── Patch extraction ──────────────────────────────────────────────────────────

def extract_patch(
    volume_uint8: np.ndarray,
    ix: int,
    iy: int,
    iz: int,
    crop_px: int,
    output_px: int,
) -> Image.Image | None:
    """Extract a crop_px × crop_px patch centred at (ix, iy) from axial slice vol[iz].

    vol[iz] has shape [Y, X]; crop rows with iy, columns with ix.
    Pads with 0 (window minimum = air) if the patch exceeds slice boundaries.
    Returns None if iz is out of the valid z range.
    """
    Z, Y, X = volume_uint8.shape
    if iz < 0 or iz >= Z:
        return None

    r = crop_px // 2
    axial = volume_uint8[iz]  # [Y, X]

    # Compute source rectangle, clamped to image bounds
    y0, y1 = iy - r, iy + r
    x0, x1 = ix - r, ix + r
    cy0, cy1 = max(0, y0), min(Y, y1)
    cx0, cx1 = max(0, x0), min(X, x1)

    patch = np.zeros((crop_px, crop_px), dtype=np.uint8)
    # Destination slice within the patch canvas
    dy0 = cy0 - y0
    dy1 = dy0 + (cy1 - cy0)
    dx0 = cx0 - x0
    dx1 = dx0 + (cx1 - cx0)
    patch[dy0:dy1, dx0:dx1] = axial[cy0:cy1, cx0:cx1]

    pil = Image.fromarray(patch, mode="L")
    return pil.resize((output_px, output_px), Image.BILINEAR)


# ── Per-scan processing ───────────────────────────────────────────────────────

def _scan_seed(scan_uid: str) -> int:
    """Deterministic per-scan seed derived from the scan UID. Machine-independent."""
    return int(hashlib.md5(scan_uid.encode()).hexdigest(), 16) % (2**32)


def process_scan(
    mhd_path: Path,
    subset_idx: int,
    nodule_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    out_dir: Path,
    cfg: ExperimentConfig,
) -> list[ManifestRow]:
    """Full pipeline for one scan. Returns manifest rows. Idempotent via .done sentinel."""
    scan_uid = mhd_path.stem
    out_subdir = out_dir / f"subset{subset_idx}"
    out_subdir.mkdir(parents=True, exist_ok=True)

    done_sentinel = out_subdir / f"{scan_uid}.done"
    cache_csv = out_dir / f"manifest_subset{subset_idx}.csv"

    if done_sentinel.exists():
        # Return cached rows from the per-subset manifest
        if cache_csv.exists():
            df = pd.read_csv(cache_csv)
            return df[df.scan_id == scan_uid].to_dict("records")  # type: ignore[return-value]
        return []

    vol, img = load_volume(mhd_path)
    vol_uint8 = apply_hu_window(vol, cfg.hu_lower, cfg.hu_upper)
    spacing_xy = img.GetSpacing()[0]
    crop_px = mm_to_pixels(cfg.crop_mm, spacing_xy)
    Z = vol.shape[0]

    rows: list[ManifestRow] = []

    # ── Nodule slices (nodule-bearing scans only) ──────────────────────────
    for nodule_idx, (_, nrow) in enumerate(nodule_rows.iterrows()):
        ix, iy, iz = world_to_voxel(img, (nrow.coordX, nrow.coordY, nrow.coordZ))
        # Clamp centre slice and extract iz-1, iz, iz+1
        iz_center = max(0, min(Z - 1, iz))
        for dz in range(-(cfg.slices_per_nodule // 2), cfg.slices_per_nodule // 2 + 1):
            z = max(0, min(Z - 1, iz_center + dz))
            patch = extract_patch(vol_uint8, ix, iy, z, crop_px, cfg.output_px)
            if patch is None:
                continue
            fname = f"{scan_uid}_nod{nodule_idx:02d}_z{z:04d}.png"
            fpath = out_subdir / fname
            patch.save(fpath)
            rows.append(
                ManifestRow(
                    filepath=str(fpath),
                    scan_id=scan_uid,
                    subset=subset_idx,
                    class_name="nodule",
                    label=LABEL_MAP["nodule"],
                    slice_z=z,
                    nodule_uid=f"{scan_uid}_{nodule_idx:02d}",
                )
            )

    # ── Normal slices ──────────────────────────────────────────────────────
    # Negatives are drawn from LUNA16 candidate locations with class==0 (real
    # non-nodule lung positions), so they share the positives' spatial/anatomical
    # distribution. Sampling the slice centre instead would put every negative on
    # the mediastinum/spine, making the task trivially separable (location, not
    # nodule presence). Candidates are already non-nodule, so no z-exclusion needed.
    rng = np.random.default_rng(_scan_seed(scan_uid))
    n_candidates = len(candidate_rows)
    if n_candidates > 0:
        n_sample = min(cfg.normal_slices_per_scan, n_candidates)
        sampled_idx = rng.choice(n_candidates, size=n_sample, replace=False)
        for neg_idx, ci in enumerate(sampled_idx):
            crow = candidate_rows.iloc[int(ci)]
            ix, iy, iz = world_to_voxel(img, (crow.coordX, crow.coordY, crow.coordZ))
            patch = extract_patch(vol_uint8, ix, iy, iz, crop_px, cfg.output_px)
            if patch is None:
                continue
            # neg_idx keeps the filename unique even when two candidates share a z-slice
            fname = f"{scan_uid}_norm{neg_idx:02d}_z{iz:04d}.png"
            fpath = out_subdir / fname
            patch.save(fpath)
            rows.append(
                ManifestRow(
                    filepath=str(fpath),
                    scan_id=scan_uid,
                    subset=subset_idx,
                    class_name="normal",
                    label=LABEL_MAP["normal"],
                    slice_z=int(iz),
                    nodule_uid="",
                )
            )

    done_sentinel.touch()
    return rows


# ── Manifest helpers ──────────────────────────────────────────────────────────

def append_to_manifest(rows: list[ManifestRow], csv_path: Path) -> None:
    """Append rows to the per-subset manifest CSV, creating it if needed."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False)


def merge_manifests(processed_dir: Path) -> pd.DataFrame:
    """Collect all per-subset manifest CSVs into a single merged manifest.csv."""
    parts = sorted(processed_dir.glob("manifest_subset*.csv"))
    if not parts:
        raise FileNotFoundError(f"No per-subset manifests found in {processed_dir}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    out = processed_dir / "manifest.csv"
    df.to_csv(out, index=False)
    log.info("Merged manifest: %d rows → %s", len(df), out)
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess a LUNA16 subset into PNG patches.")
    p.add_argument("--subset", type=Path, help="Path to a single subsetN directory")
    p.add_argument("--out", type=Path, required=True, help="Output directory for processed PNGs")
    p.add_argument(
        "--config-root",
        type=Path,
        default=Path("/mnt/sfs/lung_cancer_detection"),
        help="Project root (used to locate annotations.csv and config defaults)",
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="Skip extraction; only merge existing per-subset manifests into manifest.csv",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    cfg = cfg_from_root(args.config_root)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        df = merge_manifests(out_dir)
        print(df.groupby(["subset", "class_name"])["filepath"].count().to_string())
        return

    if args.subset is None:
        print("ERROR: --subset is required unless --merge-only is set.", file=sys.stderr)
        sys.exit(1)

    subset_dir: Path = args.subset
    subset_name = subset_dir.name  # e.g. 'subset3'
    assert subset_name.startswith("subset"), f"Unexpected directory name: {subset_name}"
    subset_idx = int(subset_name.replace("subset", ""))

    # Load annotations (positives) and candidates (negatives), filter to this subset
    annotations = pd.read_csv(cfg.annotations_csv)
    candidates = pd.read_csv(cfg.candidates_csv)
    candidates = candidates[candidates["class"] == 0]  # non-nodule lung locations
    mhd_files = find_mhd_files(subset_dir)
    if not mhd_files:
        log.warning("No .mhd files found in %s", subset_dir)
        return

    present_uids = {p.stem for p in mhd_files}
    annotations = annotations[annotations.seriesuid.isin(present_uids)]
    candidates = candidates[candidates.seriesuid.isin(present_uids)]
    log.info(
        "Subset %d: %d scans, %d nodule annotations, %d negative candidates",
        subset_idx,
        len(mhd_files),
        len(annotations),
        len(candidates),
    )

    per_subset_manifest = out_dir / f"manifest_subset{subset_idx}.csv"
    all_rows: list[ManifestRow] = []

    for mhd_path in tqdm(mhd_files, desc=f"subset{subset_idx}"):
        scan_uid = mhd_path.stem
        nodule_rows = annotations[annotations.seriesuid == scan_uid]
        candidate_rows = candidates[candidates.seriesuid == scan_uid]
        rows = process_scan(mhd_path, subset_idx, nodule_rows, candidate_rows, out_dir, cfg)
        all_rows.extend(rows)
        # Append rows immediately so progress survives interruption
        append_to_manifest(rows, per_subset_manifest)

    log.info(
        "Subset %d complete: %d total rows (%d nodule, %d normal)",
        subset_idx,
        len(all_rows),
        sum(1 for r in all_rows if r["class_name"] == "nodule"),
        sum(1 for r in all_rows if r["class_name"] == "normal"),
    )


if __name__ == "__main__":
    main()

# git commit -m "feat(data): add lidc_prepare – HU windowing, physical-crop patch extraction, idempotent PNG+manifest pipeline"
