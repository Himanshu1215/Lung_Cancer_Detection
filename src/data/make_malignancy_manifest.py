"""Build a nodule-only manifest for the malignancy-proxy classification task.

LUNA16 annotations.csv provides nodule diameter (mm). This script maps each
processed nodule patch to a binary clinical-risk label using the Fleischner
Society 6 mm threshold:
  label=0  class_name="low_risk"   diameter < threshold  (routine follow-up)
  label=1  class_name="high_risk"  diameter >= threshold  (closer follow-up / PET)

Normal slices are excluded — both classes are nodule patches differing in size
and texture, making the task substantially harder than nodule-presence detection.
The resulting manifest is a drop-in replacement for manifest.csv in the
make_luna_splits / pack_dataset pipeline.

Usage:
    python src/data/make_malignancy_manifest.py \\
        --manifest     data/luna16/processed/manifest.csv \\
        --annotations  data/luna16/metadata/annotations.csv \\
        --out          data/luna16/processed/manifest_malig.csv \\
        [--threshold   6.0]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

log = logging.getLogger(__name__)


def build_nodule_diameter_lookup(annotations: pd.DataFrame) -> dict[str, float]:
    """Map '{seriesuid}_{nodule_idx:02d}' → diameter_mm.

    nodule_idx is the 0-based index of the annotation row within its scan group,
    matching the enumerate order used in lidc_prepare.process_scan().
    """
    lookup: dict[str, float] = {}
    for scan_id, grp in annotations.groupby("seriesuid"):
        for idx, (_, row) in enumerate(grp.iterrows()):
            key = f"{scan_id}_{idx:02d}"
            lookup[key] = float(row["diameter_mm"])
    return lookup


def build_malignancy_manifest(
    manifest: pd.DataFrame,
    annotations: pd.DataFrame,
    threshold_mm: float,
) -> pd.DataFrame:
    """Return a nodule-only DataFrame with diameter-based binary labels."""
    nodules = manifest[manifest["class_name"] == "nodule"].copy()
    log.info(
        "Nodule rows in source manifest: %d  (%d unique nodule_uids)",
        len(nodules), nodules["nodule_uid"].nunique(),
    )

    lookup = build_nodule_diameter_lookup(annotations)

    missing = nodules["nodule_uid"][~nodules["nodule_uid"].isin(lookup)].unique()
    if len(missing):
        log.warning(
            "%d nodule_uids not found in annotations (outside processed subsets?); dropping.",
            len(missing),
        )
        nodules = nodules[nodules["nodule_uid"].isin(lookup)]

    nodules = nodules.copy()
    nodules["diameter_mm"] = nodules["nodule_uid"].map(lookup)
    nodules["label"] = (nodules["diameter_mm"] >= threshold_mm).astype(int)
    nodules["class_name"] = nodules["label"].map({0: "low_risk", 1: "high_risk"})

    n_low = (nodules["label"] == 0).sum()
    n_high = (nodules["label"] == 1).sum()
    u_low = nodules[nodules["label"] == 0]["nodule_uid"].nunique()
    u_high = nodules[nodules["label"] == 1]["nodule_uid"].nunique()
    log.info(
        "Threshold %.1f mm → low_risk: %d slices (%d nodules)  |  high_risk: %d slices (%d nodules)",
        threshold_mm, n_low, u_low, n_high, u_high,
    )

    cols = ["filepath", "scan_id", "subset", "class_name", "label",
            "slice_z", "nodule_uid", "diameter_mm"]
    return nodules[cols].reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a malignancy-proxy manifest from existing nodule patches."
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="Path to data/luna16/processed/manifest.csv")
    p.add_argument("--annotations", type=Path, required=True,
                   help="Path to data/luna16/metadata/annotations.csv")
    p.add_argument("--out", type=Path, required=True,
                   help="Output CSV path (e.g. data/luna16/processed/manifest_malig.csv)")
    p.add_argument("--threshold", type=float, default=6.0,
                   help="Fleischner threshold in mm. Nodules >= threshold → high_risk (label=1). "
                        "Default: 6.0 mm (Fleischner Society 2017).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    manifest = pd.read_csv(args.manifest)
    annotations = pd.read_csv(args.annotations)

    malig_df = build_malignancy_manifest(manifest, annotations, args.threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    malig_df.to_csv(args.out, index=False)
    log.info("Saved malignancy manifest: %d rows → %s", len(malig_df), args.out)

    summary = (
        malig_df.groupby(["subset", "class_name"])["label"]
        .count()
        .unstack(fill_value=0)
    )
    log.info("Per-subset class distribution:\n%s", summary.to_string())


if __name__ == "__main__":
    main()
