"""Build Protocol A (leaky) and Protocol B (patient-level) split CSVs from the manifest.

Protocol B splits on whole LUNA16 subsets so no scan_id appears in more than one split.
Protocol A uses a stratified shuffle matched to Protocol B's exact per-split N and
class counts — the only variable between A and B is whether slices from the same scan
can straddle splits (the leakage condition). This makes the AUROC gap attributable
solely to leakage, not to differences in training set size or class balance.

Usage:
    python src/data/make_luna_splits.py \
        --manifest /mnt/sfs/lung_cancer_detection/data/luna16/processed/manifest.csv \
        --out-dir  /mnt/sfs/lung_cancer_detection/data/splits \
        [--train-subsets 0 1 2 3 4] \
        [--val-subsets 5] \
        [--test-subsets 6] \
        [--seed 42]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"filepath", "scan_id", "subset", "class_name", "label"}


# ── Manifest loading ──────────────────────────────────────────────────────────

def load_and_validate_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Manifest missing columns: {missing}"
    assert df[list(REQUIRED_COLUMNS)].notna().all().all(), "NaN values in required columns"
    log.info("Manifest loaded: %d rows, %d unique scans", len(df), df.scan_id.nunique())
    return df


# ── Protocol B ────────────────────────────────────────────────────────────────

def make_protocol_b_splits(
    df: pd.DataFrame,
    train_subsets: tuple[int, ...],
    val_subsets: tuple[int, ...],
    test_subsets: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition rows by subset index. Asserts zero scan_id overlap."""
    train = df[df.subset.isin(train_subsets)].reset_index(drop=True)
    val = df[df.subset.isin(val_subsets)].reset_index(drop=True)
    test = df[df.subset.isin(test_subsets)].reset_index(drop=True)
    assert_no_scan_overlap(train, val, test, label="Protocol B")
    return train, val, test


def assert_no_scan_overlap(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label: str = "",
) -> None:
    """Raise AssertionError if any scan_id appears in more than one split."""
    tv = set(train.scan_id) & set(val.scan_id)
    tt = set(train.scan_id) & set(test.scan_id)
    vt = set(val.scan_id) & set(test.scan_id)
    violations: list[str] = []
    if tv:
        violations.append(f"train∩val: {len(tv)} scan(s): {sorted(tv)[:3]}...")
    if tt:
        violations.append(f"train∩test: {len(tt)} scan(s): {sorted(tt)[:3]}...")
    if vt:
        violations.append(f"val∩test: {len(vt)} scan(s): {sorted(vt)[:3]}...")
    assert not violations, f"{label} scan_id overlap detected:\n" + "\n".join(violations)


# ── Protocol A ────────────────────────────────────────────────────────────────

def make_protocol_a_splits(
    df: pd.DataFrame,
    target_train: dict[str, int],
    target_val: dict[str, int],
    target_test: dict[str, int],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified slice-level shuffle matched to Protocol B's exact per-class counts.

    For each class:
      1. Sort rows by filepath (stable order).
      2. Shuffle with fixed seed.
      3. Assign first target_train[cls] rows to train, next target_val[cls] to val,
         remainder to test (capped at target_test[cls]).

    Result: identical N and class balance to Protocol B; only grouping differs.
    """
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for cls in sorted(df["class_name"].unique()):
        subset_df = df[df.class_name == cls].sort_values("filepath").reset_index(drop=True)
        shuffled = subset_df.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

        n_train = target_train.get(cls, 0)
        n_val = target_val.get(cls, 0)
        n_test = target_test.get(cls, 0)
        needed = n_train + n_val + n_test

        if len(shuffled) < needed:
            log.warning(
                "Class '%s': need %d rows but only %d available; capping counts.",
                cls,
                needed,
                len(shuffled),
            )
            # Scale down proportionally
            scale = len(shuffled) / needed
            n_train = round(n_train * scale)
            n_val = round(n_val * scale)
            n_test = len(shuffled) - n_train - n_val

        train_parts.append(shuffled.iloc[:n_train])
        val_parts.append(shuffled.iloc[n_train : n_train + n_val])
        test_parts.append(shuffled.iloc[n_train + n_val : n_train + n_val + n_test])

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    return train, val, test


def _count_targets(df: pd.DataFrame) -> dict[str, int]:
    return df.groupby("class_name")["label"].count().to_dict()


# ── Reporting ─────────────────────────────────────────────────────────────────

def report_split_stats(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label: str,
) -> None:
    total = len(train) + len(val) + len(test)
    log.info("── %s ──", label)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        dist = df.groupby("class_name")["label"].count().to_dict()
        log.info("  %s: %d rows, %d scans | %s", name, len(df), df.scan_id.nunique(), dist)
    log.info("  total: %d rows", total)


def report_protocol_a_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    tv = set(train.scan_id) & set(val.scan_id)
    tt = set(train.scan_id) & set(test.scan_id)
    vt = set(val.scan_id) & set(test.scan_id)
    tvt = tv & set(test.scan_id)
    n_train_scans = train.scan_id.nunique()
    log.info("── Protocol A leakage report ──")
    log.info("  scan_ids in train∩val:  %d (%.1f%% of val scans)", len(tv), 100 * len(tv) / max(1, val.scan_id.nunique()))
    log.info("  scan_ids in train∩test: %d (%.1f%% of test scans)", len(tt), 100 * len(tt) / max(1, test.scan_id.nunique()))
    log.info("  scan_ids in val∩test:   %d", len(vt))
    log.info("  scan_ids in all three:  %d", len(tvt))


# ── Save ──────────────────────────────────────────────────────────────────────

def save_splits(
    out_dir: Path,
    protocol: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"prot{protocol}_{name}.csv"
        df.to_csv(path, index=False)
        log.info("Saved %s", path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Protocol A and B split CSVs.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-subsets", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--val-subsets", type=int, nargs="+", default=[5])
    p.add_argument("--test-subsets", type=int, nargs="+", default=[6])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    df = load_and_validate_manifest(args.manifest)

    # ── Protocol B (patient-level, correct) ──────────────────────────────────
    train_b, val_b, test_b = make_protocol_b_splits(
        df,
        train_subsets=tuple(args.train_subsets),
        val_subsets=tuple(args.val_subsets),
        test_subsets=tuple(args.test_subsets),
    )
    report_split_stats(train_b, val_b, test_b, label="Protocol B")
    save_splits(args.out_dir, "B", train_b, val_b, test_b)

    # ── Protocol A (leaky, matched counts) ───────────────────────────────────
    train_a, val_a, test_a = make_protocol_a_splits(
        df,
        target_train=_count_targets(train_b),
        target_val=_count_targets(val_b),
        target_test=_count_targets(test_b),
        seed=args.seed,
    )
    report_split_stats(train_a, val_a, test_a, label="Protocol A")
    report_protocol_a_leakage(train_a, val_a, test_a)
    save_splits(args.out_dir, "A", train_a, val_a, test_a)

    # Verify A and B have matching per-split class distributions
    for split_name, df_a, df_b in [
        ("train", train_a, train_b),
        ("val", val_a, val_b),
        ("test", test_a, test_b),
    ]:
        ca = _count_targets(df_a)
        cb = _count_targets(df_b)
        assert ca == cb, (
            f"Protocol A/B {split_name} class counts differ: A={ca} B={cb}\n"
            "This would confound the leakage comparison."
        )
    log.info("Protocol A/B class counts match across all splits — leakage comparison is clean.")


if __name__ == "__main__":
    main()

# git commit -m "feat(data): add make_luna_splits – Protocol B patient-level and Protocol A stratified-matched leaky splits"
