"""PackedLungDataset: loads a pre-packed .npy array fully into RAM.

Designed for GPU VM use — zero SFS disk I/O after __init__.
The packed .npy must be built by pack_dataset.py on the CPU VM first.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# albumentations and torch are GPU-VM dependencies; not needed on CPU VM
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    Dataset = object  # type: ignore[assignment,misc]


class PackedLungDataset(Dataset):
    """Lung CT nodule/normal dataset backed by a pre-packed uint8 .npy array.

    All images are loaded into RAM in __init__. __getitem__ applies albumentations
    on-the-fly and returns a 3-channel float32 tensor.

    Normalisation contract:
        norm_stats.json stores mean/std in [0, 1] scale.
        A.Normalize(max_pixel_value=255.0) divides the uint8 input by 255, then
        subtracts mean and divides by std. Both must agree — never change one
        without the other.
    """

    def __init__(
        self,
        npy_path: Path | str,
        labels_csv: Path | str,
        norm_stats: dict[str, list[float]],
        split: str,
        protocol: str,
        other_labels_csvs: list[Path | str] | None = None,
        augment_train: bool = True,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("torch and albumentations are required for PackedLungDataset")

        npy_path = Path(npy_path)
        labels_csv = Path(labels_csv)

        self._images: np.ndarray = np.load(npy_path)  # uint8 [N, H, W]
        labels_df = pd.read_csv(labels_csv)

        assert self._images.shape[0] == len(labels_df), (
            f"Array length {self._images.shape[0]} != labels length {len(labels_df)}"
        )
        assert self._images.dtype == np.uint8, f"Expected uint8 array, got {self._images.dtype}"
        assert self._images.ndim == 3, f"Expected [N, H, W], got shape {self._images.shape}"

        self._labels: np.ndarray = labels_df["label"].to_numpy(dtype=np.int64)
        self._scan_ids: list[str] = labels_df["scan_id"].tolist()
        self._split = split
        self._protocol = protocol

        is_train = (split == "train") and augment_train
        self._transform = self._build_transform(
            mean=norm_stats["mean"],
            std=norm_stats["std"],
            is_train=is_train,
        )

        if protocol == "B" and other_labels_csvs:
            self._check_no_leakage(
                scan_ids=set(self._scan_ids),
                other_csvs=[Path(p) for p in other_labels_csvs],
                split=split,
            )

    @staticmethod
    def _build_transform(
        mean: list[float],
        std: list[float],
        is_train: bool,
    ) -> "A.Compose":
        # norm_stats are in [0,1]; max_pixel_value=255.0 makes albumentations
        # divide uint8 input by 255 before applying mean/std. Do not change
        # max_pixel_value without also changing how norm_stats.json is computed.
        normalize = A.Normalize(mean=mean, std=std, max_pixel_value=255.0)
        if is_train:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                # VerticalFlip is valid for tight nodule patches — nodules have no
                # canonical up/down orientation, unlike full axial slices.
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.GaussNoise(p=0.2),
                normalize,
                ToTensorV2(),
            ])
        return A.Compose([normalize, ToTensorV2()])

    def _check_no_leakage(
        self,
        scan_ids: set[str],
        other_csvs: list[Path],
        split: str,
    ) -> None:
        for other_csv in other_csvs:
            other_df = pd.read_csv(other_csv)
            overlap = scan_ids & set(other_df["scan_id"])
            assert not overlap, (
                f"Protocol B leakage detected: {len(overlap)} scan_id(s) in split "
                f"'{split}' also appear in {other_csv.name}. "
                f"First offenders: {sorted(overlap)[:5]}"
            )

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, idx: int) -> tuple["torch.Tensor", int]:
        img = self._images[idx]                           # uint8 [H, W]
        img_3ch = np.stack([img, img, img], axis=-1)     # uint8 [H, W, 3] (HWC for albumentations)
        transformed = self._transform(image=img_3ch)
        tensor: "torch.Tensor" = transformed["image"]    # float32 [3, H, W]
        return tensor, int(self._labels[idx])

    def class_weights(self) -> "torch.Tensor":
        """Inverse-frequency class weights: w_c = N / (n_classes * count_c)."""
        n_classes = int(self._labels.max()) + 1
        n_total = len(self._labels)
        weights = []
        for c in range(n_classes):
            count = int((self._labels == c).sum())
            weights.append(n_total / (n_classes * max(count, 1)))
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def from_packed_dir(
        packed_dir: Path | str,
        split: str,
        protocol: str,
        **kwargs: object,
    ) -> "PackedLungDataset":
        """Convenience constructor: infer all paths from packed_dir and split name."""
        import json

        packed_dir = Path(packed_dir)
        norm_stats = json.loads((packed_dir / "norm_stats.json").read_text())
        other_splits = [s for s in ("train", "val", "test") if s != split]
        other_csvs = [packed_dir / f"labels_{s}.csv" for s in other_splits]
        return PackedLungDataset(
            npy_path=packed_dir / f"{split}.npy",
            labels_csv=packed_dir / f"labels_{split}.csv",
            norm_stats=norm_stats,
            split=split,
            protocol=protocol,
            other_labels_csvs=other_csvs,
            **kwargs,
        )

# git commit -m "feat(data): add PackedLungDataset – RAM-loaded npy, albumentations pipeline, Protocol B leakage guard"
