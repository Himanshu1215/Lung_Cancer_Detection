from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Central configuration for the lung cancer leakage study.

    Only `data_root` is required; all other fields have sensible defaults.
    Use `cfg_from_root(path)` as the primary constructor.
    """

    data_root: Path

    # HU windowing
    hu_center: float = -600.0
    hu_width: float = 1500.0

    # Extraction geometry — physical crop in mm, resized to output_px
    crop_mm: float = 64.0
    output_px: int = 224
    slices_per_nodule: int = 3   # iz−1, iz, iz+1
    normal_slices_per_scan: int = 5

    # Patient-level split assignment (7-subset dataset)
    train_subsets: tuple[int, ...] = (0, 1, 2, 3, 4)
    val_subsets: tuple[int, ...] = (5,)
    test_subsets: tuple[int, ...] = (6,)

    # Reproducibility
    seed: int = 42

    # ── Path helpers ──────────────────────────────────────────────────────────

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "data" / "luna16" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "data" / "luna16" / "processed"

    @property
    def metadata_dir(self) -> Path:
        return self.data_root / "data" / "luna16" / "metadata"

    @property
    def splits_dir(self) -> Path:
        return self.data_root / "data" / "splits"

    @property
    def packed_dir(self) -> Path:
        return self.data_root / "data" / "packed"

    @property
    def annotations_csv(self) -> Path:
        return self.metadata_dir / "annotations.csv"

    @property
    def candidates_csv(self) -> Path:
        return self.metadata_dir / "candidates.csv"

    # ── Derived HU bounds ─────────────────────────────────────────────────────

    @property
    def hu_lower(self) -> float:
        return self.hu_center - self.hu_width / 2.0

    @property
    def hu_upper(self) -> float:
        return self.hu_center + self.hu_width / 2.0

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> None:
        all_subsets = (
            list(self.train_subsets)
            + list(self.val_subsets)
            + list(self.test_subsets)
        )
        if len(set(all_subsets)) != len(all_subsets):
            raise ValueError(
                f"Subset overlap between splits: train={self.train_subsets}, "
                f"val={self.val_subsets}, test={self.test_subsets}"
            )
        if self.crop_mm <= 0 or self.output_px <= 0:
            raise ValueError("crop_mm and output_px must be positive")


def cfg_from_root(data_root: str | Path, **overrides: object) -> ExperimentConfig:
    """Build config from a root path with optional field overrides."""
    root = Path(data_root)
    cfg = ExperimentConfig(data_root=root, **overrides)  # type: ignore[arg-type]
    cfg.validate()
    return cfg


# ── Training configuration ────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """All hyperparameters and paths for a single training run."""

    # Model
    model_name: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False   # linear-probe mode for DINOv2 / ViT features

    # Training
    epochs: int = 50
    batch_size: int = 128
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4      # ignored when freeze_backbone=True
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    patience: int = 10             # early stopping on val macro AUROC

    # Data
    protocol: str = "B"            # "A" (leaky) or "B" (patient-level)
    packed_dir: str = ""           # path to packed/protX/ on SFS
    ckpt_dir: str = ""             # path to save checkpoints on SFS
    num_workers: int = 8
    seed: int = 42

    # W&B
    wandb_project: str = "lung_leakage"
    wandb_offline: bool = False
    wandb_run_name: str = ""       # auto-derived if empty

    # Misc
    compile_model: bool = False    # torch.compile (PyTorch 2+)
    smoke_test: bool = False       # 2 iterations on CPU to validate pipeline

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides: Any) -> "TrainingConfig":
        """Load from a YAML file and apply CLI overrides."""
        import yaml  # pyyaml — GPU VM dependency

        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        data.update(overrides)
        # Only pass keys that are actual fields
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    @property
    def run_name(self) -> str:
        if self.wandb_run_name:
            return self.wandb_run_name
        freeze = "_probe" if self.freeze_backbone else ""
        return f"{self.model_name}{freeze}_prot{self.protocol}"
