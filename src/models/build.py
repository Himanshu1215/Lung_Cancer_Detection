"""timm model factory for the lung cancer leakage study.

build_model(cfg) returns an nn.Module with its classification head replaced for
NUM_CLASSES outputs. When cfg.freeze_backbone is True (linear-probe mode), all
backbone parameters are frozen and only the head is trained — used for DINOv2
and ViT feature experiments.

param_groups(model, cfg) returns two param groups so train.py can apply a lower
learning rate to the backbone than to the head (or zero LR when frozen).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import TrainingConfig
from src.data import NUM_CLASSES

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required: pip install timm") from e


def build_model(cfg: TrainingConfig) -> nn.Module:
    """Create a timm model with the head replaced for NUM_CLASSES outputs."""
    model = timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=NUM_CLASSES,
    )

    if cfg.freeze_backbone:
        _freeze_backbone(model)

    return model


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except the classification head."""
    # timm models expose the head as model.head, model.fc, model.classifier,
    # or model.head.fc depending on architecture. Freeze everything first,
    # then unfreeze whichever head attribute exists.
    for p in model.parameters():
        p.requires_grad = False

    head = (
        getattr(model, "head", None)
        or getattr(model, "fc", None)
        or getattr(model, "classifier", None)
    )
    if head is None:
        raise RuntimeError(
            f"Cannot locate classification head on {type(model).__name__}. "
            "Add it to _freeze_backbone."
        )
    for p in head.parameters():
        p.requires_grad = True


def param_groups(
    model: nn.Module,
    cfg: TrainingConfig,
) -> list[dict]:
    """Return two optimizer param groups: backbone (lr_backbone) and head (lr_head).

    When freeze_backbone=True, backbone params have requires_grad=False so the
    optimizer ignores them even if included; only the head group matters.
    """
    head_ids: set[int] = set()
    head = (
        getattr(model, "head", None)
        or getattr(model, "fc", None)
        or getattr(model, "classifier", None)
    )
    if head is not None:
        head_ids = {id(p) for p in head.parameters()}

    backbone_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    head_params = [p for p in model.parameters() if id(p) in head_ids and p.requires_grad]

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": cfg.lr_head})

    assert groups, "No trainable parameters found — check freeze_backbone logic."
    return groups
