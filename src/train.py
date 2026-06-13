"""Training script for the lung cancer leakage study.

Reads entirely from a packed .npy array (zero SFS disk I/O during training).
Writes checkpoints and metrics to SFS so they survive GPU node teardown.

Usage — smoke test (CPU, ~30s, validates end-to-end before burning A100 time):
    python src/train.py --packed-dir data/packed/protB --ckpt-dir /tmp/ckpt --smoke-test

Usage — full training run:
    python src/train.py \
        --config     configs/resnet50_protB.yaml \
        --packed-dir /mnt/sfs/lung_cancer_detection/data/packed/protB \
        --ckpt-dir   /mnt/sfs/lung_cancer_detection/data/checkpoints/resnet50_protB

Resume interrupted run:
    python src/train.py ... --resume  (auto-detects latest.pth in --ckpt-dir)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import TrainingConfig
from src.data.packed_dataset import PackedLungDataset
from src.models.build import build_model, param_groups
from src.utils.seed import set_global_seed

log = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
) -> dict[str, float]:
    """Binary classification metrics. probs is the probability for class 1."""
    from sklearn.metrics import roc_auc_score, confusion_matrix

    auroc = float(roc_auc_score(labels, probs))
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)  # recall for nodule
    specificity = tn / max(tn + fp, 1)  # recall for normal
    accuracy = (tp + tn) / max(len(labels), 1)
    return {
        "auroc": auroc,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    best_auroc: float,
    cfg: TrainingConfig,
) -> None:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_auroc": best_auroc,
        "cfg": cfg.to_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    device: torch.device,
) -> tuple[int, float]:
    """Returns (start_epoch, best_auroc)."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    log.info("Resumed from epoch %d, best AUROC=%.4f", state["epoch"], state["best_auroc"])
    return state["epoch"] + 1, state["best_auroc"]


# ── Training / validation loops ───────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: AdamW,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    smoke_test: bool,
) -> float:
    model.train()
    total_loss = 0.0
    for step, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if smoke_test and step >= 1:
            break

    return total_loss / max(step + 1, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    use_amp: bool,
    smoke_test: bool,
) -> tuple[float, dict[str, float]]:
    model.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []
    all_preds: list[int] = []
    total_loss = 0.0

    for step, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels_dev = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels_dev)

        probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        total_loss += loss.item()

        if smoke_test and step >= 1:
            break

    avg_loss = total_loss / max(step + 1, 1)
    metrics = compute_metrics(
        np.array(all_labels), np.array(all_probs), np.array(all_preds)
    )
    return avg_loss, metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train lung nodule classifier.")
    p.add_argument("--config", type=Path, help="YAML config file")
    p.add_argument("--packed-dir", type=str)
    p.add_argument("--ckpt-dir", type=str)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--protocol", choices=["A", "B"])
    p.add_argument("--smoke-test", action="store_true", help="2 iterations on CPU")
    p.add_argument("--resume", action="store_true", help="Resume from latest.pth in ckpt-dir")
    p.add_argument("--wandb-offline", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    overrides: dict[str, Any] = {
        k: v for k, v in vars(args).items()
        if v is not None and k not in ("config", "resume")
    }
    # Rename CLI keys to config field names
    if "packed_dir" in overrides:
        overrides["packed_dir"] = str(overrides.pop("packed_dir"))
    if "ckpt_dir" in overrides:
        overrides["ckpt_dir"] = str(overrides.pop("ckpt_dir"))
    if args.smoke_test:
        overrides["smoke_test"] = True
    if args.wandb_offline:
        overrides["wandb_offline"] = True

    if args.config:
        cfg = TrainingConfig.from_yaml(args.config, **overrides)
    else:
        cfg = TrainingConfig(**{k: v for k, v in overrides.items()
                                if k in TrainingConfig.__dataclass_fields__})  # type: ignore[attr-defined]

    assert cfg.packed_dir, "--packed-dir is required"
    assert cfg.ckpt_dir, "--ckpt-dir is required"

    packed_dir = Path(cfg.packed_dir)
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if cfg.smoke_test:
        device = torch.device("cpu")
        use_amp = False
        log.info("SMOKE TEST mode — running on CPU, 2 iterations per phase")
    else:
        assert torch.cuda.is_available(), "CUDA not available — use --smoke-test for CPU runs"
        device = torch.device("cuda")
        use_amp = True
        log.info("Device: %s (%s)", device, torch.cuda.get_device_name(0))

    set_global_seed(cfg.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = PackedLungDataset.from_packed_dir(packed_dir, "train", cfg.protocol)
    val_ds = PackedLungDataset.from_packed_dir(packed_dir, "val", cfg.protocol, augment_train=False)

    nw = 0 if cfg.smoke_test else cfg.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    if cfg.compile_model and not cfg.smoke_test:
        model = torch.compile(model)  # type: ignore[assignment]
        log.info("torch.compile enabled")

    # ── Loss ──────────────────────────────────────────────────────────────────
    class_weights = train_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    groups = param_groups(model, cfg)
    optimizer = AdamW(groups, weight_decay=cfg.weight_decay)
    # CosineAnnealingLR: T_max = epochs - warmup_epochs so LR reaches near-zero at end
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.epochs - cfg.warmup_epochs, 1),
        eta_min=1e-6,
    )
    scaler = GradScaler(enabled=use_amp)

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb = not cfg.smoke_test
    if use_wandb:
        try:
            import wandb
            if cfg.wandb_offline:
                import os; os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name,
                config=cfg.to_dict(),
            )
        except ImportError:
            log.warning("wandb not installed — logging disabled")
            use_wandb = False

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_auroc = 0.0
    latest_ckpt = ckpt_dir / "latest.pth"
    if args.resume and latest_ckpt.exists():
        start_epoch, best_auroc = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, device
        )

    # ── Linear warmup helper ──────────────────────────────────────────────────
    def _set_warmup_lr(epoch: int) -> None:
        if epoch < cfg.warmup_epochs:
            scale = (epoch + 1) / max(cfg.warmup_epochs, 1)
            for pg in optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", pg["lr"]) * scale

    # Store initial LRs for warmup scaling
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    # ── Training loop ─────────────────────────────────────────────────────────
    patience_counter = 0
    epochs_to_run = 2 if cfg.smoke_test else cfg.epochs
    all_metrics: list[dict] = []

    log.info("Starting training: %s  protocol=%s  device=%s", cfg.run_name, cfg.protocol, device)

    for epoch in range(start_epoch, epochs_to_run):
        t0 = time.time()
        _set_warmup_lr(epoch)

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, cfg.smoke_test
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, use_amp, cfg.smoke_test
        )

        if epoch >= cfg.warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[-1]["lr"]
        elapsed = time.time() - t0

        log.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
            "AUROC=%.4f | sens=%.3f | spec=%.3f | lr=%.2e | %.1fs",
            epoch + 1, epochs_to_run, train_loss, val_loss,
            val_metrics["auroc"], val_metrics["sensitivity"], val_metrics["specificity"],
            current_lr, elapsed,
        )

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "lr": current_lr, **{f"val_{k}": v for k, v in val_metrics.items()}}
        all_metrics.append(row)

        if use_wandb:
            wandb.log(row, step=epoch)

        # Save latest checkpoint every epoch
        save_checkpoint(latest_ckpt, epoch, model, optimizer, scheduler, best_auroc, cfg)

        # Save best checkpoint
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            save_checkpoint(ckpt_dir / "best.pth", epoch, model, optimizer, scheduler, best_auroc, cfg)
            log.info("  ↑ New best AUROC=%.4f — saved best.pth", best_auroc)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience and not cfg.smoke_test:
                log.info("Early stopping at epoch %d (patience=%d)", epoch + 1, cfg.patience)
                break

    # ── Save metrics JSON ──────────────────────────────────────────────────────
    metrics_path = ckpt_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    log.info("Metrics saved to %s", metrics_path)

    if use_wandb:
        wandb.finish()

    if cfg.smoke_test:
        log.info("Smoke test PASSED — pipeline is valid. Re-run without --smoke-test on the A100.")


if __name__ == "__main__":
    main()

# git commit -m "feat: add train.py – bf16 AMP, AdamW/cosine, early stopping on AUROC, full resume, W&B, smoke-test mode"
