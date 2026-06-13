# Lung Cancer Detection — A Data-Leakage Study on LUNA16

Does **data leakage** make medical-imaging models *look* better than they really are?
This project measures it directly: we train the same lung-nodule detector under two data
splits and compare the gap.

- **Protocol A (leaky)** — slice-level split: slices from one patient's scan can land in both
  train and test.
- **Protocol B (clean)** — patient-level split: every patient is wholly in train *or* val *or* test.

The **A − B AUROC gap** is the cost of leakage.

## Headline result

| Model | Protocol | Best val AUROC | Sensitivity | Specificity |
|---|---|---|---|---|
| resnet50 | **A (leaky)** | **0.9795** | 0.901 | 0.962 |
| resnet50 | **B (clean)** | **0.9511** | 0.802 | 0.966 |

Leakage inflates apparent AUROC by **~0.028** and sensitivity by **~0.10**. The clean model
learns gradually to ~0.95 (not the suspicious 1.0 of an earlier buggy run — see below), so the
gap is a meaningful leakage estimate rather than an artifact.

> **Note on a bug that mattered.** An earlier version cropped "normal" patches at the slice
> center (the mediastinum) while nodule patches sat in the peripheral lung — so the model learned
> *location*, not *nodules*, and hit AUROC 1.0 under **both** protocols, masking the leakage
> signal. The fix: negatives are now sampled from LUNA16 `candidates.csv` (real non-nodule lung
> locations), matching the positives' spatial distribution. A trivial mean-brightness classifier
> dropped from ~1.0 to **0.54** AUROC after the fix, confirming the shortcut is gone. The full
> story is in `THESIS.md` §4 — it's a clean teaching case of preprocessing-induced leakage.

## Documentation

| File | What it is |
|---|---|
| [`THESIS.md`](THESIS.md) | Full thesis-style writeup — biology, methods, results, readable kid→researcher |
| [`GPU_TRAINING_HANDOFF.md`](GPU_TRAINING_HANDOFF.md) | How to train on the GPU node (env, commands, run matrix) |
| [`READING_LIST.md`](READING_LIST.md) | Annotated bibliography / suggested reading |
| [`CONTEXT_2026_06_11.md`](CONTEXT_2026_06_11.md) | Session log / project status snapshot |

## Repository layout

```
configs/                YAML training configs (resnet50, convnext, vit, dinov2 × protocol)
src/
  config.py             ExperimentConfig + TrainingConfig
  data/
    lidc_prepare.py     LUNA16 → HU-windowed 224×224 PNG patches + manifest
    make_luna_splits.py Protocol A (leaky) & B (clean) split CSVs
    pack_dataset.py     PNGs → packed .npy arrays + train-only norm stats
    packed_dataset.py   PyTorch Dataset over the packed arrays
  models/build.py       timm backbone factory + parameter groups
  train.py              training loop (bf16 AMP, AdamW/cosine, early stop on AUROC)
data/                   raw + processed + packed data, splits, checkpoints  (git-ignored)
```

## Quickstart

This pipeline runs in two environments: **preprocessing on a CPU box**, **training on a GPU box**
(both share the `/mnt/sfs` network filesystem).

```bash
# ── Preprocessing (CPU) ───────────────────────────────────────────────
pip install -r requirements.txt   # SimpleITK, Pillow, pandas, numpy, tqdm, psutil ...
PY=python

# Extract patches for subsets 0–6, then merge into one manifest
for i in 0 1 2 3 4 5 6; do
  $PY src/data/lidc_prepare.py --subset data/luna16/raw/subset$i \
      --out data/luna16/processed --config-root "$PWD"
done
$PY src/data/lidc_prepare.py --merge-only --out data/luna16/processed

# Build splits, then pack each protocol into .npy
$PY src/data/make_luna_splits.py --manifest data/luna16/processed/manifest.csv --out-dir data/splits
$PY src/data/pack_dataset.py --splits-dir data/splits --out-dir data/packed/protB --protocol B
$PY src/data/pack_dataset.py --splits-dir data/splits --out-dir data/packed/protA --protocol A

# ── Training (GPU) ────────────────────────────────────────────────────
# Validate the pipeline on CPU first (~30s, no GPU time burned):
python src/train.py --config configs/resnet50_protB.yaml \
    --packed-dir data/packed/protB --ckpt-dir /tmp/ckpt --smoke-test

# Full run — the core leakage comparison:
python src/train.py --config configs/resnet50_protB.yaml \
    --packed-dir data/packed/protB --ckpt-dir data/checkpoints/resnet50_protB
python src/train.py --config configs/resnet50_protA.yaml \
    --packed-dir data/packed/protA --ckpt-dir data/checkpoints/resnet50_protA
```

`--packed-dir` and `--ckpt-dir` are required (the YAMLs don't set them). See
`GPU_TRAINING_HANDOFF.md` for the full run matrix (convnext, vit, dinov2), resume, and W&B.

## Outputs

Each training run writes to its `--ckpt-dir`: `best.pth` (highest val AUROC — the model you use),
`latest.pth` (for `--resume`), `metrics.json` (per-epoch log), `train.log`. Checkpoints live on
`/mnt/sfs` so they survive GPU-node teardown.

## Data

LUNA16 (888 chest CT scans, derived from LIDC-IDRI), subsets 0–6 used here:
train = subsets 0–4, val = subset 5, test = subset 6. Raw data and all derived artifacts are
git-ignored; see `READING_LIST.md` §B for dataset citations.

## Status & caveats

- Core result (resnet50 A vs B) is done. Numbers above are **validation** AUROC; a test-set
  eval script (`src/evaluation/`) is not written yet.
- Secondary architectures (convnext, vit, dinov2) are configured but not yet run.
- This is a research study on detection, **not** a clinical or diagnostic tool.
