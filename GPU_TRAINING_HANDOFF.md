# GPU Training Handoff — Lung Cancer Leakage Study

> For an AI assistant (or human) running training on the **GPU node**.
> The data lives on the shared network filesystem `/mnt/sfs/lung_cancer_detection`,
> so the GPU node sees the same files this was prepared on. Read this top to bottom
> before running anything.

## STATUS: data is regenerated and verified — ready to train ✅ (2026-06-13)

- Preprocessing bug fixed (see §2). All subsets 0–6 re-extracted, manifest merged
  (5671 rows, 0 duplicate files), splits rebuilt, and packed to `.npy` for both protocols:
  - `data/packed/protB/{train,val,test}.npy` = (4070, 769, 832) × 224×224
  - `data/packed/protA/{train,val,test}.npy` = same shapes
- Class balance per protocol: train nodule=1845 / normal=2225, val 324/445, test 387/445.
- **Fix verified on the packed data**: a trivial mean-brightness classifier scores
  **AUROC 0.541** (≈ chance) — under the old bug it was ~1.0. The whole-image shortcut is gone.
- Split integrity passes (`pack_dataset` re-asserts no scan overlap for Protocol B).
- Old `resnet50_protB` checkpoint was deleted — do not look for it.

---

## 0. TL;DR — what to run

```bash
cd /mnt/sfs/lung_cancer_detection

# 1. Environment (once per fresh GPU node)
python -m venv /opt/gpuenv && source /opt/gpuenv/bin/activate   # or use the node's CUDA env
pip install -r requirements.txt        # installs torch, timm, albumentations, wandb, sklearn, etc.

# 2. Sanity: confirm CUDA + packed data are present
python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
ls data/packed/protB/*.npy data/packed/protA/*.npy   # train/val/test for both protocols

# 3. CPU smoke test FIRST (≈30s, no GPU time burned) — validates the whole pipeline
python src/train.py --config configs/resnet50_protB.yaml \
    --packed-dir data/packed/protB \
    --ckpt-dir   /tmp/ckpt_smoke --smoke-test

# 4. The actual study: clean (B) vs leaky (A) on resnet50
python src/train.py --config configs/resnet50_protB.yaml \
    --packed-dir data/packed/protB \
    --ckpt-dir   data/checkpoints/resnet50_protB
python src/train.py --config configs/resnet50_protA.yaml \
    --packed-dir data/packed/protA \
    --ckpt-dir   data/checkpoints/resnet50_protA
```

`--packed-dir` and `--ckpt-dir` are **required** (the YAML configs do not set them — see `train.py:240-241`).

---

## 1. What this study is

Comparing **Protocol A** (scan-level split, leaky) vs **Protocol B** (patient-level split, clean)
on LUNA16 CT data to measure how much data leakage inflates model performance. The headline
result is the **A-vs-B val/test AUROC gap**.

## 2. CRITICAL: a preprocessing bug was just fixed — data was regenerated

A prior run hit `val_auroc=1.0` at epoch 5 on the *clean* split. Root cause was **not** split
leakage — it was a patch-location artifact in `src/data/lidc_prepare.py`:

- **Positives** were cropped centered on the nodule (peripheral lung).
- **Negatives** were cropped at the **geometric center of the slice** (mediastinum/spine).

The model learned "center vs. periphery," trivially separable, saturating AUROC under *both*
protocols and masking the leakage gap. **Fix:** negatives are now sampled from
`candidates.csv` `class==0` rows (real non-nodule lung locations), matching the positives'
spatial distribution. All processed PNGs, manifests, splits, and packed `.npy` arrays for
protA/protB were regenerated. The old `resnet50_protB` checkpoint was deleted (trained on the
degenerate task — do not reuse it).

**Expectation after the fix:** val AUROC should now be **meaningfully below 1.0** and climb
gradually. If Protocol B *still* saturates at ~1.0 by epoch 5, stop and investigate — the
negatives may still be separable (e.g. candidate z-range bias). The separate **malignancy
study** (`data/packed/malig_*`, `data/splits_malig`) was untouched and is unrelated.

## 3. Environment notes (important)

- **This box (where prep ran) is a CPU VM with no GPU.** The original project `.venv` was
  broken (built for Python 3.12, which no longer exists here) and has been **removed**.
  Preprocessing deps were installed into `/opt/venv` (Python 3.10).
- **On the GPU node**, create a fresh env and `pip install -r requirements.txt`. The
  requirements file has a CPU section (preprocessing) and a GPU section (torch, torchvision,
  timm, albumentations, opencv, wandb, pyyaml) — installing all is fine.
- `models/build.py` uses **timm** for backbones (resnet50, convnext_tiny, vit_base_patch16_224,
  vit_base_patch14_dinov2). First run downloads pretrained weights — ensure network access or
  pre-cache `~/.cache/huggingface` / `~/.cache/torch`.

## 4. Training details (from `src/train.py`)

- Reads only the packed `.npy` arrays — **zero SFS disk I/O during training**.
- bf16 AMP, AdamW, cosine LR with linear warmup, class-weighted CrossEntropy.
- Early stopping on **val AUROC**, `patience` from config (10 for resnet).
- Checkpoints to `--ckpt-dir`: `best.pth`, `latest.pth`, `metrics.json` (survive node teardown
  because they're on SFS).
- **Resume** an interrupted run: add `--resume` (auto-loads `latest.pth` from `--ckpt-dir`,
  restores model/opt/scheduler/RNG).
- W&B logs by default (`project=lung_leakage`, run name auto-derived e.g. `resnet50_protB`).
  Add `--wandb-offline` if the GPU node has no network; sync later with `wandb sync wandb/offline-run-*`.

## 5. Full run matrix

| Config | Protocol | packed-dir | ckpt-dir |
|---|---|---|---|
| `configs/resnet50_protB.yaml` | B (clean) | `data/packed/protB` | `data/checkpoints/resnet50_protB` |
| `configs/resnet50_protA.yaml` | A (leaky) | `data/packed/protA` | `data/checkpoints/resnet50_protA` |
| `configs/convnext_protB.yaml` | B | `data/packed/protB` | `data/checkpoints/convnext_protB` |
| `configs/vit_b16_protB.yaml` | B | `data/packed/protB` | `data/checkpoints/vit_b16_protB` |
| `configs/dinov2_probe_protB.yaml` | B (frozen probe) | `data/packed/protB` | `data/checkpoints/dinov2_probe_protB` |

Run **resnet50 B and A first** — that pair is the core leakage finding. The other three are
secondary (different architectures on the clean split). Each `--ckpt-dir` must be distinct or
runs overwrite each other.

Example (convnext):
```bash
python src/train.py --config configs/convnext_protB.yaml \
    --packed-dir data/packed/protB \
    --ckpt-dir   data/checkpoints/convnext_protB
```

## 6. Reading results

```bash
# Best val AUROC + last epoch for a run
python3 -c "import json; m=json.load(open('data/checkpoints/resnet50_protB/metrics.json')); \
print('last epoch', m[-1]['epoch'], '| best val_auroc', round(max(x['val_auroc'] for x in m),4))"
```

The **study deliverable** = best val (and test) AUROC for `resnet50_protA` minus
`resnet50_protB`. A large positive gap (A ≫ B) quantifies the leakage effect. With the
preprocessing bug fixed, B should land in a realistic range (not 1.0), making the gap
interpretable.

## 7. Test-set evaluation

`src/evaluation/` is referenced in the project notes for scoring `best.pth` on the held-out
test split, but check whether it exists yet (`ls src/evaluation/`). If absent, test metrics
can be obtained by pointing `evaluate()` in `train.py` at the `test` split, or add a small
eval script that loads `best.pth` and runs `PackedLungDataset.from_packed_dir(packed_dir,
"test", protocol)`.

## 8. If you need to regenerate data from scratch (CPU node only)

```bash
PY=/opt/venv/bin/python
# Force re-extraction: clear sentinels + stale negatives + manifests
find data/luna16/processed/subset[0-6] -name "*.done"        -delete
find data/luna16/processed/subset[0-6] -name "*_norm_z*.png" -delete
rm -f data/luna16/processed/manifest_subset[0-6].csv data/luna16/processed/manifest.csv
for i in 0 1 2 3 4 5 6; do
  $PY src/data/lidc_prepare.py --subset data/luna16/raw/subset$i \
      --out data/luna16/processed --config-root /mnt/sfs/lung_cancer_detection
done
$PY src/data/lidc_prepare.py --merge-only --out data/luna16/processed
$PY src/data/make_luna_splits.py --manifest data/luna16/processed/manifest.csv --out-dir data/splits
$PY src/data/pack_dataset.py --splits-dir data/splits --out-dir data/packed/protB --protocol B
$PY src/data/pack_dataset.py --splits-dir data/splits --out-dir data/packed/protA --protocol A
```

Only subsets **0–6** exist on disk (train=0-4, val=5, test=6 per `src/config.py`).
