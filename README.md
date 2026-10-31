# EURS Track 1 — `mus_v4_NatRes1024` reproduction repository

Reproduction package for our MaCVi 2027 EURS Track 1 (testing-phase) submission.
The submission is produced by **three checkpoints** (links + hashes below) and the
inference scripts in this repo — no other artifacts are needed.

## Pipeline summary

**Restoration** (`stage_v4_restore/`): residual U-Net (ResNet-34 encoder, ImageNet init)
trained at the exact evaluation resolution — full-resolution degraded/GT pairs resized
with the official longest-side-1024 formula, portrait images transposed to landscape,
reflect-padded to /32. Loss: Charbonnier + SSIM + VGG19-conv5_2 + task-driven perceptual
L1 on frozen segmentation-decoder features (1/8 and full scale) + Sobel edge loss, with
component weights auto-calibrated at init. Inference: horizontal-flip TTA, un-transpose,
crop, save.

**Segmentation** (`stage_v3_seg/`): logit average of two U-Nets (ResNet-34) on raw
frames at 768×384, masks nearest-upsampled to the evaluator target size.
- Seg-A: class-balanced CE (effective-number weights from per-class image-presence
  counts, β=0.999) + multiclass Dice + per-sample difficulty gate.
- Seg-SD: self-distilled student of Seg-A adding KL distillation with cosine-annealed
  temperature 4.0→0.8 (denoises the shared per-scene GT for moving classes).

**Packaging** (`package_v2.py`): verifies stems/dims/RGB/mask-range against the phase's
degraded images, then zips `restored/` + `seg_masks/` at the archive root. Restored
images are JPEG q98 4:4:4 (fidelity ≥44 dB vs the PNG master) to fit the 1 GB upload cap;
masks are single-channel PNG with values 0–4.

## Checkpoints used for the reported submission

| role | file | epoch | SHA-256 |
|---|---|---|---|
| restoration | `rest1024_best_psnr.pth` | **14 / 16** (best PSNR and best SSIM coincide) | `a4e9131ff3b85e06c09219753d02d43e68b9b1851692823dea67ab11be505213` |
| segmentation (member 1) | `segA_best.pth` | **2 / 25** (best val mIoU 0.3052) | `0176117c523c63c2d0a307d0ebd78c1f69ef2b5deebd6711c5d8ac8620e630a1` |
| segmentation (member 2) | `segSD_best.pth` | **1 / 25** (best val mIoU 0.3332) | `54b687f816d496dcca70203ba6d54a6a42d0f9f793c998c94c121fcef48ac954` |

Download from the [weights release](../../releases/tag/weights) and place them next to
the scripts (paths below assume `weights/`).

Commit of this repository used for the reported submission: see the git tag
`submission-test-phase` (and the commit hash in the answer to the organizers).

## Environment

- Python 3.11, PyTorch 2.x + CUDA (tested: torch 2.14+cu130 on RTX 5090; any ≥2.4 works)
- `pip install segmentation-models-pytorch==0.5.0 opencv-python-headless pillow numpy kornia`
- One GPU with ≥16 GB (restoration inference batch 6 at 1024×768; training used batch 3 at 768×1024)

## Data layout expected

```
Datasets/
  train/degraded/<scene>/*.png        # full-res degraded frames (20 scenes × 200)
  train/degraded_384/<scene>/*.png    # 768×384 versions (provided by organizers)
  train/gt_images/<scene>.png         # full-res clean images (20)
  train/gt_masks_384/<scene>.png      # 768×384 masks (20)
  <phase>/degraded_flat/*.png         # the phase's degraded images (Dev or testing-phase), flat dir
```

## Reproduction steps (exact commands)

```bash
# 0) sanity: checkpoints + data in place
python -c "import torch, segmentation_models_pytorch"   # env OK

# 1) RESTORATION inference (testing-phase degraded -> restored/, flip-TTA)
python stage_v4_restore/infer_restore_v4.py \
    --ckpt weights/rest1024_best_psnr.pth \
    --indir <phase>/degraded_flat --outdir out_restored --batch 6

# 2) SEGMENTATION inference (Seg-A + Seg-SD logit ensemble on raw frames)
python stage_v3_seg/infer_seg_tta.py \
    --ckpt weights/segA_best.pth --ckpt2 weights/segSD_best.pth --w1 0.5 \
    --indir <phase>/degraded_flat --outdir out_masks --out-size target --batch 16

# 3) JPEG q98 4:4:4 for restored/ (fits the 1 GB cap; fidelity >= 44 dB)
#    (inline script in REPORT_v4.md §repro; any quality-98 4:4:4 encoder works)

# 4) verify + zip (mandatory: package_v2.py checks stems/dims/RGB/mask-range)
python package_v2.py --degraded-dir <phase>/degraded_flat \
    --restored-dir out_restored_jpg --masks-dir out_masks --zip submission.zip
```

Runtime: ~12 min restoration (1200 imgs, RTX 5090) + ~4 min segmentation + ~3 min packaging.

## Training from scratch (optional, for completeness)

```bash
# Seg-A (stage_v2_seg): CB-CE + Dice + difficulty gate, 25 ep, best-val ep2
python stage_v2_seg/train_seg.py --mode A --epochs 25 --batch 20 --lr 5e-4

# Seg-SD (stage_v3_seg): self-distillation from Seg-A, 25 ep, best-val ep1
python stage_v3_seg/train_seg_sd.py --epochs 25 --batch 20 --lr 5e-4 \
    --teacher weights/segA_best.pth

# Restorer v4 (stage_v4_restore): eval-native 1024 training, 16 ep, best ep14
python stage_v4_restore/train_restorer_1024.py --epochs 16 --batch 3 \
    --seg-ckpt weights/segSD_best.pth
```

Full lineage, ablations, negative results and per-epoch ledgers: `docs/REPORT_v2.md`,
`stage_v3_seg/REPORT_v3.md`, `stage_v4_restore/REPORT_v4.md`.

## Data disclosure

Trained exclusively on the provided EURS train split with ImageNet-pretrained encoder
initialization (standard `segmentation_models_pytorch` weights, as used by the
organizers' own baseline). An external underwater segmentation dataset (SUIM) was
evaluated during development but every SUIM-derived variant underperformed our models
and **none is used in the submitted pipeline** (kept in `stage_v3_seg/train_suim.py` for
transparency).
