# EURS Track 1 v4 — Delta Report (continues `baseline_v3/REPORT_v3.md`)
**Date:** 26-Oct-2026 · **Dir:** `baseline_v4/` · **Status: built, NOT YET submitted/scored**
**Deliverable:** `submission_v4_small.zip` (670 MB, 2400 entries, verified, integrity OK) + PNG master `submission_v4.zip` (1423 MB)

## Headline change: eval-native restorer
**v4's single big win: train the restorer at the exact evaluation resolution** (whole-image
resize to max-side 1024 via the submission formula, full-res GT pairs from `gt_images/`).
v2/v3 trained at 768×384 but *inferred* at 1024×768 — a 2.7× scale gap that v4 removes.

| Restorer (same 1024-eval-protocol val) | PSNR | SSIM |
|---|---:|---:|
| v3 final (trained @384, inferred @1024) | 17.68 | 0.774 |
| v3 + v3-psnr ensemble | 17.72 | 0.776 |
| **v4 ep14 (trained @1024)** | **18.24** | **0.776** |
| **v4 ep14 + flip-TTA (shipped)** | **18.32** | **0.783** |

Training log: 16 epochs, cosine, batch 3 (768×1024 canvas), Charbonnier+SSIM+VGG+multiTDP+
SobelEdge with v3-calibrated λ. **ep14 dominates BOTH ledgers** (17.86/0.7615 on the
training-time val subsample; 18.24/0.776 on the stride-4 probe — subsampling differences).

Architecture-of-data insight: the ep-2 law softens at native res — best epoch is 14 of 16,
not 1-2. At the scale the evaluator actually sees, the residual net has enough pixel-level
work (detail, not scene-level color) that it keeps improving; the shared-GT ceiling binds
scene-level statistics, not local detail transfer.

## Killed paths (measured this round)
1. **SUIM external pretraining** (`train_suim.py`, HF `SatwikKambham/suim`, 1488 imgs, 8-class):
   pretrain Unet-resnet34 (15 ep, 0.48 loss) → EURS finetune with CB+KD → **0.3146 vs segSD 0.3332**.
   Domain gap (open-ocean diving scenes vs our reef-tank scenes) is real. All 3-way/weighted
   ensembles with the SUIM member (incl. fish-strong ep0 variant, class-weighted variants) landed
   0.325–0.330 < A+SD 0.3311. SUIM member retained on disk (`segV4_ep0_best.pth`) but not shipped.
   *This must be disclosed as "additional data used, not in final method" per challenge rules.*
2. **High-res seg inference** (1536×768 input): fish 0.24→0.34 but bg 0.67→0.56, coral 0.62→0.47
   (scale mismatch) → mIoU 0.2855. Rejected. (The fish jump is a real signal — a scale-ensemble
   member is a future idea, but not shippable this round.)
3. **Unsharp post-sharpening at 1024:** UISM +0.13 max (vs +2.05 on v3@384) at −0.03 dB.
   The native-res + SobelEdge model is already sharp; unsharp is now a strict loss. Dropped
   (v3 shipped 0.05; v4 ships 0).
4. **Cross-res restorer ensemble (v4+v3):** 17.68 < 18.32 solo. v3 members are scale-mismatched
   at 1024 and drag. Dropped.

## Shipped recipe (v4)
- **Restore:** `rest1024_best_psnr.pth` (ep14), flip-TTA, portrait inputs transposed to landscape
  (matches training), /32 reflect-pad, JPG q98-444 (fidelity 46.9 dB).
- **Seg:** unchanged from v3 — (Seg-A + Seg-SD)/2 logit ensemble @768×384, no TTA (val 0.3311;
  masks byte-identical to v3's, reused from `baseline_v3/out_masks`).
- Val composite projection: seg block same as v3; restore block +0.6dB/+0.01 SSIM over v3
  (PSNRnorm +~0.4 if linear in the [14,24]-ish normalization window) + unsharp loss vs v3
  (v3's UISM gain ~+31% × 0.03 weight ≈ +0.01 composite — roughly a wash with SSIM gain).

## Environment note
`tr` conda env vanished mid-project; v4 runs on `pt` env (torch 2.14+cu130, smp 0.5.0 re-pinned).
All v2/v3 checkpoints re-verified loadable. A mid-session `replace` without a `path` briefly
corrupted `train_restorer_1024.py` (anchor resolved to wrong file) — restored immediately;
`ast.parse` + diff-verified. (Lesson: always pass `path`.)

## Repro (from `baseline_v4/`, env `pt`)
1. `train_suim.py` → `suim_r34.pth` (optional member; killed)
2. `train_seg_v4.py --init suim_r34.pth ...` (killed; gate 0.3146 < 0.3332)
3. `train_restorer_1024.py --epochs 16 --batch 3 --seg-ckpt ../baseline_v3/segSD_best.pth`
   → `rest1024_best_psnr.pth` (ep14; both ledgers coincide)
4. `infer_restore_v4.py --ckpt rest1024_best_psnr.pth --indir dev_flat --outdir out_restored`
5. masks: reuse v3 pipeline (`baseline_v3/infer_seg_tta.py --ckpt segA --ckpt2 segSD`)
6. `../baseline_v2/package_v2.py` → PNG master → JPG q98 ladder → `submission_v4_small.zip`

## Files
`train_suim.py`, `train_seg_v4.py`, `train_restorer_1024.py`, `infer_restore_v4.py`,
`v3_common.py` (copy), `segV4_ep0_best.pth` (unshipped SUIM member), `rest1024_*.pth`,
`REPORT_v4.md`, `submission_v4*.zip`.
