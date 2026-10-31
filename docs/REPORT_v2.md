# EURS Track 1 (Public Dev-Set Leaderboard): v2 Method Report
**Codename:** `mus_v2` · **Date:** 17-Sep-2026 · **Result: composite 0.310 (1st, +59% over runner-up)**
**Repo path:** `Task_1/track_1_public_leaderboard/baseline_v2/` · **Commit:** `b02c8d5 Muse_v2_17_Sept`
**Paper basis:** this report documents every idea, implementation choice, measurement, and negative result behind the submission. Intended skeleton for a MaCVi workshop challenge paper.

## Leaderboard (official, 17-Sep-2026)

| ID | Method | Composite | Q_seg | mIoU | F1 | PSNR | SSIM | Sharp | UIQM | UCIQE | CCF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| #18088 | **mus_v2 (ours)** | **0.310** | **0.163** | **0.361** | **0.483** | **18.00** | **0.746** | 0.665 | 0.827 | 0.553 | 0.030 |
| #18086 | v2g52 | 0.195 | 0.000 | 0.288 | 0.381 | 14.37 | 0.596 | 0.963 | 1.326 | 0.624 | 0.044 |
| #18011 | GrayWorld+CLAHE (our v1) | 0.195 | 0.000 | 0.261 | 0.353 | 14.37 | 0.596 | 0.963 | 1.326 | 0.624 | 0.044 |

Decomposition of our +0.115 margin: the segmentation block alone contributes
0.6 × 0.163 = **0.098 (~85% of the margin)**; the learned restorer (PSNR 18.0 vs 14.4,
SSIM 0.75 vs 0.60) contributes the rest, partly offset by the classical baseline's wins on
sharpness/UIQM/UCIQE/CCF (CLAHE artifacts score well on no-reference sharpness/contrast metrics while
hurting PSNR/SSIM — a tradeoff we consciously accepted, §6).

---

## 1. Task and metric analysis

**Task.** For each of 1200 degraded dev images (4000×3000, mixed orientation), ship a restored RGB image
and a 5-class semantic mask (0 background, 1 coral, 2 fish, 3 elasmobranch, 4 other), both at exactly the
evaluator's size (longest side 1024, aspect-preserved, `round()` semantics, never upscale). The evaluator does
not resize; size mismatch = rejection. Masks must be single-channel PNG with values in {0..4}.

**Metric.**
`composite = 0.6·Qseg + 0.3·(0.45·PSNRnorm + 0.45·SSIM + 0.10·sharpness) + 0.1·NRquality`,
with `Qseg = F1·mIoU·coverage_penalty` and `NRquality = mean(UIQM, UCIQE, CCF normalized)`.

**Strategic reading (drove all design):**
1. **Segmentation is 60% of the score and multiplicative** (`F1·mIoU·coverage`). A zero in any factor
   zeroes the whole block — exactly what happened to both competitors (`Qseg = 0.000` despite mIoU 0.26–0.29).
2. **Restoration is PSNR/SSIM-dominated** (90% of its block). This rules out GAN-style enhancement (which
   trades PSNR for perceptual punch) and favors direct L1/SSIM regression. Sharpness + no-reference metrics
   are only ~13% of total score combined — we deliberately ceded them (CLAHE wins them) to maximize PSNR/SSIM.
3. **Coverage is a hidden gate.** From our row: `coverage = 0.163 / (0.483·0.361) ≈ 0.935`; both competitors
   score exactly 0.000. Our hypothesis (§7.1): the gate requires all classes to be genuinely predicted
   (non-degenerate tail-class recall), which our class-balanced training satisfies and plain training does not.
   The evaluator source is needed to confirm the exact formula, but the arithmetic proves the coverage factor —
   not raw mIoU — is the largest single determinant of the final ranking.

## 2. Starting point: v1 baseline (composite 0.195)

- **Restoration:** classical, untrained — gray-world white balance + CLAHE (L-channel, clip 2.0, 8×8).
- **Segmentation:** `smp.Unet(resnet18, imagenet)`, 5 classes, 512×384 input, CE+Dice, AdamW 1e-3, 6 epochs,
  16 train / 4 val scenes. Val: mIoU 0.2712, F1 0.3650 (best @ ep2).
- **Packaging:** GPU resize to evaluator size + zip (shipped JPG restored, 408 MB — format risk unresolved in v1).

## 3. Data audit (the foundation of every v2 decision)

| Fact | Value | Decision it forced |
|---|---|---|
| Train frames | 4000 (200/scene × 20 scenes) @4000×3000; `*_384` pre-mades @768×384 (2:1, non-aspect) | Train at 768×384 (net-native, consistent across img/mask/GT) |
| GT images / masks | **1 per scene** (20 each) — every degraded frame maps to the scene's shared GT/mask | Per-scene degeneracy: label noise for moving classes (§7.3); residual learning; best-val selection mandatory |
| Pixel class freq (all 20 scenes) | bg 55.3 / coral 33.7 / fish 7.8 / elasmo 0.94 / other 2.2 (**59:1** max ratio) | Class-balanced loss (§4.1) |
| Train (16 scenes) image-presence | [2800, 2400, 2800, 200, 400] — **1 elasmo scene, 2 'other' scenes** | CB via image counts (pixel counts saturate any β); no resampling (conflicts with CB) |
| Val scenes | fish1, fish5, ray1, turtles1 — all 5 classes present (4/3/2/1/1 scenes) | Held-out-scene protocol; val is *harder* than dev test (§7.5, conservative selection) |
| Dev | 1200 PNG, 4000×3000 + portrait mix → targets 1024×768 / 768×1024 | Restore at target size; seg via 768×384 net + nearest-upscale |

## 4. Method v2

Pipeline (sequential, decoupled at inference): **Seg-A (raw) → Restorer (TDP-guided) → [Seg-B rejected] → ship Seg-A masks + restorer outputs.**

### 4.1 Segmentation (Seg-A): CB-CE + Dice + difficulty gate — adapted from [Cui et al.] + [Wu et al.]

- Architecture: `smp.Unet(encoder=resnet34, imagenet)`, 5 classes, 768×384. (No FLOP cap on Track 1; resnet18→34 is free capacity.)
- **Loss = CB-weighted CE + Dice (multiclass).** CB weights from [Cui et al.],
  $w_c \propto (1-\beta)/(1-\beta^{n_c})$, **sum-to-C normalized**, with $n_c$ = *image-presence* counts and
  $\beta=0.999$ → $w$ = [0.452, 0.467, 0.452, **2.341**, 1.287] (5.2× spread). Pixel counts (≈10⁸) saturate
  $E_n \to 1/(1-\beta)$ for any sane β — the image-count formulation is required for segmentation (derived §7.4).
  Softmax head keeps default bias init (scalar shift is gradient-inert under softmax — the paper's deliberate split).
  No focal term (CB-CE + per-class-averaged Dice suffices; less tuning risk).
- **Difficulty gate (continuous port of MaskGuide's IoU gate [Wu et al.]).** Their code halves KD loss when
  sample IoU ≥ 0.9 (paper's Eq.4 contradicts the code; we port the *code* behavior, continuous):
  per-sample $w_i = 0.5 + (1 - \mathrm{macroIoU}_i)$ over GT-present classes (no-grad), multiplying that sample's
  CE. Bounded [0.5, 1.5]; verified active (random-init model → w ≈ 1.43–1.48).
- **No resampling** — [Cui et al.] trains on the natural distribution and CB explicitly conflicts with
  sampling-based rebalancing; MaskGuide-style 0.6/0.4 resampling was considered and rejected on this ground.
- Optimization: AdamW, lr 5e-4, wd 1e-4, cosine decay, batch 20, 25 epochs, aug = hflip p=0.5 + ±10%
  brightness/contrast (same transform on image+mask). Seed 0, NaN-guard skip, resume support.
- **Selection: best-val-mIoU (macro over pooled pixels, same function as v1 for comparability) + val snapshots.**
  Result: **best 0.3052 / F1 0.4121 @ ep2**, per-class IoU [0.636, 0.575, 0.191, 0.055, 0.068].
  Train loss 0.70→0.14 while val never recovers after ep2 — best-val selection is load-bearing (§7.2).

### 4.2 Restoration: residual UNet + L1/SSIM/VGG/TDP, auto-calibrated — adapted from [Lin et al.] + [Islam et al.]

- Architecture: `smp.Unet(resnet18, imagenet)` wrapped as a **residual-from-identity restorer**
  ([Lin et al.] two-branch residual insight, lite form): `out = clamp(x + std·net(norm(x)))`.
  FUnIE's lesson (slim widths suffice for color-cast correction) justifies resnet18 over resnet34 here.
- **Loss = L1 + 0.5·(1−SSIM) + λᵥ·VGG + λₜ·TDP.**
  - VGG: frozen VGG19 **conv5_2 (features idx 30, pre-ReLU) MSE** [Islam et al.], ImageNet-normalized [0,1]
    inputs (standard perceptual contract; deviates from FUnIE's no-renorm quirk).
  - TDP (task-driven perceptual, [Lin et al.] lite): L1 on **frozen Seg-A decoder block[1] (~1/8 scale,
    128ch)** features between restored and GT. No 10-epoch warmup gate — unnecessary because the seg net is
    *pre-trained*, not co-trained (this removes the moving-target problem their gate solves).
  - **No adversarial term**: FUnIE's own ablation shows L1 does the heavy lifting (+4.58% UIQM vs +1.07% content),
    and GAN hallucination directly spends the PSNR/SSIM budget that is 90% of this block.
- **Auto-calibrated lambdas (scale-free).** Measured on 20 train batches at init:
  L1 0.2352, SSIM-term 0.6119, VGG **16.325**, TDP 0.240 → λᵥ = 0.43·L1/VGG = **0.0062**,
  λₜ = 0.50·L1/TDP = **0.4898** (VGG ratio honors FUnIE's 7:3; TDP 0.5× keeps task-guidance strong-not-dominant).
  Hand-picked VGG weights (0.1–0.3) would have been ~100× off — calibration was essential, not cosmetic.
- Optimization: AdamW 1e-4, wd 1e-3, cosine, batch 12, 25 epochs, paired hflip. Seed 0.
- **Dual-criterion checkpointing [Lin et al.]: best-PSNR AND best-downstream-mIoU** (Seg-A on restored val),
  full ledgers. Result: best-PSNR {ep5: 16.49 / 0.669 / d-mIoU 0.264} vs best-mIoU
  {ep10: **16.03 / 0.674 / 0.277**} — the criteria **diverge**, exactly as [Lin et al.] predicts.
  Auto-rule (ΔPSNR 0.46 ≤ 0.5 dB → task-useful wins): shipped ep10. SSIM was also higher there (0.674 > 0.669).
  Train loss 0.52 → 0.25, zero NaN skips.

### 4.3 Rejected path: Seg-B joint/mixed training (measured, not assumed)

Following [Lin et al.] Phase-2 exactly (CE on raw + enhanced + soft `sigmoid(randn)`-blend inputs),
Seg-B finetuned from Seg-A (15 ep, lr 2e-4): **restored-val 0.2610, raw-val 0.2738 — worse than Seg-A (0.3052)
on both distributions.** Verdict: with a 16-dB restorer, restoration destroys more discriminative detail than it
recovers, and TDP-from-a-0.3-mIoU-seg is weak guidance. The DTI-UIE thesis requires a restorer strong enough to
behave; ours is not there yet. Seg-B discarded; final masks come from Seg-A on raw inputs. (Notably, [Wu et al.]
Table VI independently reports enhance-then-segment underperforming robust-feature approaches.)

### 4.4 Submission engineering (zero-accuracy-loss logistics)

- **Restore at evaluator target size** (PIL bilinear in → net → ship; no output resize). Seg via 768×384 net,
  argmax, nearest-upscale to target. Size formula byte-identical to the official script (`round()`, no upscale).
- **1000 MB upload cap:** lossless PNG master = 1481 MB; PNG level-9 re-encode caps at ~1330 MB, so near-lossless
  JPG for `restored/` only was mandatory (masks stay lossless PNG — they carry the 0.6 block). Auto quality ladder
  (project from 60-image sample): q100 overflowed → **q98 4:4:4 optimize → 760 MB**.
  **Measured cost on all 1200: PNG-vs-JPG PSNR 45.64 dB (min 44.06), max |Δ| 12/255 → scored-PSNR impact ≈ 0.005 dB**
  (MSE-additivity: 0.0251 + 0.000028), composite impact ≈ 1e-5. Negligible by construction and by measurement.
- **Verification harness** (`package_v2.py`): stem-set equality, per-file dims vs formula, RGB check, mask
  single-channel + values ⊆ {0..4}, then zip + `testzip` integrity. Aborts on any failure. Extension note: restored
  ships as `.jpg` (README: "matching filename stem"; v1 precedent) — the one unverified assumption, forced by the
  size cap (the alternative, 1481 MB PNG, cannot be uploaded at all).

## 5. Experiments

### 5.1 Validation (4 held-out scenes: fish1, fish5, ray1, turtles1)

| Model | Val signal | Best val | Note |
|---|---|---:|---|
| Seg-A (resnet34, CB-CE+Dice+gate) | mIoU / F1 | **0.3052 / 0.4121 @ ep2** | v1: 0.2712/0.3650 → **+12.5% mIoU** |
| Restorer (L1+SSIM+VGG+TDP) | PSNR / SSIM / d-mIoU | 16.49 / 0.669 / 0.264 @ ep5 (fidelity); **16.03 / 0.674 / 0.277 @ ep10 (shipped)** | dual ledger diverges |
| Seg-B (finetune, raw+rest+mix) | mIoU restored / raw | 0.2610 / 0.2738 | **rejected** — worse on both |
| Classical gray+CLAHE (measured) | PSNR / SSIM | 13.73 / 0.611 | learned wins **+2.3 dB / +0.06** |

Seg-A per-class IoU @ best: [bg 0.636, coral 0.575, fish 0.191, elasmo 0.055, other 0.068].
Seg-A train loss 0.70→0.14; restorer 0.52→0.25; Seg-B 0.15→0.06 (deepest memorization, worst transfer).

### 5.2 Test (official dev leaderboard) — test exceeds val on every term

| Term | Val (ours) | Test (official) |
|---|---|---|
| mIoU / F1 | 0.305 / 0.412 | **0.361 / 0.483** |
| PSNR / SSIM | 16.03 / 0.674 | **18.00 / 0.746** |

Val scenes (esp. ray1/turtles1, tail-class-only) are harder than dev scenes — the val protocol was conservative,
which is the safe direction for model selection (it cannot flatter overfit models).

### 5.3 Ablation-equivalent evidence (from the staged design)

Each stage isolates one factor vs v1: Seg-A (CB + gate + resnet34 + best-val + 768×384) +0.034 mIoU;
restorer (learned vs classical, same val) +2.3 dB / +0.06 SSIM; Seg-B (−0.044, negative); JPG (−0.005 dB, negligible).
Single-seed runs (limitation §8); the leaderboard margin (+59%) dwarfs seed noise.

## 6. Metric-strategy postmortem (read the full row, not just composite)

We lose sharpness (0.665 vs 0.963), UIQM (0.827 vs 1.326), UCIQE and CCF to the CLAHE baseline — and win
overall by 59%. This was the designed tradeoff: CLAHE-style processing buys no-reference punch while spending
PSNR/SSIM, and the composite prices PSNR/SSIM ≈ 7× above sharpness+NR combined. A v3 hybrid (learned restore +
mild unsharp/UIQM-aware finetune) could reclaim part of the 0.13 sharpness+NR slice without denting PSNR — but it
was correctly deprioritized behind the 0.6 seg block and the PSNR/SSIM core.

## 7. Key insights (the paper's discussion section)

1. **Coverage is the ranking's hidden variable.** Competitors: mIoU 0.26–0.29 yet Qseg = 0.000 ⇒ coverage = 0.
   Ours: 0.935. Hypothesis: the gate demands genuine tail-class prediction; CB weights (2.34× elasmo) keep rare
   classes alive in the output distribution. A method can gain +0.03 mIoU and still score 0.000 if it collapses a
   class — future work should optimize coverage explicitly (per-class recall floors, test-time class priors).
2. **The ep-2 law.** v1 (resnet18, plain loss) and v2 (resnet34, CB+gate) both peak at epoch 2 of ~160 iters, then
   decay while train loss keeps falling. With 16 scenes and shared per-scene targets, transferable signal is
   exhausted almost immediately; the rest is memorization. Corollary: best-val checkpointing is not hygiene here,
   it is the method (last-epoch weights score 0.234 — below v1).
3. **Shared-mask degeneracy is class-selective.** Static classes transfer (coral 0.58); anything that swims is
   frame-misaligned with its "ground truth" (fish 0.19 despite 14 scenes; elasmo/other ≈ 0). No loss fixes wrong
   labels — per-frame pseudo-labels (SAM snapped to scene classes) are the principled fix and the top v3 item.
4. **CB needs image-counts for segmentation.** Pixel counts (≈10⁸) saturate $E_n$ for any β; image-presence counts
   ([2800,2400,2800,200,400], β=0.999) yield a sane 5.2× spread. Tail classes remain *data*-absent (1 elasmo scene),
   bounding what any weighting can do — external data (SUIM/UIIS, allowed with disclosure) is the other half.
5. **Perceptual-loss magnitudes are treacherous.** VGG-conv5_2 MSE measured 16.3 vs L1 0.24 at init — any hand-set
   λ would have been orders off. Auto-calibration (fix each term's relative contribution from 20 batches) should be
   default practice for multi-term restoration losses.
6. **Task-guidance is only as good as the task net.** TDP-from-Seg-A (0.3 mIoU) + a 16-dB restorer could not make
   restored inputs beat raw for seg (0.261 vs 0.305). Joint enhancement-then-segmentation needs a stronger restorer
   first — sequence matters: fidelity foundation, then task coupling.

## 8. Limitations (honest, for reviewers)

- Single-seed runs (seed 0); no error bars. Mitigated by the +59% margin and staged isolation, but reruns are needed.
- TDP extractor is a weak seg net (0.305 val); task-guidance quality is bounded by it.
- No external data, no pseudo-labels — the two highest-leverage items remain open (by design: v2 was scoped to a
  shippable milestone).
- `.jpg` restored extension relies on stem-matching + v1 precedent (unverified against evaluator source).
- Val = 4 scenes (800 imgs); scene-level noise is real, though the test>val direction shows selection was conservative.
- Sharpness/NR slice (0.13) consciously ceded to CLAHE-style processing.

## 9. Reproducibility appendix

- **Env:** RTX 5090 32GB; conda env `tr`: torch 2.8.0+cu128, torchvision 0.23, smp 0.5.0, cv2 4.11; seeds fixed (0).
- **Files:** `v2_common.py` (losses/metrics/CB/SSIM/VGG/TDP/mix/gate), `train_seg.py --mode A|B`,
  `train_restorer.py`, `infer_restore.py --mode fixed384|target`, `infer_seg.py [--out-size target]`,
  `package_v2.py` (verify+zip). Checkpoints: `segA_best.pth` (0.3052), `restorer_final.pth` (ep10 ledger).
- **Commands (from `baseline_v2/`):**
  `train_seg.py --mode A --epochs 25` → `train_restorer.py --epochs 25 --seg-ckpt segA_best.pth` →
  `infer_restore.py --mode fixed384` (train/val, for Seg-B study) → `train_seg.py --mode B ...` (rejected study) →
  `infer_restore.py --mode target` (dev) + `infer_seg.py --ckpt segA_best.pth --out-size target` (dev raw) →
  `package_v2.py` → PNG master → q98-444 shrink → `submission_v2_small.zip` (760 MB).
- **Wall time:** ≈5 h total (Seg-A ~1 h, restorer ~2 h, Seg-B ~1.5 h, infer+pack ~0.5 h).

## 10. References

- [Lin et al.] B. Lin et al., "Downstream Task Inspired Underwater Image Enhancement," IEEE TIP 2025
  (repo: oucailab/DTIUIE) — two-branch residual restorer, TDP loss, dual-criterion checkpointing, Phase-2 recipe.
- [Cui et al.] Y. Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019 —
  $(1-\beta)/(1-\beta^n)$ weights, sum-to-C norm, no-resampling discipline.
- [Islam et al.] M. Islam et al., "Fast Underwater Image Enhancement for Improved Visual Perception
  (FUnIE-GAN)," RA-L 2020 — slim UNet, 7:3 L1:VGG, conv5_2 content loss, L1-first evidence.
- [Wu et al.] X. Wu et al., "MaskGuide: Efficient Distillation for Deployable Lightweight Segmentation in
  Marine Environments," RA-L 2026 — IoU-gated loss, focal-dominant lesson, enhance-vs-robust-features evidence.
- Ronneberger et al., U-Net (MICCAI 2015); He et al., ResNet (CVPR 2016); Simonyan & Zisserman, VGG (ICLR 2015);
  Wang et al., SSIM (TIP 2004); Panetta et al., UIQM (JOE 2016); Iakubovskii, segmentation-models-pytorch.
- MaCVi 2027 EURS challenge, Track 1 — Public Dev-Set Leaderboard (macvi.org).
