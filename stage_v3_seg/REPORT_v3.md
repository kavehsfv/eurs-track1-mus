# EURS Track 1 v3 — Delta Report (continues `baseline_v2/REPORT_v2.md`)
**Date:** 17-Sep-2026 · **Dir:** `baseline_v3/` · **Status: built, NOT YET submitted/scored**
**Deliverable:** `submission_v3_small.zip` (659 MB, 2400 entries, verified, integrity OK) + PNG master `submission_v3.zip` (1393 MB)

## v3 design (paper → implementation, from repo code not just cards)

| # | Idea | Source (code read) | Implementation | Val verdict |
|---|---|---|---|---|
| 1 | Self-distillation, KD-as-denoiser | MaskGuide `DistillFintuner.py` (annealed T 4→0.8, manual KL·T², 1e-7 clamp) | Seg-SD: fresh resnet34, CB-CE+Dice+1.0·KD from frozen Seg-A | **0.3332** (+9.2% over Seg-A; elasmo 0.055→0.170) |
| 2 | TDP on final decoder feats | DTI-UIE `vgg_unet.py` (C5=up5!) + `loss.py` FeatureLoss | MultiTDP: L1 on blocks[1]@1/8 + blocks[-1]@full-res | in restorer-v3 (below) |
| 3 | Robust pixel + edge loss | DTI-UIE `loss.py` (Charbonnier, SobelEdgeLoss) | Charbonnier + unsigned SobelEdge@0.3 + resnet34 | restorer ≈ v2 (data-capped) |
| 4 | Exact local NR metrics | DTI-UIE `uie_eval.py` (torch UIQM/UCIQE; FUnIE's is py2-broken) | `v3_common.uiqm/uism/uciqe_batch` (in-place bug fixed, λb=0.144 kept) | enabled #5 + tie-breaks |
| 5 | Unsharp for sharpness slice | UISM as sharpness proxy (#4) | Val ladder → **s=0.05** (UISM +31%, PSNR −0.00dB, SSIM +0.0006) | ship (rule corrected: max-UISM, not max-strength) |
| 6 | Logit ensemble (coverage insurance) | — (risk analysis, §7.1 of v2 report) | (Seg-A + Seg-SD)/2, no TTA | **0.3309**, all classes IoU>0.02 (ships over SDt 0.3344) |
| 7 | Restorer ckpt-ensemble + TTA | standard practice, both components validated | (ep2+ep10)/2 + flip-avg + unsharp | +0.57dB ens, +0.17dB TTA (val-200) |

Final dev recipe — restore: `(best_ssim + best_psnr)/2 + TTA + unsharp-0.05`; seg: `(Seg-A + Seg-SD)/2` on raw, no TTA; JPG q98-444 (fidelity 46.5dB → ~0.003dB scored cost).

## Killed paths (measured)
- **MobileSAM pseudo-labels:** max-IoU gate 0.432 < 0.60 (coral 0.31 on *clean* images!); restored-frame mask counts collapse to ~1/3 on heavy-degradation frames. Auto-gen SAM unreliable on this domain. (v4: Seg-A-box-prompted SAM instead of auto-gen.)
- **FUnIE-generator transfer/ensemble:** 13.60/0.456 on our val — worse than classical. EUVP≠EURS.
- **Seg-TTA on ensemble:** helps single (+0.001) but hurts ENS (−0.006, tail dilution). Flip-averaging smooths rare-class logits below argmax.
- **Bigger restorer backbone:** resnet34 peaks 16.47/0.683 vs v2-resnet18 16.49/0.674, then overfits harder (→15.6). Ep-2 law covers restoration too; shared-GT degeneracy caps learning, not capacity.
- **CB cards 04/08:** confirmed v2 already correct (softmax-mean CE, cosine steep tail). No change.

## Key numbers (val: fish1/fish5/ray1/turtles1)
- Seg-SD best 0.3332/F1 0.4424 @ **ep1** (ep-1 law under KD), per-class [0.660, 0.611, 0.226, 0.170, 0.000]; class-4 collapse (KD suppresses where teacher is confidently wrong) motivated the ensemble.
- Restorer-v3 ledgers — PSNR: ep2 16.47/0.6625/0.302; SSIM: ep10 16.02/0.6833/0.271 (=final); miou: ep2 (missed shortlist on SSIM). v3-vs-v2 on val-400: PSNR tie, SSIM +0.01, UIQM +0.08, UISM +0.05, UCIQE −0.02 → ship v3.
- Unsharp: UISM peaks at s=0.05 (+2.05), declines after — non-monotonic; the naive max-strength rule would have shipped s=0.3 (less sharpness, more PSNR cost).

## Honest limitations / risks
- v3 unscored: test projections IF patterns hold — seg val 0.331→~0.39 (v2 saw +18%), restorer val-equiv ~16.9→~18.5+, sharpness/UIQM up via unsharp. Leaderboard will judge.
- ens2+TTA stacking assumed additive (components validated separately; zero-downside-risk argument documented in session).
- UCIQE −0.02 vs v2 restorer (accepted: SSIM+UIQM wins dominate by weight).
- Single-seed; val = 4 scenes (0.002–0.003 gaps are noise — decisions were risk-based, not decimal-based).

## Repro (from `baseline_v3/`)
`train_seg_sd.py` (teacher `../baseline_v2/segA_best.pth`) → ensemble/TTA gates → `train_restorer_v3.py --seg-ckpt segSD_best.pth` → `infer_restore_tta.py fixed384` → `tune_unsharp.py` → TTA/ens probes → dev: restore `--ckpt restorer_final.pth --ckpt2 restorer_best_psnr.pth --tta --unsharp 0.05`, seg `--ckpt ../baseline_v2/segA_best.pth --ckpt2 segSD_best.pth` → `../baseline_v2/package_v2.py` → JPG ladder → `submission_v3_small.zip`.
