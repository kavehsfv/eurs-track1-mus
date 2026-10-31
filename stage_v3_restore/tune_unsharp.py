#!/usr/bin/env python3
"""Tune unsharp-mask strength on restored val subset: max strength with PSNR cost <=0.15dB
and SSIM cost <=0.003; UISM (local sharpness proxy) reported. Radius 2 @384px (use 3 @1024px)."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import v3_common as C

TRAIN_ROOT = Path('/home/kaveh/Workspace/Projects/Code/Challenges/MaCVi/Datasets/train')
G384 = TRAIN_ROOT / 'gt_images_384'
VAL_SCENES = ['fish1', 'fish5', 'ray1', 'turtles1']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--restored-dir', required=True)
    ap.add_argument('--stride', type=int, default=4)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ssim = C.SSIM().to(dev)
    rdir = Path(a.restored_dir)

    pairs = []
    for s in VAL_SCENES:
        gt = cv2.cvtColor(cv2.imread(str(G384 / f'{s}.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        for f in sorted((rdir / s).glob('*.png'))[::a.stride]:
            r = cv2.cvtColor(cv2.imread(str(f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            pairs.append((r.astype(np.float32) / 255.0, gt.astype(np.float32) / 255.0))
    print(f'{len(pairs)} pairs', flush=True)
    to_t = lambda l: torch.stack([torch.from_numpy(x).permute(2, 0, 1) for x in l]).to(dev)

    def sharp(imgs, s, radius=2):
        out = []
        for im in imgs:
            f = (im * 255.0).astype(np.float32)
            u = f + s * (f - cv2.GaussianBlur(f, (0, 0), radius))
            out.append(np.clip(u, 0, 255).astype(np.float32) / 255.0)
        return out

    base_r = [p[0] for p in pairs]
    gts = to_t([p[1] for p in pairs])
    with torch.no_grad():
        b = to_t(base_r)
        p0 = float(np.mean([C.psnr_from_mse(v) for v in
                            torch.nn.functional.mse_loss(b, gts, reduction='none').mean(dim=(1, 2, 3)).cpu().tolist()]))
        s0 = float(ssim(b, gts).mean().item())
        u0 = float(C.uism_batch(b).mean())
    print(f's=0.00: PSNR {p0:.2f} SSIM {s0:.4f} UISM {u0:.4f}', flush=True)
    chosen = 0.0
    for s in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
        with torch.no_grad():
            u = to_t(sharp(base_r, s))
            ps = float(np.mean([C.psnr_from_mse(v) for v in
                                torch.nn.functional.mse_loss(u, gts, reduction='none').mean(dim=(1, 2, 3)).cpu().tolist()]))
            sm = float(ssim(u, gts).mean().item())
            um = float(C.uism_batch(u).mean())
        ok = (ps >= p0 - 0.15) and (sm >= s0 - 0.003)
        print(f's={s:.2f}: PSNR {ps:.2f} ({ps - p0:+.2f}) SSIM {sm:.4f} ({sm - s0:+.4f}) '
              f'UISM {um:.4f} ({um - u0:+.4f}) {"OK" if ok else "--"}', flush=True)
        if ok:
            chosen = s
    print(f'CHOSEN unsharp strength: {chosen} (apply with --unsharp {chosen})', flush=True)


if __name__ == '__main__':
    main()
