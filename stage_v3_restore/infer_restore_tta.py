#!/usr/bin/env python3
"""Restore images with restorer_final. Modes:
  fixed384: nested scene dirs at net-native 768x384 (train/val generation).
  target: flat dev dir -> exact evaluator size (same formula as EURS_resize_for_submission.py)."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

import v3_common as C

MAX_SIDE = 1024


def scale_to_max_side(w, h, max_side=MAX_SIDE):
    s = max_side / max(w, h)
    if s >= 1:
        return w, h
    return max(1, round(w * s)), max(1, round(h * s))


def load_restorer(ckpt, dev, encoder):
    bb = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=3)
    sd = torch.load(ckpt, map_location=dev)
    m = C.ResidualRestorer(bb).to(dev).eval()
    m.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd)
    return m


def run_batch(model, dev, arrs, tta=False, unsharp=0.0, model2=None, w1=0.5):
    x = torch.stack([torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
                     for a in arrs]).to(dev)
    with torch.no_grad():
        def _fwd(m):
            o = m(x)
            if tta:
                o = (o + torch.flip(m(torch.flip(x, [3])), [3])) / 2
            return o
        o = _fwd(model)
        if model2 is not None:
            o = w1 * o + (1 - w1) * _fwd(model2)
        o = o.clamp(0, 1)
    outs = [(t.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8) for t in o]
    if unsharp > 0:
        import cv2
        res = []
        for im in outs:
            r = 2 if max(im.shape[0], im.shape[1]) <= 768 else 3
            f = im.astype(np.float32)
            u = f + unsharp * (f - cv2.GaussianBlur(f, (0, 0), r))
            res.append(np.clip(u, 0, 255).round().astype(np.uint8))
        outs = res
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fixed384', 'target'], required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--indir', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--tta', action='store_true', help='flip-average')
    ap.add_argument('--unsharp', type=float, default=0.0, help='unsharp-mask strength (0=off)')
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--ckpt2', default=None, help='second ckpt for output ensemble')
    ap.add_argument('--w1', type=float, default=0.5)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = False
    model = load_restorer(a.ckpt, dev, a.encoder)
    model2 = load_restorer(a.ckpt2, dev, a.encoder) if a.ckpt2 else None
    indir, outdir = Path(a.indir), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    done = 0

    if a.mode == 'fixed384':
        for scene in sorted(d for d in indir.iterdir() if d.is_dir()):
            (outdir / scene.name).mkdir(parents=True, exist_ok=True)
            files = sorted(scene.glob('*.png'))
            for i in range(0, len(files), a.batch):
                chunk = files[i:i + a.batch]
                arrs = [np.asarray(Image.open(f).convert('RGB')) for f in chunk]
                outs = run_batch(model, dev, arrs, a.tta, a.unsharp, model2, a.w1)
                for f, o in zip(chunk, outs):
                    Image.fromarray(o, 'RGB').save(outdir / scene.name / f.name)
                done += len(chunk)
                if done % 400 == 0:
                    print(f'  {done}...', flush=True)
    else:
        groups = {}
        for f in sorted(indir.glob('*.png')):
            with Image.open(f) as im:
                groups.setdefault(scale_to_max_side(*im.size), []).append(f)
        for (tw, th), files in sorted(groups.items()):
            print(f'group {tw}x{th}: {len(files)}', flush=True)
            for i in range(0, len(files), a.batch):
                chunk = files[i:i + a.batch]
                arrs = [np.asarray(Image.open(f).convert('RGB').resize((tw, th), Image.BILINEAR))
                        for f in chunk]
                outs = run_batch(model, dev, arrs, a.tta, a.unsharp, model2, a.w1)
                for f, o in zip(chunk, outs):
                    Image.fromarray(o, 'RGB').save(outdir / (f.stem + '.png'))
                done += len(chunk)
                if done % 200 == 0:
                    print(f'  {done}...', flush=True)
    print(f'restored {done} -> {outdir}', flush=True)


if __name__ == '__main__':
    main()
