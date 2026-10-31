#!/usr/bin/env python3
"""Seg masks from restored images: any size -> 768x384 net -> argmax -> NEAREST back to
input size. Saves single-channel PNGs with values 0-4."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

import v2_common as C

MEAN = np.array(C.IMAGENET_MEAN, dtype=np.float32)
STD = np.array(C.IMAGENET_STD, dtype=np.float32)


def scale_to_max_side(w, h, max_side=1024):
    s = max_side / max(w, h)
    if s >= 1:
        return w, h
    return max(1, round(w * s)), max(1, round(h * s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--indir', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--out-size', choices=['input', 'target'], default='input',
                    help='target: resize mask to evaluator size (same formula as submission script)')
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = False
    model = smp.Unet(encoder_name=a.encoder, encoder_weights=None, in_channels=3, classes=C.NCLS)
    sd = torch.load(a.ckpt, map_location=dev)
    model.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd)
    model = model.to(dev).eval()
    indir, outdir = Path(a.indir), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(indir.glob('*.png'))
    done = 0
    with torch.no_grad():
        for i in range(0, len(files), a.batch):
            chunk = files[i:i + a.batch]
            ims = [Image.open(f).convert('RGB') for f in chunk]
            sizes = [im.size for im in ims]
            arrs = [np.asarray(im.resize(C.NET_SIZE, Image.BILINEAR)) for im in ims]
            x = torch.stack([torch.from_numpy(((a.astype(np.float32) / 255.0 - MEAN) / STD)).permute(2, 0, 1)
                             for a in arrs]).to(dev)
            preds = model(x).argmax(1).cpu().numpy().astype(np.uint8)
            for f, pr, (w, h) in zip(chunk, preds, sizes):
                ow, oh = scale_to_max_side(w, h) if a.out_size == 'target' else (w, h)
                m = Image.fromarray(pr, 'L').resize((ow, oh), Image.NEAREST)
                m.save(outdir / (f.stem + '.png'))
            done += len(chunk)
            if done % 200 == 0:
                print(f'  {done}...', flush=True)
    print(f'masks {done} -> {outdir}', flush=True)


if __name__ == '__main__':
    main()
