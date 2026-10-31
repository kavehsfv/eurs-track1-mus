#!/usr/bin/env python3
"""Dev inference for v4 restorer: eval-target size, portrait transposed to landscape
(matches training), /32 reflect-pad, flip-TTA, transpose back, crop, save PNG."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

import v3_common as C

MAX_SIDE = 1024


def target_size(w, h):
    s = MAX_SIDE / max(w, h)
    return (w, h) if s >= 1 else (max(1, round(w * s)), max(1, round(h * s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--indir', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--batch', type=int, default=6)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = False
    bb = smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=3)
    sd = torch.load(a.ckpt, map_location=dev)
    model = C.ResidualRestorer(bb).to(dev).eval()
    model.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd)

    indir, outdir = Path(a.indir), Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for f in sorted(indir.glob('*.png')):
        with Image.open(f) as im:
            groups.setdefault(target_size(*im.size), []).append(f)
    done = 0
    for (tw, th), files in sorted(groups.items()):
        portrait = th > tw
        W, H = (tw, th) if not portrait else (th, tw)  # landscape canvas
        Hp, Wp = (H + 31) // 32 * 32, (W + 31) // 32 * 32
        print(f'group {tw}x{th} ({"portrait" if portrait else "landscape"}, canvas {Wp}x{Hp}): {len(files)}', flush=True)
        for i in range(0, len(files), a.batch):
            chunk = files[i:i + a.batch]
            arrs = []
            for f in chunk:
                raw = cv2.cvtColor(cv2.imread(str(f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                raw = cv2.resize(raw, (tw, th), interpolation=cv2.INTER_AREA)
                if portrait:
                    raw = np.ascontiguousarray(raw.transpose(1, 0, 2))
                raw = cv2.copyMakeBorder(raw, 0, Hp - raw.shape[0], 0, Wp - raw.shape[1], cv2.BORDER_REFLECT)
                arrs.append(raw)
            x = torch.stack([torch.from_numpy(r.astype(np.float32) / 255.0).permute(2, 0, 1) for r in arrs]).to(dev)
            with torch.no_grad():
                o = model(x)
                o = (o + torch.flip(model(torch.flip(x, [3])), [3])) / 2
                o = o.clamp(0, 1)
            for f, t in zip(chunk, o):
                u = (t.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)[:H, :W]
                if portrait:
                    u = np.ascontiguousarray(u.transpose(1, 0, 2))
                Image.fromarray(u, 'RGB').save(outdir / (f.stem + '.png'))
            done += len(chunk)
            if done % 200 < a.batch:
                print(f'  {done}...', flush=True)
    print(f'restored {done} -> {outdir}', flush=True)


if __name__ == '__main__':
    main()
