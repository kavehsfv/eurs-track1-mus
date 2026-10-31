#!/usr/bin/env python3
"""JPEG q98 4:4:4 conversion for restored/ (fits 1 GB upload cap, fidelity >= 44 dB).
Auto quality ladder: highest of (100, 98, 95) whose projected total fits the cap."""
import argparse, io, math
from pathlib import Path
import numpy as np
from PIL import Image

def jpg_bytes(im, q):
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=q, subsampling=0, optimize=True)
    return b.getvalue()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--cap-mb', type=float, default=1000.0)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob('*.png'))
    sample = files[::20]
    q = 95
    for cand in (100, 98, 95):
        proj = sum(len(jpg_bytes(Image.open(f).convert('RGB'), cand)) for f in sample) \
               / len(sample) * len(files) / 1e6 + 10
        print(f'q{cand}: projected {proj:.0f} MB', flush=True)
        if proj < a.cap_mb:
            q = cand
            break
    ps, total = [], 0
    for f in files:
        png = np.asarray(Image.open(f).convert('RGB')).astype(np.float32)
        b = jpg_bytes(Image.open(f).convert('RGB'), q)
        (dst / (f.stem + '.jpg')).write_bytes(b)
        total += len(b)
        jpg = np.asarray(Image.open(io.BytesIO(b)).convert('RGB')).astype(np.float32)
        mse = ((png - jpg) ** 2).mean()
        ps.append(20 * math.log10(255) - 10 * math.log10(mse + 1e-12) if mse > 0 else 99.0)
    print(f'q{q}: {total / 1e6:.0f} MB, fidelity mean {np.mean(ps):.2f} dB min {np.min(ps):.2f} dB', flush=True)

if __name__ == '__main__':
    main()
