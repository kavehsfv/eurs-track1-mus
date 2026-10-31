#!/usr/bin/env python3
"""Verify v2 outputs (stems / dims / RGB / mask 0-4) then zip official format
(restored/ + seg_masks/ at root, PNG) + integrity test. Aborts the zip on any error."""
import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

MAX_SIDE = 1024


def scale_to_max_side(w, h, max_side=MAX_SIDE):
    s = max_side / max(w, h)
    if s >= 1:
        return w, h
    return max(1, round(w * s)), max(1, round(h * s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--degraded-dir', required=True)
    ap.add_argument('--restored-dir', required=True)
    ap.add_argument('--masks-dir', required=True)
    ap.add_argument('--zip', required=True)
    a = ap.parse_args()
    deg, rst, msk = Path(a.degraded_dir), Path(a.restored_dir), Path(a.masks_dir)

    targets = {}
    for f in sorted(os.listdir(deg)):
        with Image.open(deg / f) as im:
            targets[os.path.splitext(f)[0]] = scale_to_max_side(*im.size)
    print(f'degraded: {len(targets)}', flush=True)
    errs = []

    def check(d, stems, kind):
        have = {os.path.splitext(f)[0]: f for f in os.listdir(d)}
        missing = sorted(set(stems) - set(have))
        extra = sorted(set(have) - set(stems))
        if missing:
            errs.append(f'{kind}: {len(missing)} missing, e.g. {missing[:3]}')
        if extra:
            errs.append(f'{kind}: {len(extra)} extra, e.g. {extra[:3]}')
        bad_dim, bad_val = 0, []
        for stem, fname in sorted(have.items()):
            if stem not in targets:
                continue
            tw, th = targets[stem]
            with Image.open(d / fname) as im:
                if im.size != (tw, th):
                    bad_dim += 1
                    if len(bad_val) < 3:
                        bad_val.append(f'{stem}:{im.size}!={(tw, th)}')
                    continue
                arr = np.asarray(im)
                if kind == 'restored':
                    if arr.ndim != 3 or arr.shape[2] != 3:
                        errs.append(f'{kind}/{fname}: not RGB {arr.shape}')
                        break
                else:
                    if arr.ndim != 2:
                        errs.append(f'{kind}/{fname}: not single-channel {arr.shape}')
                        break
                    u = np.unique(arr)
                    if u.min() < 0 or u.max() > 4:
                        errs.append(f'{kind}/{fname}: values out of 0-4: {u[:8]}')
                        break
        if bad_dim:
            errs.append(f'{kind}: {bad_dim} wrong-size, e.g. {bad_val}')
        print(f'{kind}: {len(have)} files checked', flush=True)

    check(rst, targets, 'restored')
    check(msk, targets, 'seg_masks')
    if errs:
        print('VERIFY FAILED:', flush=True)
        [print(' -', e, flush=True) for e in errs]
        sys.exit(1)
    print('verify OK: stems, dims, RGB, mask-range all pass', flush=True)

    with zipfile.ZipFile(a.zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder, d in (('restored', rst), ('seg_masks', msk)):
            for fname in sorted(os.listdir(d)):
                zf.write(d / fname, f'{folder}/{fname}')
    bad = zipfile.ZipFile(a.zip).testzip()
    if bad:
        print(f'zip integrity FAILED at {bad}', flush=True)
        sys.exit(1)
    n = len(zipfile.ZipFile(a.zip).namelist())
    print(f'wrote {a.zip} ({os.path.getsize(a.zip) / 1e6:.0f} MB, {n} entries) integrity OK', flush=True)


if __name__ == '__main__':
    main()
