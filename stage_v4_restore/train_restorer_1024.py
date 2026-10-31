#!/usr/bin/env python3
"""Restorer-v4: trained at EVAL-NATIVE resolution (whole-image resize to max-side 1024,
exact submission formula) on full-res pairs. Portrait scenes transposed to landscape in
training; net runs natively at inference (fully-conv). Losses = v3 (Charb+SSIM+VGG+multiTDP+
SobelEdge, auto-calibrated). Lambdas carry calibrated defaults from v3 to save a calib pass."""
import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import segmentation_models_pytorch as smp

import v3_common as C

TRAIN_ROOT = Path('/home/kaveh/Workspace/Projects/Code/Challenges/MaCVi/Datasets/train')
D_FULL = TRAIN_ROOT / 'degraded'
G_FULL = TRAIN_ROOT / 'gt_images'
VAL_SCENES = {'fish1', 'fish5', 'ray1', 'turtles1'}
MAX_SIDE = 1024


def target_size(w, h):
    s = MAX_SIDE / max(w, h)
    if s >= 1:
        return w, h
    return max(1, round(w * s)), max(1, round(h * s))


def to_landscape(a):
    return np.ascontiguousarray(a.transpose(1, 0, 2)) if a.shape[0] > a.shape[1] else a


class RestDS(Dataset):
    """Full-res degraded + GT per scene, resized to eval target, forced landscape, /32-padded."""

    def __init__(self, scenes, aug=False):
        self.items = []
        for s in scenes:
            g = cv2.cvtColor(cv2.imread(str(G_FULL / f'{s}.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            gt = cv2.resize(g, target_size(*g.shape[1::-1]), interpolation=cv2.INTER_AREA)
            gt = to_landscape(gt)
            for ip in sorted((D_FULL / s).glob('*.png')):
                self.items.append((str(ip), gt))
        self.aug = aug
        if aug:
            random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, gt = self.items[i]
        raw = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        raw = cv2.resize(raw, target_size(*raw.shape[1::-1]), interpolation=cv2.INTER_AREA)
        raw = to_landscape(raw)
        if self.aug and random.random() < 0.5:
            raw = np.fliplr(raw).copy()
            gt = np.fliplr(gt).copy()
        # canonical 1024x768 canvas (reflect letterbox): all landscape targets fit (H<=768)
        H, W = 768, 1024
        raw = cv2.copyMakeBorder(raw, 0, H - raw.shape[0], 0, W - raw.shape[1], cv2.BORDER_REFLECT)
        gt = cv2.copyMakeBorder(gt, 0, H - gt.shape[0], 0, W - gt.shape[1], cv2.BORDER_REFLECT)
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
        return to_t(raw), to_t(gt)


class RestValDS(Dataset):
    """Per-item (batch=1 friendly): pad to /32, model, crop; metrics on target region only."""

    def __init__(self, scenes):
        self.items = []
        for s in scenes:
            g = cv2.cvtColor(cv2.imread(str(G_FULL / f'{s}.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            gt = cv2.resize(g, target_size(*g.shape[1::-1]), interpolation=cv2.INTER_AREA)
            gt = to_landscape(gt)
            for ip in sorted((D_FULL / s).glob('*.png')):
                self.items.append((str(ip), gt))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, gt = self.items[i]
        raw = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        raw = cv2.resize(raw, target_size(*raw.shape[1::-1]), interpolation=cv2.INTER_AREA)
        raw = to_landscape(raw)
        h, w = raw.shape[:2]
        H, W = (h + 31) // 32 * 32, (w + 31) // 32 * 32
        raw = cv2.copyMakeBorder(raw, 0, H - h, 0, W - w, cv2.BORDER_REFLECT)
        gt = cv2.copyMakeBorder(gt, 0, H - h, 0, W - w, cv2.BORDER_REFLECT)
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
        return to_t(raw), to_t(gt), h, w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--batch', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seg-ckpt', required=True)
    ap.add_argument('--seg-encoder', default='resnet34')
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--lam-v', type=float, default=0.0061)
    ap.add_argument('--lam-t', type=float, default=0.1881)
    ap.add_argument('--lam-e', type=float, default=0.1753)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()

    C.set_seed(0)
    torch.backends.cudnn.benchmark = True
    dev = 'cuda'
    scenes = sorted(d.name for d in D_FULL.iterdir() if d.is_dir())
    tr_scenes = [s for s in scenes if s not in VAL_SCENES]
    va_scenes = [s for s in scenes if s in VAL_SCENES]
    print(f'train {len(tr_scenes)} val {len(va_scenes)} @1024-native', flush=True)
    tr = DataLoader(RestDS(tr_scenes, aug=True), batch_size=a.batch, shuffle=True,
                    num_workers=10, pin_memory=True)
    va = DataLoader(RestValDS(va_scenes), batch_size=1, shuffle=False,
                    num_workers=6, pin_memory=True)

    backbone = smp.Unet(encoder_name=a.encoder, encoder_weights='imagenet', in_channels=3, classes=3)
    model = C.ResidualRestorer(backbone).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    ssim = C.SSIM().to(dev)
    vgg = C.VGGContent().to(dev)
    tdp = C.MultiTDP(a.seg_ckpt, a.seg_encoder, dev).to(dev)
    edge = C.SobelEdge().to(dev)
    lam_v, lam_t, lam_e = a.lam_v, a.lam_t, a.lam_e
    print(f'lam_v {lam_v} lam_t {lam_t} lam_e {lam_e} (v3-calibrated)', flush=True)

    start = 0
    best_psnr, best_ssim, best_miou = -1.0, -1.0, -1.0
    led_p, led_s, led_m = {}, {}, {}
    if a.resume and os.path.exists('rest1024_last.pth'):
        ck = torch.load('rest1024_last.pth', map_location=dev)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start = ck['epoch'] + 1
        best_psnr, best_ssim, best_miou = ck['best_psnr'], ck['best_ssim'], ck['best_miou']
        led_p, led_s, led_m = ck['led_p'], ck['led_s'], ck['led_m']
        print(f'resumed from epoch {start}', flush=True)

    for ep in range(start, a.epochs):
        model.train()
        tot, n, skipped = 0.0, 0, 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            o = model(x)
            loss = (C.charbonnier(o, y) + 0.5 * (1 - ssim(o, y)).mean()
                    + lam_v * vgg(o, y) + lam_t * tdp(o, y) + lam_e * edge(o, y))
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad()
                continue
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        model.eval()
        psnrs, ssims = [], []
        with torch.no_grad():
            for x, y, h, w in va:
                x = x.to(dev)
                o = model(x)[:, :, :h, :w]
                yt = y[:, :, :h, :w].to(dev)
                mse = F.mse_loss(o, yt, reduction='none').mean(dim=(1, 2, 3))
                psnrs.append(C.psnr_from_mse(mse.item()))
                ssims.append(ssim(o, yt).item())
        mpsnr, mssim = float(np.mean(psnrs)), float(np.mean(ssims))
        print(f'ep{ep} loss {tot / max(n, 1):.4f} (skip{skipped}) lr {sched.get_last_lr()[0]:.1e} '
              f'val PSNR {mpsnr:.2f} SSIM {mssim:.4f} bP {best_psnr:.2f} bS {best_ssim:.4f}', flush=True)
        torch.save({'epoch': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'best_psnr': best_psnr, 'best_ssim': best_ssim,
                    'led_p': led_p, 'led_s': led_s}, 'rest1024_last.pth')
        if mpsnr > best_psnr:
            best_psnr = mpsnr
            led_p = {'epoch': ep, 'psnr': mpsnr, 'ssim': mssim}
            torch.save({'model': model.state_dict(), **led_p}, 'rest1024_best_psnr.pth')
        if mssim > best_ssim:
            best_ssim = mssim
            led_s = {'epoch': ep, 'psnr': mpsnr, 'ssim': mssim}
            torch.save({'model': model.state_dict(), **led_s}, 'rest1024_best_ssim.pth')
    print(f'LEDGER psnr: {led_p}\nLEDGER ssim: {led_s}', flush=True)


if __name__ == '__main__':
    main()
