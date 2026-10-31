#!/usr/bin/env python3
"""Restorer v3: residual UNet-resnet34, Charbonnier + SSIM + VGG-conv5_2 + multi-layer TDP
(blocks[1]+blocks[-1], = DTI-UIE C5=up5) + unsigned SobelEdge. Auto-calibrated lambdas.
Triple ledger (PSNR/SSIM/downstream-mIoU); final = task-useful shortlist over the 3 saved bests."""
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
D384 = TRAIN_ROOT / 'degraded_384'
M384 = TRAIN_ROOT / 'gt_masks_384'
G384 = TRAIN_ROOT / 'gt_images_384'
VAL_SCENES = {'fish1', 'fish5', 'ray1', 'turtles1'}


class RestDS(Dataset):
    def __init__(self, scenes, aug=False):
        self.items = []
        for s in scenes:
            gt = cv2.cvtColor(cv2.imread(str(G384 / f'{s}.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            for imgp in sorted((D384 / s).glob('*.png')):
                self.items.append((str(imgp), gt))
        self.aug = aug
        if aug:
            random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, gt = self.items[i]
        raw = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if self.aug and random.random() < 0.5:
            raw = np.fliplr(raw).copy()
            gt = np.fliplr(gt).copy()
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
        return to_t(raw), to_t(gt)


class RestValDS(Dataset):
    def __init__(self, scenes):
        self.items = []
        self.masks = {s: cv2.imread(str(M384 / f'{s}.png'), cv2.IMREAD_UNCHANGED) for s in scenes}
        for s in scenes:
            gt = cv2.cvtColor(cv2.imread(str(G384 / f'{s}.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            for imgp in sorted((D384 / s).glob('*.png')):
                self.items.append((str(imgp), gt, s))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, gt, s = self.items[i]
        raw = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
        return to_t(raw), to_t(gt), torch.from_numpy(self.masks[s].astype(np.int64))


def load_seg(ckpt, encoder, dev):
    m = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=C.NCLS)
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seg-ckpt', required=True)
    ap.add_argument('--seg-encoder', default='resnet34')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    C.set_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    scenes = sorted(d.name for d in D384.iterdir() if d.is_dir())
    tr_scenes = [s for s in scenes if s not in VAL_SCENES]
    va_scenes = [s for s in scenes if s in VAL_SCENES]
    print(f'train {len(tr_scenes)} val {len(va_scenes)} seg {a.seg_ckpt}', flush=True)
    tr = DataLoader(RestDS(tr_scenes, aug=True), batch_size=a.batch, shuffle=True,
                    num_workers=8, pin_memory=True)
    va = DataLoader(RestValDS(va_scenes), batch_size=16, shuffle=False,
                    num_workers=4, pin_memory=True)

    backbone = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=3)
    model = C.ResidualRestorer(backbone).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    ssim = C.SSIM().to(dev)
    vgg = C.VGGContent().to(dev)
    tdp = C.MultiTDP(a.seg_ckpt, a.seg_encoder, dev).to(dev)
    edge = C.SobelEdge().to(dev)
    seg = load_seg(a.seg_ckpt, a.seg_encoder, dev)
    smean = torch.tensor(C.IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    sstd = torch.tensor(C.IMAGENET_STD, device=dev).view(1, 3, 1, 1)

    lam_v = lam_t = lam_e = None
    start = 0
    best_psnr, best_ssim, best_miou = -1.0, -1.0, -1.0
    led_p, led_s, led_m = {}, {}, {}
    if a.resume and os.path.exists('restorer_last.pth'):
        ck = torch.load('restorer_last.pth', map_location=dev)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start = ck['epoch'] + 1
        best_psnr, best_ssim, best_miou = ck['best_psnr'], ck['best_ssim'], ck['best_miou']
        led_p, led_s, led_m = ck['led_p'], ck['led_s'], ck['led_m']
        lam_v, lam_t, lam_e = ck['lam_v'], ck['lam_t'], ck['lam_e']
        print(f'resumed from epoch {start}', flush=True)

    if lam_v is None:  # auto-calibrate: VGG@0.43x, TDP@0.5x, Edge@0.3x of Charbonnier
        model.eval()
        acc = np.zeros(5)
        nb = 0
        with torch.no_grad():
            for x, y in tr:
                if nb >= 20:
                    break
                x, y = x.to(dev), y.to(dev)
                o = model(x)
                acc[0] += C.charbonnier(o, y).item()
                acc[1] += (1 - ssim(o, y)).mean().item()
                acc[2] += vgg(o, y).item()
                acc[3] += tdp(o, y).item()
                acc[4] += edge(o, y).item()
                nb += 1
        acc /= nb
        lam_v = float(np.clip(0.43 * acc[0] / max(acc[2], 1e-12), 1e-4, 1e3))
        lam_t = float(np.clip(0.50 * acc[0] / max(acc[3], 1e-12), 1e-4, 1e3))
        lam_e = float(np.clip(0.30 * acc[0] / max(acc[4], 1e-12), 1e-4, 1e3))
        print(f'calib Charb {acc[0]:.4f} SSIMt {acc[1]:.4f} VGG {acc[2]:.4f} TDP {acc[3]:.4f} Edge {acc[4]:.4f} '
              f'-> lam_v {lam_v:.4f} lam_t {lam_t:.4f} lam_e {lam_e:.4f}', flush=True)

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
        psnrs, ssims, ps, ts = [], [], [], []
        with torch.no_grad():
            for x, y, mk in va:
                x, y = x.to(dev), y.to(dev)
                o = model(x)
                mse = F.mse_loss(o, y, reduction='none').mean(dim=(1, 2, 3))
                psnrs += [C.psnr_from_mse(v) for v in mse.cpu().tolist()]
                ssims += ssim(o, y).cpu().tolist()
                ps.append(seg((o - smean) / sstd).argmax(1).cpu())
                ts.append(mk)
        p = torch.cat([v.reshape(-1) for v in ps]).numpy()
        t = torch.cat([v.reshape(-1) for v in ts]).numpy()
        _, dmiou, _ = C.per_class_iou_f1(p, t)
        mpsnr, mssim = float(np.mean(psnrs)), float(np.mean(ssims))
        print(f'ep{ep} loss {tot / max(n, 1):.4f} (skip{skipped}) lr {sched.get_last_lr()[0]:.1e} '
              f'val PSNR {mpsnr:.2f} SSIM {mssim:.4f} d-mIoU {dmiou:.4f} '
              f'bP {best_psnr:.2f} bS {best_ssim:.4f} bM {best_miou:.4f}', flush=True)
        torch.save({'epoch': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'best_psnr': best_psnr, 'best_ssim': best_ssim,
                    'best_miou': best_miou, 'led_p': led_p, 'led_s': led_s, 'led_m': led_m,
                    'lam_v': lam_v, 'lam_t': lam_t, 'lam_e': lam_e}, 'restorer_last.pth')
        if mpsnr > best_psnr:
            best_psnr = mpsnr
            led_p = {'epoch': ep, 'psnr': mpsnr, 'ssim': mssim, 'dmiou': dmiou}
            torch.save({'model': model.state_dict(), **led_p}, 'restorer_best_psnr.pth')
        if mssim > best_ssim:
            best_ssim = mssim
            led_s = {'epoch': ep, 'psnr': mpsnr, 'ssim': mssim, 'dmiou': dmiou}
            torch.save({'model': model.state_dict(), **led_s}, 'restorer_best_ssim.pth')
        if dmiou > best_miou:
            best_miou = dmiou
            led_m = {'epoch': ep, 'psnr': mpsnr, 'ssim': mssim, 'dmiou': dmiou}
            torch.save({'model': model.state_dict(), **led_m}, 'restorer_best_miou.pth')

    print(f'LEDGER psnr: {led_p}\nLEDGER ssim: {led_s}\nLEDGER miou: {led_m}', flush=True)
    if led_m['psnr'] >= led_p['psnr'] - 0.5 and led_m['ssim'] >= led_s['ssim'] - 0.005:
        src, why = 'restorer_best_miou.pth', 'task-useful in shortlist'
    elif led_s['psnr'] >= led_p['psnr'] - 0.5:
        src, why = 'restorer_best_ssim.pth', 'ssim-best in shortlist'
    else:
        src, why = 'restorer_best_psnr.pth', 'fidelity wins'
    sd = torch.load(src, map_location='cpu')
    torch.save({'model': sd['model']}, 'restorer_final.pth')
    print(f'FINAL: {src} -> restorer_final.pth ({why})', flush=True)


if __name__ == '__main__':
    main()
