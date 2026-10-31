#!/usr/bin/env python3
"""Restorer v2: residual UNet-resnet18 (DTI-UIE-13), L1 + SSIM + VGG-conv5_2 (FUnIE-02/03)
+ TDP from frozen Seg-A (DTI-UIE-05 lite; no warmup gate: seg pre-trained). Lambdas
auto-calibrated at start (scale-free). Dual-criterion ckpt (DTI-UIE-12: best-PSNR +
best-downstream-mIoU) with full ledgers; final auto-selected (task-useful wins ties).
Skipped: GAN term (hurts PSNR; FUnIE ablation shows L1 does the work)."""
import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

import v2_common as C

TRAIN_ROOT = Path('/home/kaveh/Workspace/Projects/Code/Challenges/MaCVi/Datasets/train')
D384 = TRAIN_ROOT / 'degraded_384'
M384 = TRAIN_ROOT / 'gt_masks_384'
G384 = TRAIN_ROOT / 'gt_images_384'
VAL_SCENES = {'fish1', 'fish5', 'ray1', 'turtles1'}
MEAN = np.array(C.IMAGENET_MEAN, dtype=np.float32)
STD = np.array(C.IMAGENET_STD, dtype=np.float32)


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


class RestValDS(Dataset):  # val with cached scene masks for downstream-mIoU
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
    print(f'train {len(tr_scenes)} val {len(va_scenes)}', flush=True)
    tr = DataLoader(RestDS(tr_scenes, aug=True), batch_size=a.batch, shuffle=True,
                    num_workers=8, pin_memory=True)
    va = DataLoader(RestValDS(va_scenes), batch_size=16, shuffle=False,
                    num_workers=4, pin_memory=True)

    backbone = smp.Unet(encoder_name='resnet18', encoder_weights='imagenet', in_channels=3, classes=3)
    model = C.ResidualRestorer(backbone).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    ssim = C.SSIM().to(dev)
    vgg = C.VGGContent().to(dev)
    tdp = C.TDPExtractor(a.seg_ckpt, a.seg_encoder, dev).to(dev)
    segA = load_seg(a.seg_ckpt, a.seg_encoder, dev)
    smean = torch.tensor(C.IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    sstd = torch.tensor(C.IMAGENET_STD, device=dev).view(1, 3, 1, 1)

    lam_v = lam_t = None
    start, best_psnr, best_miou = 0, -1.0, -1.0
    led_psnr, led_miou = {}, {}
    if a.resume and os.path.exists('restorer_last.pth'):
        ck = torch.load('restorer_last.pth', map_location=dev)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start = ck['epoch'] + 1
        best_psnr, best_miou = ck['best_psnr'], ck['best_miou']
        led_psnr, led_miou = ck['led_psnr'], ck['led_miou']
        lam_v, lam_t = ck['lam_v'], ck['lam_t']
        print(f'resumed from epoch {start} lam_v {lam_v:.4f} lam_t {lam_t:.4f}', flush=True)

    if lam_v is None:  # auto-calibrate lambdas: VGG @0.43xL1 (FUnIE 7:3), TDP @0.5xL1
        model.eval()
        acc = np.zeros(4)
        nb = 0
        with torch.no_grad():
            for x, y in tr:
                if nb >= 20:
                    break
                x, y = x.to(dev), y.to(dev)
                o = model(x)
                acc[0] += F.l1_loss(o, y).item()
                acc[1] += (1 - ssim(o, y)).mean().item()
                acc[2] += vgg(o, y).item()
                acc[3] += tdp(o, y).item()
                nb += 1
        acc /= nb
        lam_v = float(np.clip(0.43 * acc[0] / max(acc[2], 1e-12), 1e-4, 1e3))
        lam_t = float(np.clip(0.50 * acc[0] / max(acc[3], 1e-12), 1e-4, 1e3))
        print(f'calib L1 {acc[0]:.4f} SSIMt {acc[1]:.4f} VGG {acc[2]:.6f} TDP {acc[3]:.6f} '
              f'-> lam_v {lam_v:.4f} lam_t {lam_t:.4f}', flush=True)

    for ep in range(start, a.epochs):
        model.train()
        tot, n, skipped = 0.0, 0, 0
        for x, y in tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            o = model(x)
            loss = (F.l1_loss(o, y) + 0.5 * (1 - ssim(o, y)).mean()
                    + lam_v * vgg(o, y) + lam_t * tdp(o, y))
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
                ps.append(segA((o - smean) / sstd).argmax(1).cpu())
                ts.append(mk)
        p = torch.cat([v.reshape(-1) for v in ps]).numpy()
        t = torch.cat([v.reshape(-1) for v in ts]).numpy()
        _, dmiou, _ = C.per_class_iou_f1(p, t)
        mpsnr, mssim = float(np.mean(psnrs)), float(np.mean(ssims))
        print(f'ep{ep} loss {tot / max(n, 1):.4f} (skip{skipped}) lr {sched.get_last_lr()[0]:.1e} '
              f'val PSNR {mpsnr:.2f} SSIM {mssim:.4f} downstream-mIoU {dmiou:.4f} '
              f'bestP {best_psnr:.2f} bestM {best_miou:.4f}', flush=True)
        state = {'epoch': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                 'sched': sched.state_dict(), 'best_psnr': best_psnr, 'best_miou': best_miou,
                 'led_psnr': led_psnr, 'led_miou': led_miou, 'lam_v': lam_v, 'lam_t': lam_t}
        torch.save(state, 'restorer_last.pth')
        if mpsnr > best_psnr:
            best_psnr = mpsnr
            led_psnr = {'epoch': ep, 'psnr': mpsnr, 'ssim': mssim, 'dmiou': dmiou}
            torch.save({'model': model.state_dict(), **led_psnr}, 'restorer_best_psnr.pth')
        if dmiou > best_miou:
            best_miou = dmiou
            led_miou = {'epoch': ep, 'psnr': mpsnr, 'ssim': mssim, 'dmiou': dmiou}
            torch.save({'model': model.state_dict(), **led_miou}, 'restorer_best_miou.pth')

    print(f'LEDGER best-PSNR: {led_psnr}\nLEDGER best-mIoU: {led_miou}', flush=True)
    use_miou = led_miou and (led_psnr['psnr'] - led_miou['psnr'] <= 0.5)
    src = 'restorer_best_miou.pth' if use_miou else 'restorer_best_psnr.pth'
    sd = torch.load(src, map_location='cpu')
    torch.save({'model': sd['model']}, 'restorer_final.pth')
    print(f'FINAL: {src} -> restorer_final.pth '
          f'({"task-useful wins tie" if use_miou else "fidelity wins"})', flush=True)


if __name__ == '__main__':
    main()
