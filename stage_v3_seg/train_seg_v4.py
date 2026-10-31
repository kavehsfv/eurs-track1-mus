#!/usr/bin/env python3
"""Seg-SD (v3): self-distillation. Fresh student (smp-Unet resnet34, raw inputs) trained with
CB-CE + Dice + KD from frozen Seg-A (MaskGuide-01 annealed T 4.0->0.8, MaskGuide-02 manual KL*T^2).
KD-as-denoiser: teacher soft targets supervise where shared-mask GT is wrong (moving classes).
Gate: best val must beat Seg-A 0.3052 or Seg-A ships."""
import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from PIL import Image

import v3_common as C  # from baseline_v3

TRAIN_ROOT = Path('/home/kaveh/Workspace/Projects/Code/Challenges/MaCVi/Datasets/train')
D384 = TRAIN_ROOT / 'degraded_384'
M384 = TRAIN_ROOT / 'gt_masks_384'
VAL_SCENES = {'fish1', 'fish5', 'ray1', 'turtles1'}
MEAN = np.array(C.IMAGENET_MEAN, dtype=np.float32)
STD = np.array(C.IMAGENET_STD, dtype=np.float32)
PAL = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.uint8)
TEACHER_DEFAULT = '../baseline_v2/segA_best.pth'


def prep(img, flip, jb, jc):
    if flip:
        img = np.fliplr(img).copy()
    if jb != 0.0 or jc != 1.0:
        img = np.clip(img.astype(np.float32) * jc + jb * 255.0, 0, 255).astype(np.uint8)
    img = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(img).permute(2, 0, 1).float()


class SegDS(Dataset):
    def __init__(self, scenes, aug=False):
        self.items = []
        for s in scenes:
            mask = cv2.imread(str(M384 / f'{s}.png'), cv2.IMREAD_UNCHANGED)
            for imgp in sorted((D384 / s).glob('*.png')):
                self.items.append((str(imgp), mask))
        self.aug = aug
        if aug:
            random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, mask = self.items[i]
        flip = self.aug and random.random() < 0.5
        jb = (random.random() - 0.5) * 0.2 if self.aug else 0.0
        jc = 1.0 + (random.random() - 0.5) * 0.2 if self.aug else 1.0
        raw = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        m = np.fliplr(mask).copy() if flip else mask
        return prep(raw, flip, jb, jc), torch.from_numpy(m.astype(np.int64))


class ValDS(Dataset):
    def __init__(self, scenes):
        self.items = []
        for s in scenes:
            mask = cv2.imread(str(M384 / f'{s}.png'), cv2.IMREAD_UNCHANGED)
            for imgp in sorted((D384 / s).glob('*.png')):
                self.items.append((str(imgp), mask))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, m = self.items[i]
        img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(m.astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=20)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--beta', type=float, default=0.999)
    ap.add_argument('--teacher', default=TEACHER_DEFAULT)
    ap.add_argument('--init', default=None, help='warm-start ckpt (e.g. suim_r34.pth); head reset')
    ap.add_argument('--lam-kd', type=float, default=1.0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    C.set_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    scenes = sorted(d.name for d in D384.iterdir() if d.is_dir())
    tr_scenes = [s for s in scenes if s not in VAL_SCENES]
    va_scenes = [s for s in scenes if s in VAL_SCENES]
    print(f'train {len(tr_scenes)} val {len(va_scenes)} teacher {a.teacher}', flush=True)

    counts = []
    for c in range(C.NCLS):
        n = 0
        for s in tr_scenes:
            m = cv2.imread(str(M384 / f'{s}.png'), cv2.IMREAD_UNCHANGED)
            if (m == c).any():
                n += len(list((D384 / s).glob('*.png')))
        counts.append(n)
    cbw = C.compute_cb_weights(counts, a.beta)
    print(f'CB weights {np.round(cbw, 3)}', flush=True)

    tr = DataLoader(SegDS(tr_scenes, aug=True), batch_size=a.batch, shuffle=True,
                    num_workers=8, pin_memory=True)
    va = DataLoader(ValDS(va_scenes), batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model = smp.Unet(encoder_name=a.encoder,
                     encoder_weights=None if a.init else 'imagenet',
                     in_channels=3, classes=C.NCLS).to(dev)
    if a.init:
        sd = torch.load(a.init, map_location='cpu')
        sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
        own = model.state_dict()
        loaded = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        own.update(loaded)
        model.load_state_dict(own)
        print(f'warm-start {a.init}: {len(loaded)}/{len(own)} tensors (head reset)', flush=True)
    teacher = smp.Unet(encoder_name=a.encoder, encoder_weights=None, in_channels=3, classes=C.NCLS)
    sd = torch.load(a.teacher, map_location='cpu')
    teacher.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher = teacher.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    ce_fn = nn.CrossEntropyLoss(weight=torch.tensor(cbw).to(dev), reduction='none')
    dice_fn = smp.losses.DiceLoss(mode='multiclass', classes=C.NCLS)

    max_steps = a.epochs * len(tr)
    start, best, best_ep, step = 0, -1.0, -1, 0
    if a.resume and os.path.exists('segSD_last.pth'):
        ck = torch.load('segSD_last.pth', map_location=dev)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start, best, best_ep, step = ck['epoch'] + 1, ck['best'], ck['best_ep'], ck['step']
        print(f'resumed from epoch {start}, best {best:.4f} @ {best_ep}', flush=True)

    snap_idx = [0, len(va.dataset) // 3, 2 * len(va.dataset) // 3, len(va.dataset) - 1]
    os.makedirs('val_snaps', exist_ok=True)

    for ep in range(start, a.epochs):
        model.train()
        tot, tkd, n, skipped = 0.0, 0.0, 0, 0
        for im, mask in tr:
            im, mask = im.to(dev), mask.to(dev)
            T = C.annealed_T(step, max_steps)
            opt.zero_grad()
            pr = model(im)
            with torch.no_grad():
                t_log = teacher(im)
            ce_samp = ce_fn(pr, mask).mean(dim=(1, 2))
            w = C.sample_difficulty_weights(pr.detach(), mask)
            loss_ce = (ce_samp * w).mean()
            loss_dice = dice_fn(pr, mask)
            loss_kd = C.kd_loss(pr, t_log, T)
            loss = loss_ce + loss_dice + a.lam_kd * loss_kd
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad()
                step += 1
                continue
            loss.backward()
            opt.step()
            step += 1
            tot += loss.item() * mask.size(0)
            tkd += loss_kd.item() * mask.size(0)
            n += mask.size(0)
        sched.step()
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for im, mk in va:
                ps.append(model(im.to(dev)).argmax(1).cpu())
                ts.append(mk)
        p = torch.cat([x.reshape(-1) for x in ps]).numpy()
        t = torch.cat([x.reshape(-1) for x in ts]).numpy()
        per, miou, f1 = C.per_class_iou_f1(p, t)
        pc = ' '.join(f'{c}:{per[c][0]:.3f}' for c in range(C.NCLS))
        print(f'ep{ep} loss {tot / max(n, 1):.4f} kd {tkd / max(n, 1):.4f} T {T:.2f} (skip{skipped}) '
              f'lr {sched.get_last_lr()[0]:.1e} val mIoU {miou:.4f} F1 {f1:.4f} [{pc}] best {best:.4f}@{best_ep}',
              flush=True)
        torch.save({'epoch': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'best': best, 'best_ep': best_ep, 'step': step},
                   'segSD_last.pth')
        if miou > best:
            best, best_ep = miou, ep
            torch.save({'model': model.state_dict(), 'miou': miou, 'f1': f1,
                        'per': {c: per[c] for c in per}, 'epoch': ep}, 'segSD_best.pth')
            rows = []
            for si in snap_idx:
                im, mk = va.dataset[si]
                with torch.no_grad():
                    pr = model(im[None].to(dev)).argmax(1)[0].cpu().numpy()
                rgb = ((im.permute(1, 2, 0).numpy() * STD) + MEAN)
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                rows.append(np.concatenate([rgb, PAL[pr], PAL[mk.numpy()]], axis=1))
            Image.fromarray(np.concatenate(rows, axis=0)).save(f'val_snaps/segSD_best_ep{ep}.png')
    print(f'done segSD: best mIoU {best:.4f} @ ep{best_ep} (gate: beat Seg-A 0.3052)', flush=True)


if __name__ == '__main__':
    main()
