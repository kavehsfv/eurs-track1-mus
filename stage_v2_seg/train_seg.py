#!/usr/bin/env python3
"""Seg v2: smp-Unet resnet34, CB-weighted CE (CB-01) + Dice, per-sample difficulty gate
(MaskGuide-03 port). Mode A: raw only. Mode B: raw + restored + soft-mix (DTI-UIE Phase-2
recipe). Best-val-mIoU ckpt (DTI-UIE-12), cosine LR, resume support.
Skipped: focal (CB-CE+Dice suffices for v2), resampling (conflicts with CB per CB-01)."""
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

import v2_common as C

TRAIN_ROOT = Path('/home/kaveh/Workspace/Projects/Code/Challenges/MaCVi/Datasets/train')
D384 = TRAIN_ROOT / 'degraded_384'
M384 = TRAIN_ROOT / 'gt_masks_384'
G384 = TRAIN_ROOT / 'gt_images_384'
VAL_SCENES = {'fish1', 'fish5', 'ray1', 'turtles1'}
MEAN = np.array(C.IMAGENET_MEAN, dtype=np.float32)
STD = np.array(C.IMAGENET_STD, dtype=np.float32)
PAL = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], dtype=np.uint8)


def prep(img, flip, jb, jc):
    if flip:
        img = np.fliplr(img).copy()
    if jb != 0.0 or jc != 1.0:
        img = np.clip(img.astype(np.float32) * jc + jb * 255.0, 0, 255).astype(np.uint8)
    img = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(img).permute(2, 0, 1).float()


class SegDS(Dataset):
    def __init__(self, scenes, mode, restored_dir=None, aug=False):
        self.items = []
        for s in scenes:
            mask = cv2.imread(str(M384 / f'{s}.png'), cv2.IMREAD_UNCHANGED)
            gt = None
            if mode == 'B':
                gt = cv2.cvtColor(cv2.imread(str(G384 / f'{s}.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            for imgp in sorted((D384 / s).glob('*.png')):
                rp = str(restored_dir / s / imgp.name) if (mode == 'B' and restored_dir) else None
                self.items.append((str(imgp), rp, mask, gt))
        self.mode = mode
        self.aug = aug
        if aug:
            random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, rp, mask, gt = self.items[i]
        flip = self.aug and random.random() < 0.5
        jb = (random.random() - 0.5) * 0.2 if self.aug else 0.0
        jc = 1.0 + (random.random() - 0.5) * 0.2 if self.aug else 1.0
        raw = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        m = np.fliplr(mask).copy() if flip else mask
        out = [prep(raw, flip, jb, jc), torch.from_numpy(m.astype(np.int64))]
        if self.mode == 'B':
            rst = cv2.cvtColor(cv2.imread(rp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            out += [prep(rst, flip, jb, jc), prep(gt, flip, jb, jc)]
        return tuple(out)


class ValDS(Dataset):
    def __init__(self, scenes, restored_dir=None):
        self.items = []
        for s in scenes:
            mask = cv2.imread(str(M384 / f'{s}.png'), cv2.IMREAD_UNCHANGED)
            for imgp in sorted((D384 / s).glob('*.png')):
                ip = restored_dir / s / imgp.name if restored_dir else imgp
                self.items.append((str(ip), mask))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, m = self.items[i]
        img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(m.astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['A', 'B'], default='A')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--encoder', default='resnet34')
    ap.add_argument('--beta', type=float, default=0.999)
    ap.add_argument('--init', default=None)
    ap.add_argument('--restored-dir', default=None)
    ap.add_argument('--val-on', choices=['raw', 'restored'], default=None)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()

    C.set_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = a.tag or ('segA' if a.mode == 'A' else 'segB')
    batch = a.batch or (20 if a.mode == 'A' else 8)
    lr = a.lr or (5e-4 if a.mode == 'A' else 3e-4)
    val_on = a.val_on or ('raw' if a.mode == 'A' else 'restored')
    rdir = Path(a.restored_dir) if a.restored_dir else None
    if a.mode == 'B' and not rdir:
        raise SystemExit('mode B needs --restored-dir')
    if val_on == 'restored' and not rdir:
        raise SystemExit('val-on restored needs --restored-dir')

    scenes = sorted(d.name for d in D384.iterdir() if d.is_dir())
    tr_scenes = [s for s in scenes if s not in VAL_SCENES]
    va_scenes = [s for s in scenes if s in VAL_SCENES]
    print(f'train {len(tr_scenes)}: {tr_scenes}\nval {len(va_scenes)}: {va_scenes}', flush=True)

    counts = []
    for c in range(C.NCLS):
        n = 0
        for s in tr_scenes:
            m = cv2.imread(str(M384 / f'{s}.png'), cv2.IMREAD_UNCHANGED)
            if (m == c).any():
                n += len(list((D384 / s).glob('*.png')))
        counts.append(n)
    cbw = C.compute_cb_weights(counts, a.beta)
    print(f'class image-counts {counts} -> CB weights {np.round(cbw, 3)}', flush=True)

    tr = DataLoader(SegDS(tr_scenes, a.mode, rdir, aug=True), batch_size=batch, shuffle=True,
                    num_workers=8, pin_memory=True)
    va = DataLoader(ValDS(va_scenes, rdir if val_on == 'restored' else None), batch_size=16,
                    shuffle=False, num_workers=4, pin_memory=True)

    model = smp.Unet(encoder_name=a.encoder, encoder_weights='imagenet' if not a.init else None,
                     in_channels=3, classes=C.NCLS).to(dev)
    if a.init:
        sd = torch.load(a.init, map_location=dev)
        model.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd)
        print(f'init from {a.init}', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    ce_fn = nn.CrossEntropyLoss(weight=torch.tensor(cbw).to(dev), reduction='none')
    dice_fn = smp.losses.DiceLoss(mode='multiclass', classes=C.NCLS)

    start, best, best_ep = 0, -1.0, -1
    if a.resume and os.path.exists(f'{tag}_last.pth'):
        ck = torch.load(f'{tag}_last.pth', map_location=dev)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start, best, best_ep = ck['epoch'] + 1, ck['best'], ck['best_ep']
        print(f'resumed {tag} from epoch {start}, best {best:.4f} @ {best_ep}', flush=True)

    snap_idx = [0, len(va.dataset) // 3, 2 * len(va.dataset) // 3, len(va.dataset) - 1]
    os.makedirs('val_snaps', exist_ok=True)

    for ep in range(start, a.epochs):
        model.train()
        tot, n, skipped = 0.0, 0, 0
        for tup in tr:
            if a.mode == 'A':
                imgs = [tup[0].to(dev)]
                mask = tup[1].to(dev)
            else:
                imgs = [tup[0].to(dev), tup[2].to(dev)]
                mask = tup[1].to(dev)
                imgs.append(C.soft_mix(imgs[1], tup[3].to(dev)))
            opt.zero_grad()
            preds = [model(im) for im in imgs]
            ce_terms = []
            for pr in preds:
                ce_samp = ce_fn(pr, mask).mean(dim=(1, 2))
                w = C.sample_difficulty_weights(pr.detach(), mask)
                ce_terms.append((ce_samp * w).mean())
            loss_ce = sum(ce_terms) / len(ce_terms)
            loss_dice = dice_fn(torch.cat(preds), mask.repeat(len(preds), 1, 1))
            loss = loss_ce + loss_dice
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad()
                continue
            loss.backward()
            opt.step()
            tot += loss.item() * mask.size(0)
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
        print(f'ep{ep} loss {tot / max(n, 1):.4f} (skip{skipped}) lr {sched.get_last_lr()[0]:.1e} '
              f'val mIoU {miou:.4f} F1 {f1:.4f} [{pc}] best {best:.4f}@{best_ep}', flush=True)
        torch.save({'epoch': ep, 'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'best': best, 'best_ep': best_ep}, f'{tag}_last.pth')
        if miou > best:
            best, best_ep = miou, ep
            torch.save({'model': model.state_dict(), 'miou': miou, 'f1': f1,
                        'per': {c: per[c] for c in per}, 'epoch': ep}, f'{tag}_best.pth')
            rows = []
            for si in snap_idx:
                im, mk = va.dataset[si]
                with torch.no_grad():
                    pr = model(im[None].to(dev)).argmax(1)[0].cpu().numpy()
                rgb = ((im.permute(1, 2, 0).numpy() * STD) + MEAN)
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                rows.append(np.concatenate([rgb, PAL[pr], PAL[mk.numpy()]], axis=1))
            Image.fromarray(np.concatenate(rows, axis=0)).save(f'val_snaps/{tag}_best_ep{ep}.png')
    print(f'done {tag}: best mIoU {best:.4f} @ ep{best_ep}', flush=True)


if __name__ == '__main__':
    main()
