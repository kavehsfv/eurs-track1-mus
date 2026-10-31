#!/usr/bin/env python3
"""SUIM pretraining for seg encoder+decoder (8-class, native 480x640). External-data stage
for v4: breaks shared-mask ceiling via real per-frame underwater labels (disclose in submission).
Output: suim_r34.pth (full Unet state; EURS finetune resets head)."""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

SUIM = Path('/home/kaveh/Workspace/Projects/Code/Challenges/MaCVi/Datasets/SUIM/SUIM/train_val')
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NCLS = 8


class SuimDS(Dataset):
    def __init__(self, aug=False):
        self.imgs = sorted((SUIM / 'images').glob('*.jpg'))
        self.masks = {m.stem: m for m in (SUIM / 'masks').glob('*.bmp')}
        self.aug = aug

    def __getitem__(self, i):
        ip = self.imgs[i]
        mp = self.masks[ip.stem]
        raw = cv2.cvtColor(cv2.imread(str(ip), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        raw = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (640, 480), interpolation=cv2.INTER_NEAREST)
        if self.aug and random.random() < 0.5:
            raw = np.fliplr(raw).copy()
            m = np.fliplr(m).copy()
        x = torch.from_numpy((raw.astype(np.float32) / 255.0 - MEAN) / STD).permute(2, 0, 1)
        return x.float(), torch.from_numpy(m.astype(np.int64)).clamp(0, NCLS - 1)
    def __len__(self):
        return len(self.imgs)


def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = True
    dev = 'cuda'
    ds = SuimDS(aug=True)
    # verify pairing on one sample
    x, m = ds[0]
    print(f'sample: img {tuple(x.shape)} mask {tuple(m.shape)} uniq {torch.unique(m).tolist()}', flush=True)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    model = smp.Unet('resnet34', encoder_weights='imagenet', in_channels=3, classes=NCLS).to(dev)
    ce = nn.CrossEntropyLoss()
    dice = smp.losses.DiceLoss(mode='multiclass', classes=NCLS)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
    for ep in range(15):
        model.train()
        tot, n = 0.0, 0
        for x, m in dl:
            x, m = x.to(dev), m.to(dev)
            opt.zero_grad()
            pr = model(x)
            loss = ce(pr, m) + dice(pr, m)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        print(f'ep{ep} loss {tot / n:.4f}', flush=True)
    torch.save({'model': model.state_dict()}, 'suim_r34.pth')
    print('saved suim_r34.pth', flush=True)


if __name__ == '__main__':
    main()
