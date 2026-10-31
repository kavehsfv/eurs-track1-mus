"""Shared utilities for Track 1 v2 (DTI-UIE-lite + CB loss + FUnIE losses + MaskGuide gate)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # v1 values, kept for consistency
IMAGENET_STD = [0.229, 0.224, 0.225]
NCLS = 5
NET_SIZE = (768, 384)  # (W, H) — matches Datasets/*_384


def set_seed(s=0):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def compute_cb_weights(counts, beta=0.999):
    """CB-01: w=(1-b)/(1-b^n), sum-to-C normalized. counts: per-class image counts."""
    counts = np.asarray(counts, dtype=np.float64)
    w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    return (w / w.sum() * len(counts)).astype(np.float32)


def per_class_iou_f1(p, t, ncls=NCLS):
    """p, t: flat numpy arrays of class ids. Returns (dict c->(iou,f1), macro_iou, macro_f1)."""
    per, ious, f1s = {}, [], []
    for c in range(ncls):
        pc, tc = (p == c), (t == c)
        inter = int((pc & tc).sum())
        union = int((pc | tc).sum())
        if union == 0:
            per[c] = (float('nan'), float('nan'))
            continue
        iou = inter / union
        prec = inter / (int(pc.sum()) + 1e-9)
        rec = inter / (int(tc.sum()) + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        per[c] = (iou, f1)
        ious.append(iou)
        f1s.append(f1)
    mi = float(np.mean(ious)) if ious else 0.0
    mf = float(np.mean(f1s)) if f1s else 0.0
    return per, mi, mf


def psnr_from_mse(mse, dr=1.0):
    return 20 * math.log10(dr) - 10 * math.log10(mse + 1e-12)


class SSIM(nn.Module):
    """Gaussian-window SSIM, differentiable. forward -> per-image SSIM (B,)."""
    def __init__(self, window=11, sigma=1.5):
        super().__init__()
        g = torch.exp(-(torch.arange(window) - window // 2) ** 2 / (2 * sigma ** 2))
        g = (g[:, None] * g[None, :])
        g = g / g.sum()
        self.register_buffer('w', g[None, None].repeat(3, 1, 1, 1))
        self.pad = window // 2

    def forward(self, x, y):
        mu_x = F.conv2d(x, self.w, padding=self.pad, groups=3)
        mu_y = F.conv2d(y, self.w, padding=self.pad, groups=3)
        s_xx = F.conv2d(x * x, self.w, padding=self.pad, groups=3) - mu_x ** 2
        s_yy = F.conv2d(y * y, self.w, padding=self.pad, groups=3) - mu_y ** 2
        s_xy = F.conv2d(x * y, self.w, padding=self.pad, groups=3) - mu_x * mu_y
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        m = ((2 * mu_x * mu_y + c1) * (2 * s_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (s_xx + s_yy + c2))
        return m.mean(dim=(1, 2, 3))


class VGGContent(nn.Module):
    """FUnIE-03: frozen VGG19 conv5_2 (features idx 30, pre-ReLU) MSE. Inputs [0,1] RGB,
    ImageNet-normalized inside (standard perceptual contract; deviates from FUnIE's no-renorm quirk)."""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        v = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:31].eval()
        for p in v.parameters():
            p.requires_grad_(False)
        self.v = v
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, pred, true):
        fp = self.v((pred - self.mean) / self.std)
        with torch.no_grad():
            ft = self.v((true - self.mean) / self.std)
        return F.mse_loss(fp, ft)


class TDPExtractor(nn.Module):
    """DTI-UIE-05 lite: L1 on frozen Seg-A decoder block[1] (~1/8 scale, = their C5 scale).
    No warmup gate needed: the seg net is pre-trained, not co-trained. Inputs [0,1] RGB."""
    def __init__(self, ckpt, encoder='resnet34', device='cuda'):
        super().__init__()
        import segmentation_models_pytorch as smp
        m = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=NCLS)
        sd = torch.load(ckpt, map_location='cpu')
        m.load_state_dict(sd['model'] if isinstance(sd, dict) and 'model' in sd else sd)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.m = m.to(device)
        self._feat = None
        self.m.decoder.blocks[1].register_forward_hook(lambda mo, i, o: setattr(self, '_feat', o))
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        with torch.no_grad():  # fail-fast: verify hook path
            d = torch.zeros(1, 3, 384, 768, device=device)
            self.m((d - self.mean.to(device)) / self.std.to(device))
        assert self._feat is not None, 'TDP hook failed: no feature captured'
        print(f'TDP extractor ready: {ckpt} feats={tuple(self._feat.shape)}', flush=True)

    def _feats(self, x):
        self.m((x - self.mean) / self.std)
        return self._feat

    def forward(self, pred, true):
        fp = self._feats(pred)
        with torch.no_grad():
            ft = self._feats(true)
        return F.l1_loss(fp, ft)


def soft_mix(enh, gt):
    """DTI-UIE-06: sigmoid(randn 8x8)->bilinear-up soft blend. Affine-safe on normalized tensors."""
    m = torch.sigmoid(torch.randn(enh.size(0), 1, 8, 8, device=enh.device))
    m = F.interpolate(m, size=enh.shape[-2:], mode='bilinear', align_corners=False)
    return m * enh + (1 - m) * gt


class ResidualRestorer(nn.Module):
    """DTI-UIE-13: residual-from-identity output: out = clamp(x + std*net(norm(x))). Near-identity init."""
    def __init__(self, backbone):
        super().__init__()
        self.net = backbone
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return torch.clamp(x + self.net((x - self.mean) / self.std) * self.std, 0, 1)


def sample_difficulty_weights(preds, targets):
    """MaskGuide-03 port (continuous): w = 0.5 + (1 - sample macro-IoU over present classes). Bounded [0.5, 1.5]."""
    with torch.no_grad():
        pm = preds.argmax(1)
        w = torch.empty(preds.size(0), device=preds.device)
        for i in range(preds.size(0)):
            p, t = pm[i], targets[i]
            ious = []
            for c in torch.unique(t).tolist():
                pc, tc = (p == c), (t == c)
                u = int((pc | tc).sum().item())
                if u > 0:
                    ious.append(float((pc & tc).sum().item()) / u)
            miou = sum(ious) / len(ious) if ious else 1.0
            w[i] = 0.5 + (1.0 - miou)
    return w
