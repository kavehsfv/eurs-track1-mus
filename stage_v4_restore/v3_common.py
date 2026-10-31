"""Shared utilities for Track 1 v3 (Seg self-distillation + stronger restorer + local NR metrics)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NCLS = 5
NET_SIZE = (768, 384)  # (W, H)


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
    """FUnIE-03: frozen VGG19 conv5_2 MSE. Inputs [0,1] RGB, normalized inside."""
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


def charbonnier(pred, target, eps=1e-6):
    """Smooth-L1 pixel loss (DTI-UIE loss.py; mean-reduced, eps=1e-6 for fp32 safety)."""
    return torch.sqrt((pred - target) ** 2 + eps).mean()


_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)


class SobelEdge(nn.Module):
    """Unsigned Sobel edge loss (DTI-UIE SobelEdgeLoss, vectorized; mean-reduced)."""
    def __init__(self):
        super().__init__()
        self.register_buffer('kx', _SOBEL_X.repeat(3, 1, 1, 1))
        self.register_buffer('ky', _SOBEL_Y.repeat(3, 1, 1, 1))

    def forward(self, pred, true):
        gh_p = F.conv2d(pred, self.kx, padding=1, groups=3)
        gv_p = F.conv2d(pred, self.ky, padding=1, groups=3)
        gh_t = F.conv2d(true, self.kx, padding=1, groups=3)
        gv_t = F.conv2d(true, self.ky, padding=1, groups=3)
        return (gh_p - gh_t).abs().mean() + (gv_p - gv_t).abs().mean()


class MultiTDP(nn.Module):
    """DTI-UIE FeatureLoss-lite: L1 on frozen seg decoder blocks[1] (1/8) + blocks[-1] (full-res,
    = their C5=up5). Inputs [0,1] RGB."""
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
        self._f1 = self._fN = None
        self.m.decoder.blocks[1].register_forward_hook(lambda mo, i, o: setattr(self, '_f1', o))
        self.m.decoder.blocks[-1].register_forward_hook(lambda mo, i, o: setattr(self, '_fN', o))
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        with torch.no_grad():
            d = torch.zeros(1, 3, 384, 768, device=device)
            self.m((d - self.mean.to(device)) / self.std.to(device))
        assert self._f1 is not None and self._fN is not None, 'MultiTDP hook failed'
        print(f'MultiTDP ready: {ckpt} f1={tuple(self._f1.shape)} fN={tuple(self._fN.shape)}', flush=True)

    def _feats(self, x):
        self.m((x - self.mean) / self.std)
        return self._f1, self._fN

    def forward(self, pred, true):
        fp1, fpN = self._feats(pred)
        with torch.no_grad():
            ft1, ftN = self._feats(true)
        return F.l1_loss(fp1, ft1) + F.l1_loss(fpN, ftN)


def kd_loss(s_log, t_log, T):
    """MaskGuide-02 form on class logits: manual KL(teacher||student) * T^2, 1e-7 clamped."""
    s = F.softmax(s_log / T, dim=1).clamp(1e-7, 1 - 1e-7)
    t = F.softmax(t_log / T, dim=1).clamp(1e-7, 1 - 1e-7)
    return (t * (t / s).log()).sum(1).mean() * T * T


def annealed_T(step, max_steps):
    """MaskGuide-01: cosine 4.0 -> 0.8 over training steps, clamped."""
    p = min(max(step / max(max_steps, 1), 0.0), 1.0)
    return min(4.0, max(0.8, 0.8 + 3.2 * (1 + math.cos(math.pi * p)) / 2))


class ResidualRestorer(nn.Module):
    """DTI-UIE-13 residual-from-identity: out = clamp(x + std*net(norm(x)))."""
    def __init__(self, backbone):
        super().__init__()
        self.net = backbone
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return torch.clamp(x + self.net((x - self.mean) / self.std) * self.std, 0, 1)


def sample_difficulty_weights(preds, targets):
    """MaskGuide-03 port (continuous): w = 0.5 + (1 - sample macro-IoU over present classes)."""
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


# ---------------- UIQM / UCIQE (exact DTI-UIE uie_eval.py ports, in-place bug fixed) ----------------
def _mu_a(x, aL=0.1, aR=0.1):
    x = x.sort()[0]
    K = len(x)
    TaL, TaR = math.ceil(aL * K), math.floor(aR * K)
    return x[TaL + 1:K - TaR].sum() * (1.0 / (K - TaL - TaR))


def _s_a(x, mu):
    return ((x - mu) ** 2).sum() / len(x)


def _uicm(x):  # x: CHW float [0,255]
    R, G, B = x[0].flatten(), x[1].flatten(), x[2].flatten()
    RG, YB = R - G, (R + G) / 2 - B
    mRG, mYB = _mu_a(RG), _mu_a(YB)
    l = torch.sqrt(mRG ** 2 + mYB ** 2)
    r = torch.sqrt(_s_a(RG, mRG) + _s_a(YB, mYB))
    return -0.0268 * l + 0.1586 * r


def _sobel_mag(ch):  # ch: HW
    kx, ky = _SOBEL_X.to(ch.device), _SOBEL_Y.to(ch.device)
    mag = torch.hypot(F.conv2d(ch[None, None], kx, padding=1), F.conv2d(ch[None, None], ky, padding=1))
    return (mag * 255.0 / mag.max()).squeeze()


def _eme(x, ws=10):  # x: HW
    k1, k2 = x.shape[1] // ws, x.shape[0] // ws
    b = x[:k2 * ws, :k1 * ws].reshape(k2, ws, k1, ws).permute(0, 2, 1, 3).reshape(-1, ws * ws)
    mx, mn = b.max(1).values, b.min(1).values
    m = (mn != 0) & (mx != 0)
    lr = torch.zeros_like(mx)
    lr[m] = torch.log(mx[m] / mn[m])
    return 2.0 / (k1 * k2) * lr.sum()


def _uism(x):  # x: CHW [0,255]; NOTE lambda_b=0.144 quirk preserved
    e = [_eme(_sobel_mag(x[i]) * x[i]) for i in range(3)]
    return 0.299 * e[0] + 0.587 * e[1] + 0.144 * e[2]


def _uiconm(x, ws=10):  # x: CHW
    k1, k2 = x.shape[2] // ws, x.shape[1] // ws
    b = x[:, :k2 * ws, :k1 * ws].unfold(1, ws, ws).unfold(2, ws, ws).reshape(-1, k2, k1, ws, ws)
    mn, mx = b.amin(dim=(0, -1, -2)), b.amax(dim=(0, -1, -2))
    top, bot = mx - mn, mx + mn
    val = (top / bot) * torch.log(top / bot)
    val = torch.where(torch.isnan(val) | (bot == 0) | (top == 0), torch.zeros_like(val), val)
    return -1.0 / (k1 * k2) * val.sum()


def uiqm_batch(imgs):
    """imgs: BCHW [0,1] torch -> (B,) numpy UIQM."""
    return np.array([(0.0282 * _uicm(x) + 0.2953 * _uism(x) + 3.5753 * _uiconm(x)).item()
                     for x in (imgs * 255.0)])


def uism_batch(imgs):
    """Sharpness component only (cheaper)."""
    return np.array([_uism(x).item() for x in (imgs * 255.0)])


def uciqe_batch(imgs):
    """imgs: BCHW [0,1] torch -> (B,) numpy UCIQE (kornia HSV)."""
    import kornia.color as K
    out = []
    for i in range(imgs.shape[0]):
        H, S, V = (t.squeeze() for t in torch.chunk(K.rgb_to_hsv(imgs[i][None]), 3, dim=1))
        n = V.numel() // 100
        bot = V.flatten().sort()[0][:n].mean()
        top = V.flatten().sort(descending=True)[0][:n].mean()
        out.append((0.4680 * H.std() / (2 * np.pi) + 0.2745 * (top - bot) + 0.2576 * S.mean()).item())
    return np.array(out)
