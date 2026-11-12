"""Scores against a clean original. All take premultiplied RGBA uint8 arrays.

psnr      over texels with alpha > 0
edges     PSNR over soft-edge texels (0 < alpha < 255)
seam      the render error when the texture is drawn over its own extended solid colour (what a same-coloured layer
          underneath would be), blurred 1.5 px so only line-like error counts; RMS in 8-bit levels over the soft band.
          This is what shows up as a seam where two skin pieces overlap. Clean original = 0.
ssim      SSIM of that same composite over the soft band (1 = identical)
"""
import numpy as np, torch, torch.nn.functional as F


def gauss(t, s):
    k = int(2 * np.ceil(3 * s) + 1); x = torch.arange(k, device=t.device) - k // 2
    g = torch.exp(-x.float() ** 2 / (2 * s * s)); g = g / g.sum(); C = t.shape[1]
    t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode='replicate'), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode='replicate'), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


def own_colour(t, sigmas=(1.5, 3, 6, 12, 24)):
    """t: (B,4,H,W) premultiplied 0..1 -> (straight colour of the solid part extended outward, validity mask)."""
    a = t[:, 3:4]; solid = (a >= 200 / 255).float(); num = t[:, :3] * solid; S = torch.zeros_like(num); done = torch.zeros_like(solid)
    for s in sigmas:
        n, d = gauss(num, s), gauss(solid * a, s); ok = (d > 0.02).float() * (1 - done); S = S + ok * n / d.clamp(min=1e-4); done = done + ok
    return S.clamp(0, 1), done


def ssim_map(a, b, sigma=1.5):
    C1, C2 = 0.01 ** 2, 0.03 ** 2; ma, mb = gauss(a, sigma), gauss(b, sigma)
    va = gauss(a * a, sigma) - ma * ma; vb = gauss(b * b, sigma) - mb * mb; cov = gauss(a * b, sigma) - ma * mb
    return ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))


def score(out, clean, dev=None):
    dev = dev or ('cuda' if torch.cuda.is_available() else 'cpu')
    tt = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev).float().permute(2, 0, 1)[None] / 255
    c = clean[..., 3] > 0; e = c & (clean[..., 3] < 255)
    ps = lambda m: float(10 * np.log10(255 ** 2 / max(((out.astype(np.float64) - clean)[m] ** 2).mean(), 1e-12))) if m.any() else float('nan')
    Y, O = tt(clean), tt(out); S, valid = own_colour(Y); a = Y[:, 3:4]
    band = (F.max_pool2d(((a > 0) & (a < 1)).float() * valid, 3, 1, 1) > 0).float() * valid; n = band.sum().clamp(min=1)
    D = ((O[:, :3] - Y[:, :3]) - S * (O[:, 3:4] - a)) * 255
    lum = (D * torch.tensor([0.3, 0.59, 0.11], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
    coh = gauss(lum * band, 1.5) / gauss(band, 1.5).clamp(min=1e-3)
    sm = ssim_map(O[:, :3] + (1 - O[:, 3:4]) * S, Y[:, :3] + (1 - a) * S).mean(1, keepdim=True)
    return dict(psnr=ps(c), edges=ps(e), seam=float(((coh ** 2) * band).sum().div(n).sqrt()), ssim=float((sm * band).sum() / n))


def fmt(r): return f"PSNR {r['psnr']:.2f} dB / edges {r['edges']:.2f} dB / seam {r['seam']:.3f} / ssim {r['ssim']:.4f}"
