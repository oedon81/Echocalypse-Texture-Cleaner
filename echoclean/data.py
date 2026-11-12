"""Training data: clean art in, (game-damaged patch, clean patch) out. The damage is made with the game's own encoder."""
import os, glob, time
import numpy as np, torch
from torch.utils.data import IterableDataset, get_worker_info
from . import blocks as B, encoder


def list_images(spec):
    """'path' or 'path:weight'. path = a folder (searched recursively for .png) or a text file with one image path per line."""
    path, _, w = spec.rpartition(':') if (':' in spec and spec.rsplit(':', 1)[1].replace('.', '', 1).isdigit()) else (spec, '', '1')
    if os.path.isdir(path): files = sorted(glob.glob(os.path.join(path, '**', '*.png'), recursive=True))
    else: files = [l.strip() for l in open(path, encoding='utf-8') if l.strip() and not l.startswith('#')]
    if not files: raise SystemExit(f'no images found in {path}')
    return files, float(w)


class Source:
    """A rotating pool of clean images kept in memory; yields patches aligned to nothing in particular (each patch is encoded on its own)."""
    def __init__(s, files, rng, pool=6, swap=24):
        s.files, s.rng, s.swap, s.n = files, rng, swap, 0
        s.pool = [im for im in (s._load(s._pick()) for _ in range(min(pool, len(files)))) if im is not None]
    def _pick(s): return s.files[s.rng.integers(len(s.files))]
    def _load(s, p):
        try:
            im = B.load_clean(p); h, w = im.shape[:2]; return im[:h // 4 * 4, :w // 4 * 4]
        except Exception: return None
    def patch(s, P):
        s.n += 1
        if len(s.files) > len(s.pool) and s.n % s.swap == 0:
            im = s._load(s._pick())
            if im is not None: s.pool[s.rng.integers(len(s.pool))] = im
        for _ in range(40):
            if not s.pool: return None
            im = s.pool[s.rng.integers(len(s.pool))]; H, W = im.shape[:2]
            if H < P + 8 or W < P + 8: continue
            y0 = s.rng.integers(0, H - P - 3); x0 = s.rng.integers(0, W - P - 3); y = im[y0:y0 + P, x0:x0 + P]; al = y[..., 3]
            if al.mean() < 20: continue
            if ((al > 0) & (al < 255)).mean() < 0.01 and s.rng.random() < 0.6: continue          # prefer patches with soft edges
            return y
        return None


class PatchStream(IterableDataset):
    """Yields (x, y, blocks): x = (B, P, P, 13) float32 stage-1 input (texture 0..1 + side channels), y = clean uint8, game blocks.
    Finite on purpose: DataLoader workers are recreated each pass, which keeps their memory bounded."""
    def __init__(s, groups, P, B_, pool=6, epoch=250):
        s.groups, s.P, s.B, s.pool, s.epoch = groups, P, B_, pool, epoch          # groups: [(files, weight), ...]
    def __iter__(s):
        wi = get_worker_info(); torch.set_num_threads(1)
        rng = np.random.default_rng((wi.id if wi else 0) * 1000 + int(time.time() * 1000) % 1000000)
        srcs = [Source(f, rng, pool=s.pool) for f, _ in s.groups]; w = np.array([g[1] for g in s.groups], np.float64); w /= w.sum()
        for _ in range(s.epoch):
            xs, ys, bs = [], [], []
            while len(xs) < s.B:
                y = srcs[rng.choice(len(srcs), p=w)].patch(s.P)
                if y is None: continue
                k = rng.integers(4); y = np.rot90(y, k); y = np.ascontiguousarray(y[:, ::-1] if rng.random() < 0.5 else y)
                bl = encoder.encode_blocks(y); x = B.decode(bl)
                xs.append(np.concatenate([x.astype(np.float32) / 255, B.with_pos(B.coding_features(bl, x[..., 3]))], -1)); ys.append(y); bs.append(bl)
            yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(bs))


def forever(loader):
    while True:
        for b in loader: yield b
