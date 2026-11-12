"""The cleaner: stage 1 with 8x self-ensemble, then the consistency refiner."""
import os, multiprocessing as mp
import numpy as np, torch
from . import blocks as B, encoder, models

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE1 = os.path.join(HERE, 'models', 'echoclean_stage1.pt'); REFINER = os.path.join(HERE, 'models', 'echoclean_refiner.pt')
PAD = 32


def _tiles(net, inp, h, w, tile, half):
    out = torch.zeros((4, h, w), device=inp.device)
    for yy in range(0, h, tile):
        for xx in range(0, w, tile):
            with torch.autocast(inp.device.type, dtype=torch.float16, enabled=half and inp.device.type == 'cuda'):
                r = net(inp[:, :, yy:yy + tile + 2 * PAD, xx:xx + tile + 2 * PAD])[0].float()
            out[:, yy:yy + tile, xx:xx + tile] = r[:, PAD:PAD + min(tile, h - yy), PAD:PAD + min(tile, w - xx)]
    return out


def finish(o):
    o = np.clip(o + 0.5, 0, 255).astype(np.uint8); o[..., :3] = np.minimum(o[..., :3], o[..., 3:4]); return o


@torch.no_grad()
def stage1(net, blocks, dev, tta=True, tile=1024, half=True):
    """blocks -> float32 RGBA 0..255 at the padded (multiple of 4) size."""
    x = B.decode(blocks); feat = B.coding_features(blocks, x[..., 3])
    base = torch.cat([torch.from_numpy(x).to(dev).float().div_(255), torch.from_numpy(feat).to(dev)], -1).permute(2, 0, 1)[None]
    acc = torch.zeros((4,) + tuple(base.shape[2:]), device=dev); n = 0
    for k in (range(4) if tta else (0,)):
        for f in ((False, True) if tta else (False,)):
            t = torch.rot90(base, k, (2, 3)); t = torch.flip(t, (3,)) if f else t; h, w = t.shape[2:]
            px = (torch.arange(w, device=dev) % 4).float().div_(3).view(1, 1, 1, w).expand(1, 1, h, w); py = (torch.arange(h, device=dev) % 4).float().div_(3).view(1, 1, h, 1).expand(1, 1, h, w)
            out = _tiles(net, torch.nn.functional.pad(torch.cat([t[:, :4], px, py, t[:, 4:]], 1), (PAD,) * 4), h, w, tile, half).clamp_(0, 1).mul_(255)
            out = torch.flip(out, (2,)) if f else out; acc += torch.rot90(out, -k, (1, 2)); n += 1
    return (acc / n).permute(1, 2, 0).cpu().numpy()


def refiner_input(x, o1, blocks, reencoded):
    """x, o1: uint8 images; blocks: the game's blocks; reencoded: blocks of encode(o1). -> (H, W, 13) float32"""
    fb = (B.decode(reencoded).astype(np.float32) - x) / 255
    mm = np.repeat(np.repeat((reencoded != blocks).any(-1).astype(np.float32), 4, 0), 4, 1)[..., None]
    return np.concatenate([x / 255.0, o1 / 255.0, fb, mm], -1).astype(np.float32)


class Pipeline:
    def __init__(s, stage1_path=STAGE1, refiner_path=REFINER, device=None, procs=None):
        s.dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        for p in (stage1_path, refiner_path):
            if p and not os.path.exists(p): raise FileNotFoundError(f'{p} not found. Download the model files from the GitHub release into models/ (see README).')
        s.net1 = models.load_stage1(stage1_path, s.dev); s.ref = models.load_refiner(refiner_path, s.dev) if refiner_path else None
        s.pool = mp.get_context('spawn').Pool(procs or max(1, (os.cpu_count() or 4) - 2), maxtasksperchild=64) if s.ref is not None else None

    @torch.no_grad()
    def __call__(s, blocks, w=None, h=None, tta=True, refine=True, half=True):
        """blocks from blocks.load_raw -> cleaned premultiplied RGBA uint8."""
        o = finish(stage1(s.net1, blocks, s.dev, tta, half=half))
        if refine and s.ref is not None:
            x = B.decode(blocks); inp = refiner_input(x, o, blocks, encoder.encode_blocks_mp(o, s.pool)); H, W = x.shape[:2]
            t = torch.nn.functional.pad(torch.from_numpy(inp).to(s.dev).permute(2, 0, 1)[None], (PAD,) * 4)
            o = finish(_tiles(s.ref, t, H, W, 1024, False).clamp_(0, 1).mul_(255).permute(1, 2, 0).cpu().numpy())
        return o[:h or o.shape[0], :w or o.shape[1]]

    def close(s):
        if s.pool: s.pool.terminate()
