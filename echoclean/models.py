"""The two networks.

Stage 1 (SideNet): residual CNN on the decoded texture plus 9 side channels from the compressed blocks (position inside the
4x4 block, the alpha interval each texel's EAC level stands for, the block's colour mode).
Stage 2 (Refiner): sees the texture, the stage-1 output and what the game's encoder does with that output, and corrects it.
"""
import torch, torch.nn as nn
from .blocks import NSIDE


class RB(nn.Module):
    def __init__(s, c): super().__init__(); s.a = nn.Conv2d(c, c, 3, padding=1); s.b = nn.Conv2d(c, c, 3, padding=1); s.r = nn.ReLU()
    def forward(s, x): return x + s.b(s.r(s.a(x)))


def _valid(y):
    a = y[:, 3:4].clamp(0, 1); return torch.cat([torch.minimum(y[:, :3].clamp(0, 1), a), a], 1)     # premultiplied: colour <= alpha


class SideNet(nn.Module):
    """input (B, 4 + 9, H, W): game texture (premultiplied, 0..1) + side channels; output (B, 4, H, W)."""
    def __init__(s, c=256, n=8):
        super().__init__()
        s.down = nn.PixelUnshuffle(2); s.head = nn.Conv2d((4 + NSIDE) * 4, c, 3, padding=1)
        s.body = nn.Sequential(*[RB(c) for _ in range(n)]); s.tail = nn.Conv2d(c, 16, 3, padding=1); s.up = nn.PixelShuffle(2)
    def forward(s, inp): return _valid(inp[:, :4] + s.up(s.tail(s.body(s.head(s.down(inp))))))


class Refiner(nn.Module):
    """input (B, 13, H, W): texture (4), stage-1 output (4), decode(encode(stage-1 output)) - texture (4), mismatch mask (1)."""
    def __init__(s, c=128, n=8):
        super().__init__()
        s.down = nn.PixelUnshuffle(2); s.head = nn.Conv2d(13 * 4, c, 3, padding=1)
        s.body = nn.Sequential(*[RB(c) for _ in range(n)]); s.tail = nn.Conv2d(c, 16, 3, padding=1); s.up = nn.PixelShuffle(2)
        nn.init.zeros_(s.tail.weight); nn.init.zeros_(s.tail.bias)                                    # starts as "no correction"
    def forward(s, inp): return _valid(inp[:, 4:8] + s.up(s.tail(s.body(s.head(s.down(inp))))))


def _shape(sd): return sd['head.weight'].shape[0], len({k.split('.')[1] for k in sd if k.startswith('body.')})


def load_stage1(path, dev='cpu'):
    sd = torch.load(path, map_location='cpu', weights_only=True)['net']; c, n = _shape(sd)
    if sd['head.weight'].shape[1] == 16:                       # a plain 4-channel checkpoint: new input channels start at zero
        w = torch.zeros((c, (4 + NSIDE) * 4, 3, 3)); w[:, :16] = sd['head.weight']; sd = dict(sd); sd['head.weight'] = w
    net = SideNet(c, n); net.load_state_dict(sd); return net.to(dev).eval()


def load_refiner(path, dev='cpu'):
    sd = torch.load(path, map_location='cpu', weights_only=True)['net']; net = Refiner(*_shape(sd)); net.load_state_dict(sd); return net.to(dev).eval()


def save(net, path, **meta):
    torch.save({'net': {k: v.detach().cpu() for k, v in net.state_dict().items()}, **meta}, path)
