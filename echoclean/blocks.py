"""Reading the game's texture files and working with their ETC2 RGBA8 blocks.

A game texture (*.png in the game folders, despite the name) is  UF wrapper -> CCZ (zlib) -> PVR v3, pixel format 23
(ETC2 RGBA8: per 4x4 block 8 bytes EAC alpha + 8 bytes ETC2 colour), premultiplied alpha.
Block arrays here have shape (H/4, W/4, 16), uint8.
"""
import io, zlib, struct
import numpy as np
from PIL import Image
import texture2ddecoder

UF_KEY = bytes.fromhex('135b0c0d6616222b1119584024100e423157382c351c0b0574253a69140f4d071d')


def uf_decode(buf):
    """Remove the game's 'UF' wrapper (a repeating-key XOR). Data without the wrapper is returned unchanged."""
    if len(buf) < 4 or buf[0] != 0x55 or buf[1] != 0x46: return buf
    p = 2; extra = 0
    if buf[2] == 0x4f: p = 4; extra = 1
    mode = buf[p + 1]; L = len(UF_KEY)
    if mode == 1:
        off = buf[p + 2]; p += 3
        return bytes(UF_KEY[(i + off) % L] ^ c for i, c in enumerate(buf[p:]))
    if mode == 2:
        off = buf[p + 2]; p += 3; n = len(buf) - p; out = bytearray(buf[:n])
        for j in range(p): out[j] = UF_KEY[(j + off) % L] ^ buf[n + j]
        for i in range(p, p + min(n - p, 0x5f)): out[i] ^= UF_KEY[(i + off) % L]
        return bytes(out)
    return buf[extra * 2 + 5:]


class NotEtc2(ValueError):
    pass


def load_raw(path):
    """-> (blocks, width, height). Raises NotEtc2 for files that are not ETC2 RGBA8 (plain PNGs, JPEGs, other formats)."""
    d = uf_decode(open(path, 'rb').read())
    if d[:4] == b'CCZ!': d = zlib.decompress(d[16:])
    if d[:4] != b'PVR\x03': raise NotEtc2(f'{path}: not a PVR texture')
    _, flags, pf, cs, ct, h, w, dep, ns, nf, nm, meta = struct.unpack('<IIQIIIIIIIII', d[:52])
    if pf != 23: raise NotEtc2(f'{path}: PVR pixel format {pf}, not ETC2 RGBA8')
    bw, bh = (w + 3) // 4, (h + 3) // 4
    return np.frombuffer(d[52 + meta:52 + meta + bw * bh * 16], np.uint8).reshape(bh, bw, 16).copy(), w, h


def decode(blocks, w=None, h=None):
    """blocks -> premultiplied RGBA uint8 image (cropped to w x h if given)."""
    bh, bw = blocks.shape[:2]
    px = np.frombuffer(texture2ddecoder.decode_etc2a8(np.ascontiguousarray(blocks).tobytes(), bw * 4, bh * 4), np.uint8).reshape(bh * 4, bw * 4, 4)[..., [2, 1, 0, 3]]
    return px[:h or bh * 4, :w or bw * 4].copy()


def load_clean(path):
    """Load a clean RGBA image as premultiplied uint8 (straight-alpha files are premultiplied; already premultiplied ones are kept)."""
    a = np.asarray(Image.open(path).convert('RGBA'))
    part = (a[..., 3] > 0) & (a[..., 3] < 255)
    over = (a[..., :3].max(2) > a[..., 3])[part].mean() if part.any() else 1.0
    if over < 0.02: return a.copy()
    f = a.astype(np.float32); f[..., :3] *= f[..., 3:] / 255
    return f.round().astype(np.uint8)


MODES = ('individual', 'differential', 'T', 'H', 'planar')
def colour_mode(blocks):
    """ETC2 colour mode per block: 0 individual, 1 differential, 2 T, 3 H, 4 planar."""
    c = blocks[..., 8:12].astype(np.int32); diff = (blocks[..., 11] >> 1) & 1
    def over(b):
        d = b & 7; s = (b >> 3) + np.where(d > 3, d - 8, d); return (s < 0) | (s > 31)
    return np.where(diff == 0, 0, np.where(over(c[..., 0]), 2, np.where(over(c[..., 1]), 3, np.where(over(c[..., 2]), 4, 1))))


EAC = np.array([[-3, -6, -9, -15, 2, 5, 8, 14], [-3, -7, -10, -13, 2, 6, 9, 12], [-2, -5, -8, -13, 1, 4, 7, 12], [-2, -4, -6, -13, 1, 3, 5, 12],
                [-3, -6, -8, -12, 2, 5, 7, 11], [-3, -7, -9, -11, 2, 6, 8, 10], [-4, -7, -8, -11, 3, 6, 7, 10], [-3, -5, -8, -11, 2, 4, 7, 10],
                [-2, -6, -8, -10, 1, 5, 7, 9], [-2, -5, -8, -10, 1, 4, 7, 9], [-2, -4, -8, -10, 1, 3, 7, 9], [-2, -5, -7, -10, 1, 4, 6, 9],
                [-3, -4, -7, -10, 2, 3, 6, 9], [-1, -2, -3, -10, 0, 1, 2, 9], [-4, -6, -8, -9, 3, 5, 7, 8], [-3, -5, -7, -9, 2, 4, 6, 8]])


def alpha_bounds(blocks, alpha):
    """Per texel: the interval of true alpha that the encoder would have coded as this texel's EAC level -> (lo, hi) float images."""
    bh, bw = blocks.shape[:2]; base = blocks[..., 0].astype(np.int32); mult = (blocks[..., 1] >> 4).astype(np.int32); tab = blocks[..., 1] & 15
    lev = np.sort(np.clip(base[..., None] + EAC[tab] * mult[..., None], 0, 255), -1)[:, :, None, :]
    a = alpha.reshape(bh, 4, bw, 4).transpose(0, 2, 1, 3).reshape(bh, bw, 16, 1).astype(np.int32)
    below = np.where(lev < a, lev, -10 ** 6).max(-1); above = np.where(lev > a, lev, 10 ** 6).min(-1); a = a[..., 0]
    lo = np.where(below > -10 ** 5, (a + below) / 2.0, 0.0); hi = np.where(above < 10 ** 5, (a + above) / 2.0, 255.0)
    un = lambda v: v.reshape(bh, bw, 4, 4).transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4)
    return un(lo), un(hi)


NSIDE = 9
def coding_features(blocks, alpha):
    """Side information for the stage-1 model, (H, W, 7) float32: alpha interval lo/hi (0..1) and the colour mode one-hot.
    The two position-in-block channels are added on the fly (they depend on orientation)."""
    lo, hi = alpha_bounds(blocks, alpha); m = colour_mode(blocks); bh, bw = m.shape
    oh = np.zeros((bh, bw, 5), np.float32); np.put_along_axis(oh, m[..., None], 1.0, -1)
    return np.concatenate([(lo / 255)[..., None], (hi / 255)[..., None], np.repeat(np.repeat(oh, 4, 0), 4, 1)], -1).astype(np.float32)


def with_pos(feat):
    h, w = feat.shape[:2]; yy, xx = np.mgrid[0:h, 0:w]
    return np.concatenate([((xx % 4) / 3.0)[..., None], ((yy % 4) / 3.0)[..., None], feat], -1).astype(np.float32)
