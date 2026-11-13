"""The game's ETC2 encoder, in process: Ericsson ETCPACK v2.74, 'fast' speed, 'perceptual' metric.

Encoding a clean original with it reproduces the game's blocks byte for byte (100% on the pair this was measured on), so
training data made with it carries the real damage. That needs the same rand() the game's build machine had (macOS); the
library brings its own, see native/etcpack_lib.cxx. Build it first with  ./build_etcpack.sh  (see README).
"""
import os, ctypes
import numpy as np
from . import blocks as B

_LIB = None
def _lib():
    global _LIB
    if _LIB is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.environ.get('ETCPACK_LIB') or os.path.join(here, 'native', 'libetcpack.so')
        if not os.path.exists(path): raise FileNotFoundError(f'{path} not found. Run ./build_etcpack.sh first (or set ETCPACK_LIB).')
        _LIB = ctypes.CDLL(path)
        _LIB.etcpack_rgba8.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]; _LIB.etcpack_rgba8.restype = ctypes.c_int
    return _LIB


def encode_blocks(pm, fast=True, perceptual=True):
    """premultiplied RGBA uint8 (H and W multiples of 4) -> blocks (H/4, W/4, 16). Single-threaded. May be called from several threads at once
    (ctypes releases the GIL), provided one call has finished before the others start: the first call sets up the tables."""
    h, w = pm.shape[:2]
    if h % 4 or w % 4: raise ValueError('size must be a multiple of 4')
    out = ctypes.create_string_buffer(h * w)
    if _lib().etcpack_rgba8(np.ascontiguousarray(pm, np.uint8).tobytes(), w, h, out, int(fast), int(perceptual)): raise RuntimeError('etcpack failed')
    return np.frombuffer(out.raw, np.uint8).reshape(h // 4, w // 4, 16)


def encode(pm):
    """clean premultiplied image -> the same image as the game would ship it (decoded ETC2)."""
    return B.decode(encode_blocks(pm))


def _stripe(args): return encode_blocks(args)
def encode_blocks_mp(pm, pool):
    """Whole pages faster: rows of blocks are independent, so stripes go to a multiprocessing pool."""
    h = pm.shape[0]; n = max(1, min(pool._processes, h // 64)); cuts = [round(i * (h // 4) / n) * 4 for i in range(n + 1)]
    return np.concatenate(pool.map(_stripe, [pm[cuts[i]:cuts[i + 1]] for i in range(n) if cuts[i + 1] > cuts[i]]), 0)
