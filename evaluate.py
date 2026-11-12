#!/usr/bin/env python3
"""Score the cleaner on a real pair: a raw game texture and the clean original of the same image.

  python evaluate.py --raw path/to/raw_game_texture.png --clean path/to/clean_original.png

Only meaningful when the clean file is exactly what the game encoded. Check that first: the line 'identical blocks'
says how many of the game's blocks come out byte for byte when the clean file is re-encoded (about 99% for a true pair).
"""
import argparse, numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', required=True); ap.add_argument('--clean', required=True)
    ap.add_argument('--stage1', default=None); ap.add_argument('--refiner', default=None); ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    from echoclean import blocks as B, pipeline as PL, metrics as M, encoder
    blocks, w, h = B.load_raw(a.raw); y = B.load_clean(a.clean)
    if y.shape[:2] != (h, w): raise SystemExit(f'size differs: raw {w}x{h}, clean {y.shape[1]}x{y.shape[0]}')
    P = PL.Pipeline(a.stage1 or PL.STAGE1, a.refiner or PL.REFINER, 'cpu' if a.cpu else None)
    try:
        if h % 4 == 0 and w % 4 == 0:
            live = (blocks != 0).any(-1); same = (encoder.encode_blocks_mp(y, P.pool) == blocks).all(-1)
            print(f'identical blocks when the clean file is re-encoded: {100 * (same & live).sum() / max(live.sum(), 1):.1f}%')
        print(f"{'raw game texture':34s} {M.fmt(M.score(B.decode(blocks, w, h), y, P.dev))}")
        for lab, kw in (('stage 1, single pass', dict(tta=False, refine=False)), ('stage 1, 8x self-ensemble', dict(refine=False)), ('full pipeline', {})):
            print(f'{lab:34s} {M.fmt(M.score(P(blocks, w, h, **kw), y, P.dev))}', flush=True)
    finally: P.close()


if __name__ == '__main__':
    main()
