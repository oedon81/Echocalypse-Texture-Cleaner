#!/usr/bin/env python3
"""Clean Echocalypse textures.

  python clean.py INPUT [INPUT ...] -o OUTPUT_DIR

INPUT is a raw game texture or a folder of them (searched recursively). Raw means the files exactly as the game ships
them (e.g. .../res/common/knight_spine/<id>/<id>.png): they keep the .png name but are ETC2 data, not real PNGs.
OUTPUT gets real, premultiplied-alpha PNGs with the same names and folder layout. Files that are not ETC2 textures
(real PNGs, JPEGs, atlases, skeletons) are skipped, or copied with --copy-others. Finished files are skipped on a re-run.
"""
import os, sys, time, shutil, argparse
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+'); ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--stage1', default=None, help='stage-1 model (default: models/echoclean_stage1.pt)')
    ap.add_argument('--refiner', default=None, help='refiner model (default: models/echoclean_refiner.pt)')
    ap.add_argument('--no-refiner', action='store_true', help='skip stage 2 (no encoder library needed; slightly lower quality)')
    ap.add_argument('--no-ensemble', action='store_true', help='one pass instead of 8 flips/rotations: about 3x faster, about 0.35 dB lower')
    ap.add_argument('--fp32', action='store_true', help='full precision on the GPU (default: half precision, same scores)')
    ap.add_argument('--cpu', action='store_true'); ap.add_argument('--copy-others', action='store_true', help='also copy non-texture files (atlas, skel, ...) to the output')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    from echoclean import blocks as B, pipeline as PL
    jobs = []
    for inp in a.inputs:
        if os.path.isdir(inp):
            root = os.path.abspath(inp)
            for d, _, fs in os.walk(root):
                for f in sorted(fs): jobs.append((os.path.join(d, f), os.path.relpath(os.path.join(d, f), root)))
        else: jobs.append((inp, os.path.basename(inp)))
    P = PL.Pipeline(a.stage1 or PL.STAGE1, None if a.no_refiner else (a.refiner or PL.REFINER), 'cpu' if a.cpu else None)
    print(f'{len(jobs)} files; running on {P.dev}' + ('' if P.dev.type == 'cuda' else ' (no GPU found: expect minutes per texture)'), flush=True)
    t0 = time.time(); n = skipped = 0
    try:
        for src, rel in jobs:
            dst = os.path.join(a.out, rel)
            if os.path.exists(dst) and not a.overwrite: continue
            try: blocks, w, h = B.load_raw(src)
            except B.NotEtc2:
                skipped += 1
                if a.copy_others: os.makedirs(os.path.dirname(dst) or '.', exist_ok=True); shutil.copy2(src, dst)
                continue
            out = P(blocks, w, h, tta=not a.no_ensemble, half=not a.fp32)
            os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
            Image.fromarray(out, 'RGBA').save(dst + '.tmp.png', compress_level=3); os.replace(dst + '.tmp.png', dst); n += 1
            print(f'[{n}] {rel}  {w}x{h}  ({(time.time() - t0) / n:.1f}s per texture)', flush=True)
    finally: P.close()
    print(f'cleaned {n} textures, {skipped} other files {"copied" if a.copy_others else "skipped"}')


if __name__ == '__main__':
    main()
