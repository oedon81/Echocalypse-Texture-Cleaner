# Echocalypse Texture Cleaner

Removes ETC2 compression damage from the Spine character textures of *Echocalypse: The Scarlet Covenant*: the dotted
block noise, the grey fringes on soft edges, and most of the seam lines where skin pieces overlap.

The game ships every character texture as ETC2. This project identified the exact encoder the game's build uses, so the
models are trained on the real damage and not on an approximation of it.

| On a real game texture with a known clean original (held out from training) | PSNR | Soft edges | Seam error | SSIM |
|---|---|---|---|---|
| Raw game texture | 36.38 dB | 36.05 dB | 0.880 | 0.9112 |
| Stage 1, single pass | 42.50 dB | 42.92 dB | 0.551 | 0.9893 |
| Stage 1, 8x self-ensemble | 42.84 dB | 43.27 dB | 0.533 | 0.9906 |
| Full pipeline (default) | 42.97 dB | 43.45 dB | 0.512 | 0.9907 |

Seam error is the render error when a piece is drawn over its own skin colour, counting only line-like error, in 8-bit
levels; a clean original scores 0. Seams get much fainter. They do not all disappear: very small pieces (a mouth patch a
few blocks wide) lose their edge colour at compression time and no model can know what it was.

## What is in here

- `clean.py`: clean raw game textures into ordinary PNGs.
- `evaluate.py`: score the cleaner on a raw texture plus its clean original.
- `train_stage1.py`, `train_refiner.py`: train or fine-tune the two models on your own clean art.
- `tools/fetch_textures.py`: download raw hero textures (and the better "censor pack" versions) from the game's update CDN.
- `echoclean/`: the library the scripts use.

Not in here: any game art, and the encoder itself (Ericsson ETCPACK; its licence does not allow redistribution, so a
script downloads and builds it on your machine).

## Setup

Linux with Python 3.10 or newer. On Windows use WSL. An NVIDIA or AMD GPU is strongly recommended: a 2048x2048 texture
takes about 6 seconds on a mid-range GPU and minutes on a CPU.

```bash
git clone https://github.com/oedon81/Echocalypse-Texture-Cleaner
cd Echocalypse-Texture-Cleaner
```

1. Install PyTorch for your GPU from <https://pytorch.org> (CUDA for NVIDIA, ROCm for AMD).
2. Install the rest:

```bash
pip install -r requirements.txt
```

3. Download `echoclean_stage1.pt` and `echoclean_refiner.pt` from the
   [Releases page](https://github.com/oedon81/Echocalypse-Texture-Cleaner/releases) into the `models/` folder.
4. Build the encoder library (needed for the refiner and for training). It needs `git` and a C++ compiler:

```bash
./build_etcpack.sh
```

If you have no compiler, `pip install ziglang` provides one and the script will use it. If you skip this step entirely
you can still clean with `--no-refiner`.

## Cleaning textures

The input must be the **raw** game files, exactly as the game ships them, for example
`.../res/common/knight_spine/<id>/<id>.png`. Despite the name these are not PNGs; they are ETC2 data in the game's own
wrapper. The cleaner needs the compressed blocks, so a texture that was already decoded to a real PNG will not work.

```bash
python clean.py path/to/knight_spine/600090 -o cleaned/600090 --copy-others
```

- A folder is searched recursively and the output keeps the same layout and file names.
- The output files are real PNGs with premultiplied alpha, the same as the game's textures.
- `--copy-others` also copies the `.atlas` and `.skel` files, so the output folder is a complete Spine 3.8 model.
- Re-running skips files that are already done.

| Option | Effect |
|---|---|
| `--no-ensemble` | One pass instead of eight: about 3x faster, about 0.35 dB lower |
| `--no-refiner` | Skip stage 2: no encoder library needed, about 0.1 dB lower |
| `--fp32` | Full precision on the GPU (the default half precision gives the same scores) |
| `--cpu` | Run without a GPU |

### Getting the raw files

Either copy them from your own game install, or download them:

```bash
python tools/fetch_textures.py list                       # every hero id
python tools/fetch_textures.py get 600090 700120 -o raw   # raw/<id>/...
```

### Use the censor pack when there is one

The China build has a separate "censor pack" for 316 heroes. For about 130 older heroes that pack contains the same
art encoded from the original picture, while the normal file was encoded from a dithered export. The pack is then a
much better input: about 3 to 4 dB closer to the original before any cleaning, and without the dotted texture on faces.

```bash
python tools/fetch_textures.py censor 700120 -o raw_censor
python clean.py raw_censor/700120 -o cleaned/700120 --copy-others
```

Check a pack before you rely on it:

- Some packs are byte-identical to the normal file (nothing gained), and newer heroes have no pack.
- A few are really censored (the art is changed), and some use a different atlas layout or a different export, so their
  texture only works with the pack's own `.atlas` and `.skel`.

## Checking a result

If you have a clean original of a texture (a few old game builds shipped some uncompressed), you can measure:

```bash
python evaluate.py --raw raw/600090/600090.png --clean originals/600090.png
```

The first line reports how many of the game's blocks come out byte for byte when the clean file is re-encoded. About
99% means the clean file really is what the game encoded. A low number means it is a different export of the same art,
and the scores below it are then not meaningful.

## Training

Training needs clean, uncompressed RGBA art. More and closer to the game's style is better. The scripts damage it on
the fly with the game's encoder, so the clean images are all you provide.

Fine-tune the released stage-1 model:

```bash
python train_stage1.py --name myrun \
    --data path/to/echocalypse_clean:0.4 --data path/to/other_clean_art:0.6 \
    --init models/echoclean_stage1.pt \
    --val-raw raw/600090/600090.png --val-clean originals/600090.png
```

- `--data` is a folder of PNGs (searched recursively) or a text file of image paths. `:0.4` sets how often that set is
  sampled.
- `--val-raw` / `--val-clean` is an optional real pair scored every 2,000 steps. Keep that image out of `--data`.
- Leave out `--init` to train from scratch (use about 150,000 steps).
- Results go to `runs/<name>/`: `ckpt_last.pt`, `ckpt_best.pt` and `log.txt`.

Then train the refiner on top of it:

```bash
python train_refiner.py --name myrefiner --stage1 runs/myrun/ckpt_best.pt \
    --data path/to/echocalypse_clean:0.4 --data path/to/other_clean_art:0.6 \
    --init models/echoclean_refiner.pt
```

Use your models with `python clean.py ... --stage1 runs/myrun/ckpt_best.pt --refiner runs/myrefiner/ckpt_best.pt`.

Things learned the hard way:

- **The last gains come at low learning rates.** A run that looks flat usually steps up again when the rate drops. The
  released stage 1 was finished at 5e-6.
- **Memory.** Each data worker keeps a few clean images in memory; lower `--pool` or `--workers` if RAM is short.
- **One training at a time** on a 16 GB machine.

## How it works

1. **The encoder.** The game's textures are made with Ericsson ETCPACK 2.74 in `fast` + `perceptual` mode. Encoding a
   clean original with it reproduces about 99% of the game's blocks byte for byte. A general-purpose ETC2 encoder gets
   about 2% at soft edges, and its errors there go in the opposite direction.
2. **Stage 1** is a residual CNN (9.6M parameters) on the decoded texture plus side information read from the compressed
   blocks: each texel's position inside its 4x4 block, the alpha range its code stands for, and the block's colour mode.
   Besides plain pixel error it is trained on the texture composited over its own colour, which is where seams show.
3. **Self-ensemble.** The texture is cleaned in eight flips and rotations and the results are averaged.
4. **Stage 2, the refiner** (2.4M parameters). A clean original always re-compresses to exactly the game's blocks. The
   refiner is shown how the stage-1 output re-compresses and which blocks do not match, and corrects it.

What did not help, in case you want to try: a much bigger model, a four times heavier seam loss, larger training
patches, a second refiner pass, and blending in a second, worse copy of the same texture.

## Credits

- ETC2 encoder: [Ericsson ETCPACK](https://github.com/Ericsson/ETCPACK), under Ericsson's licence; downloaded at build
  time, not redistributed here.
- Block decoding: [texture2ddecoder](https://github.com/K0lb3/texture2ddecoder).

## Licence

- **Code:** MIT, see [LICENSE](LICENSE).
- **Model files** (the `.pt` files on the Releases page): not covered by the MIT licence. They are provided as-is for
  personal, non-commercial use.
- **ETCPACK** is Ericsson's and stays under Ericsson's licence; `build_etcpack.sh` downloads it to your machine.

This is a fan project and is not affiliated with the game's developer or publisher. No game assets are included.
