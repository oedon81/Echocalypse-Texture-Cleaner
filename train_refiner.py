#!/usr/bin/env python3
"""Train (or fine-tune) the stage-2 refiner on top of a finished stage-1 model.

  python train_refiner.py --name myrefiner --stage1 models/echoclean_stage1.pt --data CLEAN_DIR[:weight] [--init models/echoclean_refiner.pt]

For every training patch the stage-1 output is re-compressed with the game's encoder; the refiner gets the texture, that
output, how the re-compression differs from the game's data, and which blocks do not match, and learns the correction.
Options as in train_stage1.py. Writes runs/<name>/ckpt_last.pt, ckpt_best.pt and log.txt.
"""
import os, sys, time, argparse
import numpy as np, torch
from torch.utils.data import DataLoader
from multiprocessing import get_context


def _enc(a):
    from echoclean import encoder
    return encoder.encode_blocks(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True); ap.add_argument('--stage1', required=True); ap.add_argument('--data', action='append', required=True)
    ap.add_argument('--init', default=None); ap.add_argument('--channels', type=int, default=128); ap.add_argument('--blocks', type=int, default=8)
    ap.add_argument('--steps', type=int, default=30000); ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=None, help='default: 1e-4 from scratch, 3e-5 when fine-tuning'); ap.add_argument('--lr-low', type=float, default=None)
    ap.add_argument('--batch', type=int, default=12); ap.add_argument('--patch', type=int, default=96); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--pool', type=int, default=6)
    ap.add_argument('--seam-weight', type=float, default=1.0); ap.add_argument('--val-raw'); ap.add_argument('--val-clean')
    a = ap.parse_args()
    from echoclean import blocks as B, models, metrics as MT, data as D, pipeline as PL, encoder
    enc_pool = get_context('spawn').Pool(4, maxtasksperchild=400)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = os.path.join('runs', a.name); os.makedirs(out, exist_ok=True); log = open(os.path.join(out, 'log.txt'), 'a')
    def say(*x):
        s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
    say(f'=== {time.strftime("%Y-%m-%d %H:%M")}  ' + ' '.join(sys.argv[1:]))
    groups = [D.list_images(s) for s in a.data]; say('data:', ', '.join(f'{len(f)} images (weight {w:g})' for f, w in groups))
    net1 = models.load_stage1(a.stage1, dev); ref = models.load_refiner(a.init, dev) if a.init else models.Refiner(a.channels, a.blocks).to(dev)
    ref.train(); lr = a.lr or (3e-5 if a.init else 1e-4); lr_low = a.lr_low or lr / 3.3; opt = torch.optim.Adam(ref.parameters(), lr=lr)
    say(f'refiner {sum(p.numel() for p in ref.parameters()) / 1e6:.2f}M parameters on {dev}, lr {lr:g} then {lr_low:g}')
    stream = D.forever(DataLoader(D.PatchStream(groups, a.patch // 4 * 4, a.batch, a.pool), batch_size=None, num_workers=a.workers, prefetch_factor=6))
    val = None
    if a.val_raw and a.val_clean:
        vb, vw, vh = B.load_raw(a.val_raw); val = (vb, vw, vh, B.load_clean(a.val_clean))
    lumw = torch.tensor([0.3, 0.59, 0.11], device=dev).view(1, 3, 1, 1); best = -1.0
    u8 = lambda t: (t.clamp(0, 1) * 255 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    @torch.no_grad()
    def evaluate(step):
        nonlocal best
        models.save(ref, os.path.join(out, 'ckpt_last.pt'), step=step, kind='refiner')
        if val is None: return
        ref.eval(); vb, vw, vh, vy = val; x = B.decode(vb); o1 = PL.finish(PL.stage1(net1, vb, dev, tta=False, half=False))
        inp = PL.refiner_input(x, o1, vb, encoder.encode_blocks_mp(o1, enc_pool))
        t = torch.nn.functional.pad(torch.from_numpy(inp).to(dev).permute(2, 0, 1)[None], (PL.PAD,) * 4)
        o = PL.finish(PL._tiles(ref, t, x.shape[0], x.shape[1], 1024, False).clamp_(0, 1).mul_(255).permute(1, 2, 0).cpu().numpy()); ref.train()
        r = MT.score(o[:vh, :vw], vy, dev)
        if step == 0: say(f'--- stage 1 alone:  {MT.fmt(MT.score(o1[:vh, :vw], vy, dev))}')
        say(f'--- step {step}: validation pair  {MT.fmt(r)}')
        if r['psnr'] > best: best = r['psnr']; models.save(ref, os.path.join(out, 'ckpt_best.pt'), step=step, kind='refiner')
    evaluate(0); t0 = time.time()
    for s in range(a.steps):
        for g in opt.param_groups: g['lr'] = (lr if s < int(a.steps * 0.7) else lr_low) * min(1.0, (s + 1) / 1000)
        x13, y, bx = next(stream); x13 = x13.permute(0, 3, 1, 2).to(dev); y = (y.permute(0, 3, 1, 2).float() / 255).to(dev); x = x13[:, :4]; bx = bx.numpy()
        with torch.no_grad(): o1 = net1(x13)
        bo = np.stack(enc_pool.map(_enc, list(u8(o1))))
        fb = torch.from_numpy(np.stack([(B.decode(p).astype(np.float32) - B.decode(q)) / 255 for p, q in zip(bo, bx)])).permute(0, 3, 1, 2).to(dev)
        mm = torch.from_numpy(np.repeat(np.repeat((bo != bx).any(-1).astype(np.float32), 4, 1), 4, 2)[:, None]).to(dev)
        o = ref(torch.cat([x, o1, fb, mm], 1)); ya = y[:, 3:4]; w = 1 + 3 * ((ya > 0) & (ya < 1)).float()
        loss = (w * (o - y).abs()).mean() + 2.0 * ((ya > 2 / 255).float() * (o[:, :3] - y[:, :3]).abs() / ya.clamp(min=0.03)).mean()
        if a.seam_weight > 0:
            with torch.no_grad(): S, valid = MT.own_colour(y, sigmas=(1.5, 3, 6, 12))
            e = ((o[:, :3] - y[:, :3]) - S * (o[:, 3:4] - ya)) * valid * w
            loss = loss + a.seam_weight * (e.abs().mean() + 2.0 * MT.gauss((e * lumw).sum(1, keepdim=True), 1.5).abs().mean())
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(ref.parameters(), 1.0); opt.step()
        if (s + 1) % 500 == 0:
            el = time.time() - t0; print(f'step {s + 1}/{a.steps}  loss {loss.item():.5f}  {(s + 1) / el:.1f} steps/s  ~{el / (s + 1) * (a.steps - s - 1) / 60:.0f} min left', flush=True)
        if (s + 1) % a.eval_every == 0 or s + 1 == a.steps: evaluate(s + 1)
    say(f'done in {(time.time() - t0) / 60:.0f} min' + (f'; best validation PSNR {best:.2f} dB' if val else '')); os._exit(0)


if __name__ == '__main__':
    main()
