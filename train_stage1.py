#!/usr/bin/env python3
"""Train (or fine-tune) the stage-1 model on clean art damaged with the game's encoder.

  python train_stage1.py --name myrun --data CLEAN_DIR[:weight] [--data ...] --init models/echoclean_stage1.pt

--data  a folder of clean RGBA PNGs (searched recursively), or a text file listing image paths. Add ':weight' to control how
        often a set is sampled (default 1), e.g.  --data echocalypse_clean:0.4 --data other_games:0.6
--val-raw / --val-clean  an optional real pair (raw game texture + its clean original) scored every --eval-every steps.
        Keep that image out of --data.
Writes runs/<name>/ckpt_last.pt, ckpt_best.pt (best validation PSNR) and log.txt.
"""
import os, sys, time, argparse
import numpy as np, torch
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True); ap.add_argument('--data', action='append', required=True)
    ap.add_argument('--init', default=None, help='checkpoint to continue from (omit to train from scratch)')
    ap.add_argument('--channels', type=int, default=256); ap.add_argument('--blocks', type=int, default=8)
    ap.add_argument('--steps', type=int, default=40000); ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=None, help='default: 2e-4 from scratch, 5e-5 when fine-tuning'); ap.add_argument('--lr-low', type=float, default=None, help='rate for the last 30%% of steps (default: lr / 3.3)')
    ap.add_argument('--batch', type=int, default=12); ap.add_argument('--patch', type=int, default=96)
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--pool', type=int, default=6, help='clean images each worker keeps in memory per data set')
    ap.add_argument('--hue-weight', type=float, default=2.0, help='straight-colour loss on see-through texels')
    ap.add_argument('--seam-weight', type=float, default=1.0, help='loss on the texture composited over its own colour')
    ap.add_argument('--ssim-weight', type=float, default=1.0, help='SSIM loss on that composite')
    ap.add_argument('--val-raw'); ap.add_argument('--val-clean')
    a = ap.parse_args()
    from echoclean import blocks as B, models, metrics as MT, data as D, pipeline as PL
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = os.path.join('runs', a.name); os.makedirs(out, exist_ok=True); log = open(os.path.join(out, 'log.txt'), 'a')
    def say(*x):
        s = ' '.join(str(v) for v in x); print(s, flush=True); log.write(s + '\n'); log.flush()
    say(f'=== {time.strftime("%Y-%m-%d %H:%M")}  ' + ' '.join(sys.argv[1:]))
    groups = [D.list_images(s) for s in a.data]; say('data:', ', '.join(f'{len(f)} images (weight {w:g})' for f, w in groups))
    net = models.load_stage1(a.init, dev) if a.init else models.SideNet(a.channels, a.blocks).to(dev)
    if not a.init: torch.nn.init.zeros_(net.tail.weight); torch.nn.init.zeros_(net.tail.bias)      # start as "output = input"
    net.train(); lr = a.lr or (5e-5 if a.init else 2e-4); lr_low = a.lr_low or lr / 3.3; opt = torch.optim.Adam(net.parameters(), lr=lr)
    say(f'model {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters on {dev}, ' + (f'continuing from {a.init}' if a.init else 'from scratch') + f', lr {lr:g} then {lr_low:g}')
    stream = D.forever(DataLoader(D.PatchStream(groups, a.patch // 4 * 4, a.batch, a.pool), batch_size=None, num_workers=a.workers, prefetch_factor=6))
    val = None
    if a.val_raw and a.val_clean:
        vb, vw, vh = B.load_raw(a.val_raw); val = (vb, vw, vh, B.load_clean(a.val_clean))
    lumw = torch.tensor([0.3, 0.59, 0.11], device=dev).view(1, 3, 1, 1); best = -1.0
    def evaluate(step):
        nonlocal best
        models.save(net, os.path.join(out, 'ckpt_last.pt'), step=step, kind='stage1')
        if val is None: return
        net.eval(); r = MT.score(PL.finish(PL.stage1(net, val[0], dev, tta=False, half=False))[:val[2], :val[1]], val[3], dev); net.train()
        say(f'--- step {step}: validation pair  {MT.fmt(r)}')
        if r['psnr'] > best: best = r['psnr']; models.save(net, os.path.join(out, 'ckpt_best.pt'), step=step, kind='stage1')
    evaluate(0); t0 = time.time()
    for s in range(a.steps):
        for g in opt.param_groups: g['lr'] = (lr if s < int(a.steps * 0.7) else lr_low) * (min(1.0, (s + 1) / 2000) if not a.init else 1.0)
        x, y, _ = next(stream); x = x.permute(0, 3, 1, 2).to(dev); y = (y.permute(0, 3, 1, 2).float() / 255).to(dev)
        o = net(x); ya = y[:, 3:4]; w = 1 + 3 * ((ya > 0) & (ya < 1)).float()
        loss = (w * (o - y).abs()).mean()
        if a.hue_weight > 0: loss = loss + a.hue_weight * ((ya > 2 / 255).float() * (o[:, :3] - y[:, :3]).abs() / ya.clamp(min=0.03)).mean()
        if a.seam_weight > 0 or a.ssim_weight > 0:
            with torch.no_grad(): S, valid = MT.own_colour(y, sigmas=(1.5, 3, 6, 12))
            if a.seam_weight > 0:
                e = ((o[:, :3] - y[:, :3]) - S * (o[:, 3:4] - ya)) * valid * w
                loss = loss + a.seam_weight * (e.abs().mean() + 2.0 * MT.gauss((e * lumw).sum(1, keepdim=True), 1.5).abs().mean())
            if a.ssim_weight > 0:
                sm = MT.ssim_map(o[:, :3] + (1 - o[:, 3:4]) * S, y[:, :3] + (1 - ya) * S).mean(1, keepdim=True)
                loss = loss + a.ssim_weight * ((1 - sm) * valid * w).sum() / (valid * w).sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        if (s + 1) % 500 == 0:
            el = time.time() - t0; print(f'step {s + 1}/{a.steps}  loss {loss.item():.5f}  {(s + 1) / el:.1f} steps/s  ~{el / (s + 1) * (a.steps - s - 1) / 60:.0f} min left', flush=True)
        if (s + 1) % a.eval_every == 0 or s + 1 == a.steps: evaluate(s + 1)
    say(f'done in {(time.time() - t0) / 60:.0f} min' + (f'; best validation PSNR {best:.2f} dB' if val else '')); os._exit(0)


if __name__ == '__main__':
    main()
