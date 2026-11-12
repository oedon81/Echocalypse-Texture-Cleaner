#!/usr/bin/env python3
"""Download raw hero textures from the game's public update CDN (only the files asked for, via HTTP range requests).

  python tools/fetch_textures.py list   [--region tw]                 print every hero id on that CDN
  python tools/fetch_textures.py get    ID [ID ...] -o raw            raw/<id>/<files>  (textures, atlas, skeleton)
  python tools/fetch_textures.py censor ID [ID ...] -o raw_censor     the CN "censor pack" of a hero, if it has one

Regions: tw and jp carry every hero; global carries the newer ones (shared files are byte-identical on all three).
About censor packs: for many older heroes the pack holds the SAME art encoded from the original picture, while the
normal file was made from a dithered export, so the pack is the better input for the cleaner (roughly 3-4 dB closer to
the original before cleaning). A few packs are really censored or use a different layout: compare before use.
"""
import os, io, re, sys, json, zipfile, argparse, urllib.request
UA = {'User-Agent': 'Mozilla/5.0'}
REGIONS = {'global': 'https://version-amj.gtarcade.com/opinfo?opGameId=3054&opId=2942&channel=0&md5=',
           'tw': 'https://version-twjgame.gtarcade.com/opinfo?opGameId=1853&opId=3268&channel=0&md5=',
           'jp': 'https://version-jpjgame.gtarcade.com/opinfo?opGameId=2972&opId=2335&channel=0&md5='}
CENSOR = 'https://cdn-wx-studio.uuzuonline.com/product-2013001/cdn1/upgrade_hquality'


def fetch(url, rng=None, timeout=120):
    h = dict(UA)
    if rng: h['Range'] = 'bytes=%d-%d' % rng
    for t in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r: return r.read()
        except Exception:
            if t == 4: raise


class RangeFile(io.RawIOBase):
    BLK = 1 << 20
    def __init__(s, url, size): s.url, s.size, s.pos, s.cache = url, size, 0, {}
    def readable(s): return True
    def seekable(s): return True
    def tell(s): return s.pos
    def seek(s, off, wh=0):
        s.pos = off if wh == 0 else s.pos + off if wh == 1 else s.size + off; return s.pos
    def read(s, n=-1):
        if n is None or n < 0: n = s.size - s.pos
        n = max(0, min(n, s.size - s.pos)); out = bytearray()
        while n:
            i, o = divmod(s.pos, s.BLK)
            if i not in s.cache:
                if len(s.cache) > 24: s.cache.pop(next(iter(s.cache)))
                s.cache[i] = fetch(s.url, (i * s.BLK, min(s.size, (i + 1) * s.BLK) - 1))
            d = s.cache[i][o:o + n]
            if not d: break
            out += d; s.pos += len(d); n -= len(d)
        return bytes(out)
    def readinto(s, b):
        d = s.read(len(b)); b[:len(d)] = d; return len(d)


def index(region, cache_dir):
    path = os.path.join(cache_dir, f'index_{region}.json')
    if os.path.exists(path): return json.load(open(path))
    c = json.loads(fetch(REGIONS[region])); c = c.get('config', c); cfg = json.loads(fetch(c['android_upgrade_config'])); base = cfg['upgrade_url']; z = {}
    for v in cfg['versions'].values():
        for e in (v if isinstance(v, list) else [v]): z[e['url']] = e['size']
    res = {}
    for i, (u, s) in enumerate(sorted(z.items()), 1):
        for x in zipfile.ZipFile(RangeFile(base + '/' + u, s)).infolist():
            if '/knight_spine/' in x.filename and not x.is_dir(): res[x.filename] = [base + '/' + u, s, x.file_size]
        print(f'\rreading package lists {i}/{len(z)}', end='', flush=True)
    print(); os.makedirs(cache_dir, exist_ok=True); json.dump(res, open(path, 'w')); return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['list', 'get', 'censor']); ap.add_argument('ids', nargs='*')
    ap.add_argument('--region', default='tw', choices=sorted(REGIONS)); ap.add_argument('-o', '--out', default='raw')
    a = ap.parse_args(); gid = lambda p: re.search(r'knight_spine/([^/]+)/', p).group(1)
    if a.cmd == 'censor':
        for cid in a.ids:
            d = os.path.join(a.out, cid); os.makedirs(d, exist_ok=True)
            try: zf = zipfile.ZipFile(io.BytesIO(fetch(f'{CENSOR}/ks_{cid}.zip', timeout=600)))
            except Exception as e: print(cid, 'no censor pack:', type(e).__name__); continue
            for x in zf.infolist():
                if not x.is_dir(): open(os.path.join(d, os.path.basename(x.filename)), 'wb').write(zf.read(x))
            print(cid, 'censor pack ->', d)
        return
    ix = index(a.region, a.out)
    if a.cmd == 'list':
        ids = sorted({gid(k) for k in ix}, key=lambda s: (len(s), s)); print(len(ids), 'hero ids on', a.region); print(' '.join(ids)); return
    opened = {}
    for cid in a.ids:
        files = sorted(k for k in ix if f'/knight_spine/{cid}/' in k)
        if not files: print(cid, 'not on this CDN'); continue
        for k in files:
            url, size, fsize = ix[k]; dst = os.path.join(a.out, cid, os.path.basename(k))
            if os.path.exists(dst) and os.path.getsize(dst) == fsize: continue
            if url not in opened: opened[url] = zipfile.ZipFile(RangeFile(url, size))
            os.makedirs(os.path.dirname(dst), exist_ok=True); open(dst, 'wb').write(opened[url].read(k))
        print(cid, len(files), 'files ->', os.path.join(a.out, cid), flush=True)


if __name__ == '__main__':
    main()
