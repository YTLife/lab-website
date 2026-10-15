#!/usr/bin/env python3
"""Render an HTML page to PNG tiles with macOS Quick Look (no browser needed).
usage: render.py page.html outdir [--mobile] [--tiles N]
Desktop: page rendered at ~1250 CSS px viewport, one 1400px PNG per tile.
Mobile: page shown inside a 390px-wide iframe (media queries apply)."""
import sys, os, re, shutil, subprocess, tempfile, argparse
ap = argparse.ArgumentParser()
ap.add_argument("page"); ap.add_argument("outdir")
ap.add_argument("--mobile", action="store_true"); ap.add_argument("--tiles", type=int, default=8)
a = ap.parse_args()
page = os.path.abspath(a.page); os.makedirs(a.outdir, exist_ok=True)
src = open(page, encoding="utf-8").read()
base = os.path.splitext(os.path.basename(page))[0] + ("-m" if a.mobile else "-d")
tmp = tempfile.mkdtemp(dir=os.path.dirname(page))
try:
    step = 1050 if a.mobile else 1250
    for k in range(a.tiles):
        off = k * step
        if a.mobile:
            inner = os.path.join(tmp, "inner.html")
            shutil.copy(page, inner)
            html = (f'<!doctype html><html><body style="margin:0;background:#888;overflow:hidden">'
                    f'<div style="width:390px;height:1250px;overflow:hidden;position:relative;margin:0 auto">'
                    f'<iframe src="inner.html" style="position:absolute;top:-{off}px;left:0;width:390px;height:{off+1300}px;border:0"></iframe>'
                    f'</div></body></html>')
        else:
            inj = f'<style>html,body{{overflow:hidden!important}}body{{transform:translateY(-{off}px)}}.topbar,header.topbar{{position:relative!important}}</style>'
            html = re.sub(r'</head>', inj + '</head>', src, count=1, flags=re.I)
        f = os.path.join(tmp, f"{base}-{k:02d}.html")
        open(f, "w", encoding="utf-8").write(html)
        subprocess.run(["qlmanage", "-t", "-s", "1400", "-o", a.outdir, f], capture_output=True)
        png = os.path.join(a.outdir, os.path.basename(f) + ".png")
        if os.path.exists(png): os.replace(png, os.path.join(a.outdir, f"{base}-{k:02d}.png"))
    print("rendered", a.tiles, "tiles to", a.outdir)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
