#!/Users/elm/tools/.venv/bin/python
"""Real-browser screenshots (Playwright Chromium, JS on).
usage: shot.py page.html outdir [--routes home,research,...] [--full] [--widths 1440,390] [--webgl]
--webgl launches Chromium with a WebGL backend chosen for the host: ANGLE on Metal (the real GPU) on macOS,
ANGLE + SwiftShader (software) elsewhere. Without it headless Chromium still has WebGL through SwiftShader, but the
software compositor's canvas readback logs "GPU stall due to ReadPixels" warnings.
Per width and route (#route hash; '' = no hash) saves <name>-<w>-<route>.png (viewport, or full page with --full,
split into 1600px-high tiles). Prints console errors, page errors, horizontal overflow and failed requests."""
import sys, os, argparse
from playwright.sync_api import sync_playwright
ap = argparse.ArgumentParser(); ap.add_argument("page"); ap.add_argument("outdir")
ap.add_argument("--routes", default=""); ap.add_argument("--full", action="store_true")
ap.add_argument("--widths", default="1440,390"); ap.add_argument("--tile", type=int, default=1600)
ap.add_argument("--webgl", action="store_true")
a = ap.parse_args(); os.makedirs(a.outdir, exist_ok=True)
if a.page.startswith("http"):
    import re as _re; url = a.page; name = _re.sub(r"[^a-z0-9]+", "-", a.page.split("//", 1)[1].lower()).strip("-")[:60]
else:
    url = "file://" + os.path.abspath(a.page); name = os.path.splitext(os.path.basename(a.page))[0]
routes = a.routes.split(",") if a.routes else [""]
with sync_playwright() as p:
    gl_args = (["--use-angle=metal"] if sys.platform == "darwin" else ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    b = p.chromium.launch(args=gl_args if a.webgl else [])
    for w in [int(x) for x in a.widths.split(",")]:
        ctx = b.new_context(viewport={"width": w, "height": 900 if w > 600 else 844}, device_scale_factor=1)
        pg = ctx.new_page(); errs = []
        pg.on("console", lambda m: errs.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        pg.on("requestfailed", lambda r: errs.append(f"requestfailed: {r.url[:120]}"))
        for r in routes:
            pg.goto(url + (f"#{r}" if r else ""), wait_until="load", timeout=45000); pg.wait_for_timeout(1500 if url.startswith("http") else 700)
            # scroll through so IntersectionObserver reveals fire, then back to top
            h = pg.evaluate("document.documentElement.scrollHeight")
            for y in range(0, h, 700): pg.evaluate(f"window.scrollTo(0,{y})"); pg.wait_for_timeout(60)
            pg.evaluate("window.scrollTo(0,0)"); pg.wait_for_timeout(300)
            ov = pg.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            tag = r or "top"; base = os.path.join(a.outdir, f"{name}-{w}-{tag}")
            if a.full:
                h = pg.evaluate("document.documentElement.scrollHeight")
                for i, y in enumerate(range(0, h, a.tile)):
                    pg.screenshot(path=f"{base}-{i:02d}.png", full_page=True,
                                  clip={"x": 0, "y": y, "width": w, "height": min(a.tile, h - y)})
                print(f"{w}px #{tag}: height {h}px -> {len(range(0, h, a.tile))} tiles; h-overflow {ov}px")
            else:
                pg.screenshot(path=base + ".png"); print(f"{w}px #{tag}: viewport shot; h-overflow {ov}px")
        for e in sorted(set(errs)): print("  ", e)
        ctx.close()
    b.close()
