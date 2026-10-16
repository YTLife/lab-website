# Lab website — source (LabWebsite v1.7)

Human Motion & Molecular Health Lab, Purdue University (Michael Yu Zhu, Timothy Reese; lab manager Yuyue Zhang).
Live site: https://elwyn921.github.io/lab-website/

## Layout
- `versions/labwebsite-v1.7/index.html` (v1.6 kept beside it) — the editable source page (five views: Home · Research · People · News · Contact). `NOTES.md` describes the design decisions and checks.
- `shared/assets/` — assets referenced by the source: logo, portraits, pose figures, the two Tai Chi motion packs (`motion/taiji-umons-{f,m}.bin`, format in `shared/MOTION_FORMAT.md`) and the vendored three.js r170.
- `tools/build.py` — builds the single self-contained `dist/<id>.html` (three.js and motion packs inlined, no fetch).
- `dist/labwebsite-v1.7.html` — the built page; this exact file is what is deployed as `index.html` on GitHub Pages.

## Build
```
python3 tools/build.py versions/labwebsite-v1.7/index.html   # -> dist/labwebsite-v1.7.html
python3 tools/jscheck.py dist/labwebsite-v1.7.html            # syntax + dead in-page links
python3 tools/shot.py dist/labwebsite-v1.7.html --webgl       # Playwright screenshots (optional)
```
Opening `versions/labwebsite-v1.7/index.html` directly from disk also works in a browser (it loads assets relatively).

## Deploy
Copy `dist/labwebsite-v1.7.html` to `index.html` in the Pages repo (`Elwyn921/lab-website`, branch `main`) and push; Pages rebuilds in about a minute.
