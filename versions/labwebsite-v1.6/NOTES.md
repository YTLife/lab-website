# LabWebsite v1.6: notes (2026-10-09)

v1.6 is v1.5 with five views (Home · Research · People · News · Contact): a more general Home, Research rebuilt on the v2.0 Research template without direction C (Framework and Applications folded in), and Publications renamed News. Visual system, hero, fold and build/check procedure as v1.5 / v1.4 (`versions/labwebsite-v1.4/NOTES.md`). Source edited in place over several rounds the same day; this file describes the final state.

## Home
- Order: hero · 00 Thesis · 01 Research (two scale cards) · 02 Framework (flow) · 03 People · 04 News. Removed: 03 Application domain, 04 Methods, 07 Collaborate (duplicated Contact). 06 Selected work replaced by 04 News.
- 04 News: `sec-head` micro "04 / News", h2 "Lab news." (`a.hl` -> `#news`), lede "Papers, grants, awards and people from across the lab.", four `.plist` rows newest first (2026 Paper Cong et al. DOI ↗; 2026 Award Reese, Purdue Online Award, "Purdue ↗" to the Purdue Statistics news page; 2025 Grant NIH/NIA R03 1R03AG102039, "NIH RePORTER ↗" to project-details/11250186; 2024 Award Reese, BMVC reviewer, no link), CONFIRM WITH PI comment, more-row "All news & publications →" `#news`. A linkless row has two children only; an empty third span would add a 14 px grid row at <=740.
- Links: hero "How the work connects" and 02 h2 -> `#research-loop`; 01 card feet -> `#research-a` / `#research-b`.
- 01 rows link into Research: each of the eight `.srows` rows wraps its label + keywords in `<a href="#research-a-body|motion|intelligence|extending">` / `#research-b-outcomes|molecules|inference|personalization`. The link spans the row grid and the full row height (`margin:-12px 0; padding:12px 0`), hover inks the label (gold `.fut` labels stay gold), focus ring is the site's `:focus-visible`. Row reveal animation unchanged (it now fades the link as one).

## Research (`#research`, v2.0 template, no C)
- phead "01 / Research", h1 "Two scales, one loop.", v2.0 lede. Sub-nav: Loop · A · Movement · B · Biological · Applications (scrollspy as before).
- `#research-loop` (light), `#research-a` (paper = mist ground), `#research-b` (light): markup copied from v2.0; differences only ring node 4 "Learn" -> `#research-b`, A's manuscript card link -> `#news-papers`, ids on the eight path stages (`research-a-body` … `research-b-personalization`).
- `#research-applications` (paper): `dir-head` with micro "Application domain", h2.dir-q "Health-oriented movement as a scientific model system.", the Tai Chi/Qigong sentence as `.dir-aim`, then the Q1–Q4 `ol.cells.c4` (aria-label "Questions we want to understand"; the "Questions ·" label and lead-split paragraphs dropped).
- Removed: v1.5 movement/biological area blocks, methods cells, the merged framework/statement/applications blocks, the Framework and Applications views.
- Path stages: `#research .path li[id]{scroll-margin-top:160px}` (148 at <=740), so a stage lands 35 px (31 at 390) under the sticky sub-nav. No `:target` highlight pattern exists on the site, so none was added; the router focuses the stage's h3.
- CSS copied from v2.0 (`.loop-txt`, `.disc`, `.dir-*`, `.path`, `.vision.where`, `.wgrid`, `.wcard`, `.wvis`, `.vtag`, `.scl` + its draw-in, `.where-side`); no `.dcard`, `.rvis-c`, `.gen`/`.chk`, mission or Home-only rules. `.bg-paper .cells` rules get the firmer `--line2` rule colour.
- JS: RV list + `.dir-head>*,.path,.vision.where,.wcard`. Module: three body views from one male pack decode + the hero: Home 01 A (`#home-body`, fn = `#home .rvis-b`) and Research A (`#research-body`, fn = `.wvis .rvis-b`, zone `.wgrid-a[data-loop]`); `fnPanel` reveal waits on `.rcard,.wcard`. The v1.6 and v2.0 module scripts were otherwise identical (diffed).
- Footer Lab column: Research · Body & motion · Molecular response.
- Dead CSS (v1.5 area, steps-list, lead-split, app/qlist, m-layout rules and their RV selectors) left in place: pruning was not done because no pixel-identity pass was run.

## News (was Publications)
- View id `news`; nav "News"; phead "03 / News", h1 "News & publications.", same lede as Home 04. Blocks: `#news-lab` (Lab news, same four rows) · `#news-papers` (Publications, 17 papers, fold `#news-more`) · `#news-projects`. People "Paper" chip -> `#news-papers`. Lab news rows sit in the `.pub-group` gutter (one top rule, 64 px year column so text aligns with the papers).
- People "02 / People", Contact "04 / Collaborate".

## Routing
- `ok` list, CSS `data-view` rule, JS `NAMES`: home research people news contact. Old hashes: `#framework`, `#applications`, `#publications` -> Home; `#research-movement` (and other unknown `research-*`) -> Research at the top.

## Files
`versions/labwebsite-v1.6/index.html` 184,174 B. `dist/labwebsite-v1.6.html` 2,229,731 B. No artifact build, not published.

## Checks (final dist; Playwright Chromium, Metal WebGL, 1440 and 390, normal and reduced motion)
- jscheck: source 4 inline scripts, dist 3 inline scripts + 2 data blocks, syntax OK, in-page links without target id: none.
- shot.py routes home, research, research-a, research-b, research-applications, people, news, contact: h-overflow 0 px everywhere; no console/page errors. Heights 1440: home 5050, research 4467, people 2286, news 2770, contact 2144; 390: 8313 / 6863 / 3554 / 3949 / 2762.
- Probe (all four passes): 0 errors; 5 nav links with the right aria-current per route; section hashes land at 124 px (108 at 390) with the matching sub-nav item on; scrollspy Loop / A / B / Applications while scrolling, Applications at the page bottom. `.gl-on` 4 (hero figure + stage, Home A, Research A), Home `.rvis-b.fn-on`, Research `.wvis .rvis-b.fn-on`. Eight path dots 9 px each (done grey, now teal, future gold ring, plain hollow); both `ol.path` opacity 1; B schematic curves dashoffset 0 after reveal. Bands: loop light, A mist, B light, applications mist, footer light.
- Home rows: each of the 8 clicks opens Research with the stage li at y 160 (header bottom 73, sub-nav bottom 125: +87 / +35) at 1440 and at y 148 (65 / 117: +83 / +31) at 390; focus on the stage h3 (tabindex -1); sub-nav shows A or B. Keyboard: Tab reaches the row link (2 px outline), Enter does the same. Hover: label rgb(48,83,90) -> ink; gold rows stay rgb(155,123,57).
- Fold on #news: false/inert/4 papers -> true/17 -> false/inert/4; `#news-more` opens it; `#news-papers` lands at 96 px (80 at 390).
- Shots: `dist/shots-v1.6/labwebsite-v1.6-{1440,390}-<route>-NN.png` (56 tiles; full-page tiles can catch reveals mid-way).
