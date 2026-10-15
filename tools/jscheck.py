#!/usr/bin/env python3
"""Syntax-check inline <script> blocks with node --check; also list anchor hrefs without a matching id.
type="importmap" blocks are parsed as JSON; type="module" blocks are checked as ES modules (.mjs);
blocks of any other non-JavaScript type (e.g. text/plain data) are skipped and counted."""
import sys, re, subprocess, tempfile, os, json
for p in sys.argv[1:]:
    s = open(p, encoding="utf-8").read()
    blocks = re.findall(r'<script(?![^>]*\bsrc=)([^>]*)>(.*?)</script>', s, flags=re.S | re.I)
    ok = True; data = 0
    for i, (attrs, b) in enumerate(blocks):
        t = re.search(r'type=["\']([^"\']+)["\']', attrs)
        t = t.group(1).lower() if t else ""
        if t not in ("", "text/javascript", "application/javascript", "module", "importmap"):
            data += 1; continue
        if t == "importmap":
            try: json.loads(b)
            except ValueError as e: ok = False; print(f"script {i}: IMPORT MAP IS NOT JSON: {e}")
            continue
        suffix = ".mjs" if t == "module" else ".js"
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f: f.write(b); fn = f.name
        r = subprocess.run(["node", "--check", fn], capture_output=True, text=True); os.unlink(fn)
        if r.returncode: ok = False; print(f"script {i}: SYNTAX ERROR\n{r.stderr[-800:]}")
    ids = set(re.findall(r'\bid=["\']([^"\']+)["\']', s))
    missing = sorted({h for h in re.findall(r'href=["\']#([^"\'/][^"\']*)["\']', s) if h not in ids})
    print(f"{p}: {len(blocks) - data} inline scripts{f' (+{data} data blocks skipped)' if data else ''}, syntax {'OK' if ok else 'FAILED'}; in-page links without target id: {missing or 'none'}")
