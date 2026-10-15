#!/usr/bin/env python3
"""Report colors in a page that are not in the advisor's palette (original preview).
Allowed: exact original hex colors, rgba() of any original rgb triple at any alpha,
and hex colors within a small distance of an original color (DELTA)."""
import sys, re, colorsys
ORIG = "research_lab_website_preview (2).html"
DELTA = 18  # max per-channel-ish euclidean distance for 'same color'
def hexes(s):
    out=set()
    for h in re.findall(r'#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b', s):
        if len(h)==3: h=''.join(c*2 for c in h)
        out.add(tuple(int(h[i:i+2],16) for i in (0,2,4)))
    for r,g,b in re.findall(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', s):
        out.add((int(r),int(g),int(b)))
    return out
import os
root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
strip=lambda s: re.sub(r'data:[a-z/+]+;base64,[A-Za-z0-9+/=]+','',s)
orig=hexes(strip(open(os.path.join(root,ORIG),encoding="utf-8").read()))
for p in sys.argv[1:]:
    cand=hexes(strip(open(p,encoding="utf-8").read()))
    bad=[]
    for c in sorted(cand):
        d=min(sum((a-b)**2 for a,b in zip(c,o))**.5 for o in orig)
        if d>DELTA:
            h,l,s_=colorsys.rgb_to_hls(*[x/255 for x in c])
            bad.append((c,round(d),round(h*360),round(s_,2),round(l,2)))
    print(f"{p}: {len(cand)} colors, {len(bad)} outside palette (dist>{DELTA})")
    for c,d,h,s_,l in bad: print(f"   #{c[0]:02x}{c[1]:02x}{c[2]:02x}  dist={d}  hue={h} sat={s_} light={l}")
