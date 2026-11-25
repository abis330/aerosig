"""Figure: anomaly score + top channel z-scores for one flight, with the injected fault onset.

python scripts/plot_detection.py --lake data/lake --art artifacts --flight FLT-00054 --out docs/detection.png
"""
import argparse

import matplotlib.pyplot as plt
from flightlake import Lake

from aerosig.score import Detector

BLUE, ORANGE, AQUA, INK, MUTED = "#257de0", "#f47230", "#00b894", "#1f1f1f", "#8a8a8a"

ap = argparse.ArgumentParser()
ap.add_argument("--lake", default="data/lake")
ap.add_argument("--art", default="artifacts")
ap.add_argument("--flight", required=True)
ap.add_argument("--out", default="docs/detection.png")
a = ap.parse_args()

lake, det = Lake(a.lake), Detector.load(a.art)
tel = lake.telemetry(a.flight)
s = det.score_flight(tel)
meta = lake.summary(a.flight)["meta"]
fault = meta["faults"][0] if meta["faults"] else None
ev = det.events(s)
zc = [c for c in s.columns if c.startswith("z_")]
post = s[s.eligible & (s.t_end > (fault["t_start"] if fault else 0))]
top = post[zc].median().sort_values(ascending=False).index[:2]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    ax.tick_params(colors=MUTED)
    if fault:
        ax.axvline(fault["t_start"], color=INK, lw=1, ls="--")
s_el = s[s.eligible]
ax1.plot(s_el.t_end, s_el.score, color=BLUE, lw=2)
ax1.axhline(det.calib["threshold"], color=MUTED, lw=1, ls=":")
ax1.text(s_el.t_end.min(), det.calib["threshold"], " alarm threshold", va="bottom", color=MUTED, fontsize=9)
for e in ev:
    ax1.axvspan(e["t_start"], e["t_end"], color=ORANGE, alpha=0.12, lw=0)
ax1.set_ylabel("anomaly score", color=INK)
title = f"{a.flight}: injected {fault['type']} at t={fault['t_start']:.0f} s" if fault else a.flight
ax1.set_title(title, loc="left", color=INK, fontsize=11)
for c, col in zip(top, (ORANGE, AQUA)):
    ax2.plot(s_el.t_end, s_el[c], color=col, lw=2)
    ax2.text(s_el.t_end.max() + 5, s_el[c].iloc[-1], c[2:], color=INK, fontsize=9, va="center")
ax2.set_ylabel("error ÷ nominal p99", color=INK)
ax2.set_xlabel("time since log start (s)", color=INK)
if fault:
    ax1.text(fault["t_start"], ax1.get_ylim()[1], " onset", color=INK, fontsize=9, va="top")
fig.tight_layout()
fig.savefig(a.out, dpi=150)
print(a.out)
