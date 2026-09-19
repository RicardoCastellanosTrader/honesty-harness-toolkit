# -*- coding: utf-8 -*-
"""
plot_ladder.py — Figure: the search-size ladder (best-of-N on pure noise vs BTC).

Deterministic and reproducible: reads examples/toy/placebo_verdict_results.json
(produced by placebo_verdict.py --run with the FIXED seeds listed in the frozen
pre-registration) and renders examples/toy/ladder.png. No randomness here.

Requires matplotlib (not a core requirement — figure-only dependency):
    pip install matplotlib
Usage (from the toolkit root):
    python examples/toy/plot_ladder.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "placebo_verdict_results.json")
OUT = os.path.join(_HERE, "ladder.png")

# Two entities, fixed CVD-safe pair: noise floor = blue, real datapoint = orange.
C_FLOOR = "#3b6bd6"
C_REAL = "#e4572e"
C_TEXT = "#333333"
C_MUTED = "#666666"


def main():
    if not os.path.exists(RESULTS):
        sys.exit(f"ERROR: {RESULTS} not found. Run placebo_verdict.py --run first.")
    with open(RESULTS) as f:
        r = json.load(f)

    sizes = [10, 61, 1000]
    keys = ["10", "61", "1000"]
    mean = [r["floor_stats"][k]["media"] for k in keys]
    lo = [r["floor_stats"][k]["p2_5"] for k in keys]
    hi = [r["floor_stats"][k]["p97_5"] for k in keys]
    real61 = r["real"]["61"]

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.fill_between(sizes, lo, hi, color=C_FLOOR, alpha=0.14, linewidth=0,
                    label="noise floor, p2.5–p97.5 (20 GBM worlds, drift 0)")
    ax.plot(sizes, mean, color=C_FLOOR, linewidth=2, marker="o", markersize=5,
            label="noise floor, mean of best-of-N")
    ax.plot([61], [real61], marker="o", markersize=10, color=C_REAL,
            linestyle="none", zorder=5, label="BTC 2017–2026, best-of-61")

    # Selective direct labels (values in text ink, identity carried by the marks)
    for x, y in zip(sizes, mean):
        ax.annotate(f"+{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=C_TEXT)
    ax.annotate(f"BTC best-of-61: +{real61:.0f}%", (61, real61),
                textcoords="offset points", xytext=(14, -4), ha="left",
                fontsize=10, color=C_TEXT, fontweight="bold")
    ax.set_ylim(top=max(real61, max(hi)) * 1.18)
    ax.annotate("selection alone lifts the bar:\nthe best config on PURE NOISE",
                (1000, mean[2]), textcoords="offset points", xytext=(-8, -34),
                ha="right", fontsize=8.5, color=C_MUTED)

    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(["10", "61", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("search size N (configurations tried)", color=C_TEXT)
    ax.set_ylabel("net PnL of the selected best config (%)", color=C_TEXT)
    ax.set_title("The search-size ladder: best-of-N on pure noise vs BTC",
                 color=C_TEXT, fontsize=12)
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(colors=C_MUTED)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.92,
              edgecolor="none", facecolor="white")

    fig.tight_layout()
    fig.savefig(OUT, facecolor="white")
    print(f"[FIGURE] {OUT} written (deterministic: rendered from {os.path.basename(RESULTS)})")


if __name__ == "__main__":
    main()
