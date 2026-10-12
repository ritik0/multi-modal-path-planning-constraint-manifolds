#!/usr/bin/env python3
"""
plot_foliation_1d.py -- 1-D transfer-foliation curve (Claim 4).

Peak torque utilisation of the carry vs transfer height, for a fixed task.  The
infeasible region below the obstacle is shaded; the height selected by the
planner is marked.  Shows that raising the carry reduces peak torque utilisation
up to a plateau, and that the planner selects a height on that plateau.

Reads  experiments/foliation_1d.csv, experiments/foliation_selected.csv
Writes experiments/fig_foliation_1d.pdf (and .png)
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def load_curve(case="4p1", derate="0.6"):
    hs, us, feas = [], [], []
    with open(HERE / "foliation_1d.csv") as f:
        for r in csv.DictReader(f):
            if r["case"] != case:
                continue
            hs.append(float(r["h"]))
            us.append(float(r["util_peak"]) if r["util_peak"] else None)
            feas.append(r["feasible"] == "1")
    pts = sorted(zip(hs, us, feas))
    return pts


def load_selected(case="4p1"):
    """Return the 1-D continuous selection with the LOWEST util (latest good run)."""
    best = None
    with open(HERE / "foliation_selected.csv") as f:
        for r in csv.DictReader(f):
            if r["case"] != case or r["method"] != "1d_continuous":
                continue
            if not r["util_peak"]:
                continue
            u = float(r["util_peak"])
            if best is None or u < best[1]:
                best = (float(r["h"]), u)
    return best


def main():
    pts = load_curve()
    hs = [h for h, u, fe in pts if u is not None and fe]
    us = [u for h, u, fe in pts if u is not None and fe]
    infeas_h = [h for h, u, fe in pts if not fe or u is None]

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    # shade infeasible-below-obstacle region if any low heights failed
    if infeas_h:
        hcut = max(infeas_h)
        ax.axvspan(min(hs + infeas_h) - 0.02, hcut, color="0.92", zorder=0)
        ax.text(hcut, max(us), " below obstacle:\n infeasible", fontsize=9.5,
                color="0.45", ha="left", va="top")

    ax.plot(hs, us, marker="o", color="#0072B2", linewidth=1.9, markersize=5,
            label="carry utilisation $u(h)$")

    sel = load_selected()
    if sel:
        ax.axvline(sel[0], color="#D55E00", linestyle=":", linewidth=1.4)
        ax.plot([sel[0]], [sel[1]], marker="*", color="#D55E00", markersize=16,
                linestyle="none", label=f"selected $h^\\star={sel[0]:.2f}$")

    ax.set_xlabel("transfer height  $h$  (m)")
    ax.set_ylabel("peak torque utilisation  $u(h)$")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / "fig_foliation_1d.pdf")
    fig.savefig(HERE / "fig_foliation_1d.png", dpi=150)
    print("wrote fig_foliation_1d.{pdf,png}")


if __name__ == "__main__":
    main()