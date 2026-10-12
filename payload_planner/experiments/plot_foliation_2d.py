#!/usr/bin/env python3
"""
plot_foliation_2d.py -- 2-D transfer-foliation heatmap (Claim 5).

Peak torque utilisation over the two leaf coordinates: transfer height h and
carry yaw psi.  Infeasible cells are left blank.  The planner's selected
(h*, psi*) is marked.  Shows that height is the dominant lever and yaw a
secondary trim, with the joint optimum selected.

Reads  experiments/foliation_2d.csv, experiments/foliation_selected.csv
Writes experiments/fig_foliation_2d.pdf (and .png)
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def main():
    rows = [r for r in csv.DictReader(open(HERE / "foliation_2d.csv"))
            if r["case"] == "4p1"]
    hs = sorted(set(float(r["h"]) for r in rows))
    ps = sorted(set(float(r["psi"]) for r in rows))
    H = {h: i for i, h in enumerate(hs)}
    P = {p: i for i, p in enumerate(ps)}
    grid = np.full((len(ps), len(hs)), np.nan)
    for r in rows:
        if r["feasible"] == "1" and r["util_peak"]:
            grid[P[float(r["psi"])], H[float(r["h"])]] = float(r["util_peak"])

    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r",
                   extent=[min(hs), max(hs), min(ps), max(ps)],
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("peak torque utilisation  $u(h,\\psi)$")

    # mark selected (h*, psi*): the 2d coord-descent row with lowest util
    best = None
    for r in csv.DictReader(open(HERE / "foliation_selected.csv")):
        if r["case"] == "4p1" and r["method"] == "2d_coorddesc" and r["util_peak"]:
            u = float(r["util_peak"])
            if best is None or u < best[2]:
                best = (float(r["h"]), float(r["psi"]), u)
    if best:
        ax.plot([best[0]], [best[1]], marker="*", color="#D55E00",
                markersize=18, markeredgecolor="white", markeredgewidth=0.8,
                linestyle="none",
                label=f"selected $(h^\\star,\\psi^\\star)=({best[0]:.2f},{best[1]:.2f})$")
        ax.legend(frameon=True, loc="upper right", fontsize=10)

    ax.set_xlabel("transfer height  $h$  (m)")
    ax.set_ylabel("carry yaw  $\\psi$  (rad)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_foliation_2d.pdf")
    fig.savefig(HERE / "fig_foliation_2d.png", dpi=150)
    print("wrote fig_foliation_2d.{pdf,png}")


if __name__ == "__main__":
    main()