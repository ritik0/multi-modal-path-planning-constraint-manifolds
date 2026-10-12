#!/usr/bin/env python3
"""
plot_util_vs_derate.py -- headline envelope figure (Claim 3).

Worst-case lift utilisation over the workspace vs torque derate, one curve per
payload mass.  The horizontal line u = 1 is the direct-fly boundary: above it the
worst pose cannot be lifted (a must-slide region exists); below it the whole
workspace is liftable.  Each curve's crossing of u = 1 marks the derate at which
that payload becomes liftable everywhere; a curve that never crosses always needs
sliding.

Reads  experiments/envelope_cells.csv
Writes experiments/fig_util_vs_derate.pdf (and .png)
"""

import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV = HERE / "envelope_cells.csv"

# colourblind-safe, print-friendly
COLORS = {"2p1": "#0072B2", "3p1": "#E69F00", "4p1": "#D55E00"}
LABEL = {"2p1": "2.1 kg", "3p1": "3.1 kg", "4p1": "4.1 kg"}
MARK = {"2p1": "o", "3p1": "s", "4p1": "^"}


def load():
    data = defaultdict(list)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["worst_lift_util"] == "":
                continue
            data[r["case"]].append((float(r["derate"]),
                                    float(r["worst_lift_util"])))
    for c in data:
        data[c].sort()
    return data


def main():
    data = load()
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for case in ("2p1", "3p1", "4p1"):
        if case not in data:
            continue
        xs = [d for d, _ in data[case]]
        ys = [u for _, u in data[case]]
        ax.plot(xs, ys, marker=MARK[case], color=COLORS[case],
                label=LABEL[case], linewidth=1.8, markersize=5)

    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)

    # shade the must-slide (infeasible-to-lift) region lightly
    ax.axhspan(1.0, 4.1, color="0.92", zorder=0)
    ax.text(0.40, 3.55, "must-slide region exists", fontsize=10.5,
            color="0.40", ha="center", va="center", style="italic")
    ax.text(0.70, 0.83, "liftable everywhere", fontsize=10.5,
            color="0.40", ha="center", va="center", style="italic")
    ax.text(0.263, 1.045, "direct-fly boundary  ($u=1$)", fontsize=9.5,
            color="0.35", va="bottom")

    ax.set_xlabel("torque derate $\\rho$")
    ax.set_ylabel("worst-case lift utilisation  $u_{\\max}$")
    ax.set_xlim(0.24, 0.91)
    ax.set_ylim(0.78, 3.95)
    ax.legend(title="payload mass", frameon=False, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_pdf = HERE / "fig_util_vs_derate.pdf"
    out_png = HERE / "fig_util_vs_derate.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    print("wrote", out_pdf)
    print("wrote", out_png)


if __name__ == "__main__":
    main()