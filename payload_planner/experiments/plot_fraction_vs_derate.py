#!/usr/bin/env python3
"""
plot_fraction_vs_derate.py -- complementary envelope figure (Claim 3).

Fraction of the reachable workspace that must be slid (cannot be lifted) vs
torque derate, one curve per payload mass.  Falls to zero at the derate where the
payload becomes liftable everywhere.  Complements the utilisation figure: that
one shows HOW overloaded the workspace is; this one shows HOW MUCH of it needs
sliding.

Reads  experiments/envelope_cells.csv
Writes experiments/fig_fraction_vs_derate.pdf (and .png)
"""

import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV = HERE / "envelope_cells.csv"
COLORS = {"2p1": "#0072B2", "3p1": "#E69F00", "4p1": "#D55E00"}
LABEL = {"2p1": "2.1 kg", "3p1": "3.1 kg", "4p1": "4.1 kg"}
MARK = {"2p1": "o", "3p1": "s", "4p1": "^"}


def load():
    data = defaultdict(list)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if r["frac_must_slide"] == "":
                continue
            data[r["case"]].append((float(r["derate"]),
                                    100.0 * float(r["frac_must_slide"])))
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
        ys = [f for _, f in data[case]]
        ax.plot(xs, ys, marker=MARK[case], color=COLORS[case],
                label=LABEL[case], linewidth=1.8, markersize=5)
    ax.set_xlabel("torque derate $\\rho$")
    ax.set_ylabel("must-slide fraction of workspace  (\\%)")
    ax.set_xlim(0.24, 0.91)
    ax.set_ylim(-3, 103)
    ax.legend(title="payload mass", frameon=False, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / "fig_fraction_vs_derate.pdf")
    fig.savefig(HERE / "fig_fraction_vs_derate.png", dpi=150)
    print("wrote fig_fraction_vs_derate.{pdf,png}")


if __name__ == "__main__":
    main()