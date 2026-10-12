#!/usr/bin/env python3
"""
make_freehold_figs.py  --  turn the raw partition maps into pgfplots-ready tables.

Uses only csv + numpy (no pandas), matching the rest of the repo.

Reads   ../data/exp_freehold_map_{2p1,3p1,4p1}.csv   (from exp_freehold_map.py)
Writes  the tables the thesis figures read directly, all plain CSV so the
        plotting happens in pgfplots, NOT in matplotlib:

  1) data/fig_freehold_pts_<case>.csv    (x, y, u)  reachable cells only
        -> the partition-map field (one coloured square per cell).
  2) data/fig_freehold_reach.csv         (reach, u2, u3, u4)
        -> reach-collapse curve (lower envelope per reach bin).
  3) data/fig_freehold_threshold.csv     (derate, r2, r3, r4)
        -> r*(rho), the free-hold reach limit per payload.

No PNG is produced; fig_freehold.tex / tab_freehold.tex render everything.
"""

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CASES = [("2p1", "u2"), ("3p1", "u3"), ("4p1", "u4")]


def load(case):
    """Read a raw map CSV into a dict of numpy arrays. Blank u -> nan."""
    path = os.path.join(DATA, f"exp_freehold_map_{case}.csv")
    x, y, reach, u = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            x.append(float(row["x"]))
            y.append(float(row["y"]))
            reach.append(float(row["reach"]))
            v = row["u_star_raw"]
            u.append(float(v) if v not in ("", "nan") else np.nan)
    return {"x": np.array(x), "y": np.array(y),
            "reach": np.array(reach), "u": np.array(u)}


def write_points(case):
    """Scatter file (kept for reference) + complete-lattice matrix .dat for the
    pgfplots matrix plot.  Returns the number of distinct x values (mesh/cols)."""
    d = load(case)
    ok = np.isfinite(d["u"])
    # scatter file: reachable cells only (handy for quick checks / other tools)
    out_pts = os.path.join(DATA, f"fig_freehold_pts_{case}.csv")
    with open(out_pts, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "u"])
        for xi, yi, ui in zip(d["x"][ok], d["y"][ok], d["u"][ok]):
            w.writerow([f"{xi:.5f}", f"{yi:.5f}", f"{ui:.5f}"])

    # complete-lattice matrix .dat: every (x,y) node, sentinel 9.99 for holes,
    # blank line between rows of constant y.  This is what fig_freehold.tex reads
    # via 'matrix plot' -- memory-safe for dense grids, unlike a scatter of marks.
    # A full 121x121 lattice exceeds pgfplots' default TeX memory, so we thin the
    # lattice by STRIDE (every STRIDE-th node) for the figure.  The underlying
    # analysis (reach curve, thresholds) always uses the full-resolution data;
    # only the rendered field is thinned, which is invisible in print.
    xs = np.unique(np.round(d["x"], 5))
    ys = np.unique(np.round(d["y"], 5))
    stride = max(1, int(np.ceil(len(xs) / 55.0)))   # aim for <=55 cols (TeX memory)
    xs_r = xs[::stride]
    ys_r = ys[::stride]
    lut = {(round(xi, 5), round(yi, 5)): ui
           for xi, yi, ui in zip(d["x"], d["y"], d["u"])}
    out_dat = os.path.join(DATA, f"fig_freehold_map_{case}.dat")
    with open(out_dat, "w") as f:
        f.write("x y u\n")
        for yv in ys_r:
            for xv in xs_r:
                u = lut.get((round(xv, 5), round(yv, 5)), np.nan)
                u = 9.99 if (isinstance(u, float) and np.isnan(u)) else u
                f.write(f"{xv:.5f} {yv:.5f} {u:.5f}\n")
            f.write("\n")
    print(f"  wrote {out_pts}  ({int(ok.sum())} reachable pts)")
    print(f"  wrote {out_dat}  (matrix {len(xs_r)}x{len(ys_r)}, stride {stride};  "
          f"set mesh/cols={len(xs_r)} in fig_freehold.tex)")
    return len(xs_r)


def reach_curve(nbins=22):
    """Lower envelope of u_star_raw per reach bin, per payload.

    Quantile-spaced bins over the populated reach range (no empty bins), then
    forward/backward filled: a missing bin is a sampling gap, not a physical dip.
    """
    cols = {}
    ref_centres = None
    for case, col in CASES:
        d = load(case)
        ok = np.isfinite(d["u"])
        r, u = d["reach"][ok], d["u"][ok]
        edges = np.unique(np.quantile(r, np.linspace(0, 1, nbins + 1)))
        centres = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(np.digitize(r, edges) - 1, 0, len(centres) - 1)
        vals = np.array([u[idx == b].min() if np.any(idx == b) else np.nan
                         for b in range(len(centres))])
        for i in range(1, len(vals)):
            if np.isnan(vals[i]):
                vals[i] = vals[i - 1]
        for i in range(len(vals) - 2, -1, -1):
            if np.isnan(vals[i]):
                vals[i] = vals[i + 1]
        cols[col] = vals
        if ref_centres is None or len(centres) < len(ref_centres):
            ref_centres = centres
    n = len(ref_centres)
    for col in cols:
        if len(cols[col]) != n:
            old = np.linspace(0, 1, len(cols[col]))
            new = np.linspace(0, 1, n)
            cols[col] = np.interp(new, old, cols[col])

    out = os.path.join(DATA, "fig_freehold_reach.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reach", "u2", "u3", "u4"])
        for i in range(n):
            w.writerow([f"{ref_centres[i]:.4f}",
                        f"{cols['u2'][i]:.4f}",
                        f"{cols['u3'][i]:.4f}",
                        f"{cols['u4'][i]:.4f}"])
    print(f"  wrote {out}")
    return ref_centres, cols


def thresholds(centres, cols):
    """r*(rho): first reach where the lower envelope exceeds rho."""
    rhos = [0.40, 0.50, 0.60, 0.70, 0.80]
    out = os.path.join(DATA, "fig_freehold_threshold.csv")
    table = []
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["derate", "r2", "r3", "r4"])
        for rho in rhos:
            cells = [f"{rho:.2f}"]
            disp = [rho]
            for col in ("u2", "u3", "u4"):
                u = cols[col]
                m = np.isfinite(u) & (u > rho)
                if m.any():
                    r = float(centres[np.argmax(m)])
                    cells.append(f"{r:.4f}")
                    disp.append(round(r, 4))
                else:
                    cells.append("")
                    disp.append(None)
            w.writerow(cells)
            table.append(disp)
    print(f"  wrote {out}")
    print("\n  r*(rho) - free-hold reach limit (m); blank = never must-slide in range:")
    print("   rho     r2      r3      r4")
    for row in table:
        parts = []
        for j, v in enumerate(row):
            if v is None:
                parts.append("  --  ")
            elif j == 0:
                parts.append(f"{v:.2f}")
            else:
                parts.append(f"{v:.4f}")
        print("  " + "  ".join(parts))


def main():
    print("=" * 60)
    cols = {}
    for case, _ in CASES:
        cols[case] = write_points(case)
    centres, curves = reach_curve()
    thresholds(centres, curves)
    print("=" * 60)
    print("  REMINDER: fig_freehold.tex uses mesh/cols for the 4.1 kg map.")
    print(f"  Set mesh/cols={cols['4p1']} (and match per-case if you plot the others).")
    print("=" * 60)


if __name__ == "__main__":
    main()