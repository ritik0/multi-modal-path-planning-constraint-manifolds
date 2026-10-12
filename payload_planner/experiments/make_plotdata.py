#!/usr/bin/env python3
"""
make_plotdata.py -- turn raw experiment CSVs into tidy, plot-ready CSVs that
pgfplots can read with simple \\addplot table calls (one column per payload
case, so no row-filtering by string is needed in LaTeX).

Reads (from the same folder):
  envelope_cells.csv, foliation_1d.csv, foliation_2d.csv,
  foliation_selected.csv, existence.csv

Writes (into ./data/ by default):
  plot_util_vs_derate.csv     derate, u_2p1, u_3p1, u_4p1
  plot_fraction_vs_derate.csv derate, f_2p1, f_3p1, f_4p1     (percent)
  plot_foliation_1d.csv       h, u_2p1, u_3p1, u_4p1
  plot_foliation_2d_4p1.csv   matrix-plot form: h, psi, u   (4p1 only)
  foliation_selected_clean.csv  case, method, h, psi, util_peak (best per case/method)
  (existence stays as-is; it becomes a booktabs table, not a plot)

Run from wherever the raw CSVs are:
    python3 make_plotdata.py            # writes ./data/*.csv
    python3 make_plotdata.py OUTDIR     # writes OUTDIR/*.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"
OUT.mkdir(parents=True, exist_ok=True)

CASES = ["2p1", "3p1", "4p1"]


def _read(name):
    with open(HERE / name) as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# 1. envelope: utilisation vs derate, and fraction vs derate (wide by case)
# --------------------------------------------------------------------------- #
def envelope():
    rows = _read("envelope_cells.csv")
    util = defaultdict(dict)     # derate -> case -> u
    frac = defaultdict(dict)
    for r in rows:
        if r["worst_lift_util"] == "":
            continue
        d = float(r["derate"])
        util[d][r["case"]] = float(r["worst_lift_util"])
        frac[d][r["case"]] = 100.0 * float(r["frac_must_slide"])

    with open(OUT / "plot_util_vs_derate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["derate"] + [f"u_{c}" for c in CASES])
        for d in sorted(util):
            w.writerow([f"{d:.2f}"] + [util[d].get(c, "nan") for c in CASES])

    with open(OUT / "plot_fraction_vs_derate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["derate"] + [f"f_{c}" for c in CASES])
        for d in sorted(frac):
            w.writerow([f"{d:.2f}"] + [frac[d].get(c, "nan") for c in CASES])


# --------------------------------------------------------------------------- #
# 2. foliation 1-D: utilisation vs height, wide by case (feasible points only)
# --------------------------------------------------------------------------- #
def foliation_1d():
    rows = _read("foliation_1d.csv")
    byh = defaultdict(dict)      # h -> case -> u
    for r in rows:
        if r["feasible"] != "1" or not r["util_peak"]:
            continue
        byh[float(r["h"])][r["case"]] = float(r["util_peak"])
    with open(OUT / "plot_foliation_1d.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["h"] + [f"u_{c}" for c in CASES])
        for h in sorted(byh):
            w.writerow([f"{h:.3f}"] + [byh[h].get(c, "nan") for c in CASES])


# --------------------------------------------------------------------------- #
# 3. foliation 2-D (4p1): matrix-plot form  h  psi  u  (blank line between h)
# --------------------------------------------------------------------------- #
def foliation_2d(case="4p1"):
    rows = [r for r in _read("foliation_2d.csv") if r["case"] == case]
    hs = sorted(set(float(r["h"]) for r in rows))
    ps = sorted(set(float(r["psi"]) for r in rows))
    val = {}
    for r in rows:
        if r["feasible"] == "1" and r["util_peak"]:
            val[(float(r["h"]), float(r["psi"]))] = float(r["util_peak"])
    # pgfplots matrix plot wants a full grid; use nan for infeasible
    path = OUT / f"plot_foliation_2d_{case}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["h", "psi", "u"])
        for h in hs:
            for p in ps:
                u = val.get((h, p), "nan")
                w.writerow([f"{h:.3f}", f"{p:.3f}", u])
            f.write("\n")   # blank line separates scanlines for matrix/mesh
    # also record grid dims for the LaTeX (mesh/rows cols)
    return len(hs), len(ps)


# --------------------------------------------------------------------------- #
# 4. selections: best (lowest-util) per (case, method)
# --------------------------------------------------------------------------- #
def selections():
    rows = _read("foliation_selected.csv")
    best = {}
    for r in rows:
        if not r["util_peak"]:
            continue
        key = (r["case"], r["method"])
        u = float(r["util_peak"])
        if key not in best or u < best[key][0]:
            best[key] = (u, r)
    with open(OUT / "foliation_selected_clean.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "method", "h", "psi", "util_peak"])
        for case in CASES:
            for method in ("1d_continuous", "2d_coorddesc"):
                if (case, method) in best:
                    _, r = best[(case, method)]
                    w.writerow([case, method, r["h"], r.get("psi", ""),
                                f"{float(r['util_peak']):.4f}"])
    return best


def main():
    envelope()
    foliation_1d()
    nh, npsi = foliation_2d("4p1")
    best = selections()
    print(f"wrote plot data to {OUT}/")
    print(f"  2-D grid for 4p1: {nh} heights x {npsi} yaws")
    print("  selections (best per case/method):")
    for case in CASES:
        for method in ("1d_continuous", "2d_coorddesc"):
            if (case, method) in best:
                _, r = best[(case, method)]
                psi = f", psi={r['psi']}" if r.get("psi") else ""
                print(f"    {case} {method}: h={r['h']}{psi}, u={float(r['util_peak']):.3f}")


if __name__ == "__main__":
    main()