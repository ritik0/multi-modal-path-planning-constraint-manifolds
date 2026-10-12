"""
make_traj_summary.py -- turn an exported system_traj.csv into a tidy per-step
summary (time, phase, peak utilisation, ee height) for the Chapter 7 worked
trajectory figure.

The export (export_trajectory.py) writes system_traj.csv with columns
  t, q1..q7, dq1..dq7, ddq1..ddq7, tau1..tau7, [ee_x, ee_y, ee_z]
where tau1..tau7 are the feed-forward inverse-dynamics torques (which, on the
slide phases, already reflect contact load-sharing). This script computes, per
timestep:
  - umax   = max_j |tau_j| / (derate * tau_max_j)   -- peak utilisation
  - phase  = 0 while the ee is near table height (slide), 1 while lifted (carry)
and writes traj_exec_summary.csv (t, phase, umax, ee_z).

WHY: the previous summary CSV came from a soft run and sat near u=0.5, which
misleadingly suggests no sliding was needed. Run this on a TORQUE-STRESSED
export (heavy payload + tight derate) so u(t) rides near 1.0 on the slides and
dips at lift-off -- the load-sharing story over time.

Usage:
  python -B payload_planner/make_traj_summary.py \
      --exec-csv payload_planner/system_traj.csv \
      --derate 0.50 --table-height 0.30 --lift-eps 0.03
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

TAU_MAX = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec-csv", required=True,
                    help="path to system_traj.csv from export_trajectory.py")
    ap.add_argument("--derate", type=float, required=True,
                    help="the derate the trajectory was planned at")
    ap.add_argument("--table-height", type=float, default=None,
                    help="ee_z at the table (slide) height; default = min ee_z")
    ap.add_argument("--lift-eps", type=float, default=0.03,
                    help="ee_z above table+eps counts as lifted (phase 1)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    with open(args.exec_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames
    if not rows:
        print("empty exec csv"); return

    tau_cols = [c for c in cols if c.startswith("torque")]
    if len(tau_cols) < 7:
        print(f"expected torque1..torque7, found {tau_cols}"); return
    has_z = "ee_z" in cols

    t = np.array([float(r["time_s"]) for r in rows])
    tau = np.array([[float(r[c]) for c in tau_cols[:7]] for r in rows])
    budget = args.derate * TAU_MAX
    umax = np.max(np.abs(tau) / budget, axis=1)

    if has_z:
        ee_z = np.array([float(r["ee_z"]) for r in rows])
        table = args.table_height if args.table_height is not None else float(ee_z.min())
        phase = (ee_z > table + args.lift_eps).astype(int)
    else:
        ee_z = np.full_like(t, np.nan)
        phase = np.zeros_like(t, dtype=int)
        print("WARNING: no ee_z column; phase set to 0 throughout")

    out = args.out or os.path.join(os.path.dirname(args.exec_csv), "..",
                                   "data", "traj_exec_summary.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "phase", "umax", "ee_z"])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}", int(phase[i]),
                        f"{umax[i]:.4f}", f"{ee_z[i]:.4f}"])

    print(f"wrote {out}  ({len(t)} rows)")
    print(f"  peak utilisation over trajectory: {umax.max():.3f}")
    print(f"  slide-phase (0) peak u: {umax[phase==0].max():.3f}" if (phase==0).any() else "  no slide phase")
    print(f"  carry-phase (1) peak u: {umax[phase==1].max():.3f}" if (phase==1).any() else "  no carry phase")
    if umax.max() < 0.8:
        print("  NOTE: peak u < 0.8 -- this run was NOT torque-stressed; "
              "re-plan with a heavier payload / tighter derate for a "
              "figure that shows the load-sharing story.")


if __name__ == "__main__":
    main()