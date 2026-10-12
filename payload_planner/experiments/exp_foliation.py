#!/usr/bin/env python3
"""
exp_foliation.py -- transfer-foliation sweeps (Claims 4 and 5).

Generates the raw data behind the two foliation figures:
  * Claim 4 (1-D): peak torque utilisation u(h) as the carry height h is swept
    across the feasible interval, for a fixed transport task.  The planner's
    selected h* is the minimiser.
  * Claim 5 (2-D): the utilisation landscape u(h, psi) over carry height and
    carry yaw, from which the (h*, psi*) optimum and the relative importance of
    the two dimensions are read.

Unlike the golden-section SELECTION (which visits only a few points), this sweep
evaluates a DENSE grid so the figures show the full u(h) curve and u(h,psi)
surface -- we want to SEE the landscape, not just the chosen point.  The selected
optimum is recorded separately so it can be marked on the plot.

Outputs (into experiments/):
  foliation_1d.csv    columns: case, derate, h, feasible, util_peak, util_mean, path_len
  foliation_2d.csv    columns: case, derate, h, psi, feasible, util_peak, util_mean, path_len
  foliation_selected.csv  the golden-section / coordinate-descent optima

Run from payload_planner/:
    PAYLOAD_CASE=4p1 PAYLOAD_TORQUE_DERATE=0.60 PAYLOAD_WALL=1 PAYLOAD_WALL_TOP=0.15 \
    python3 experiments/exp_foliation.py
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
OUT = HERE
OUT.mkdir(exist_ok=True)

import run_panda                      # noqa: E402
import transport as T                 # noqa: E402
import transfer_foliation as fol      # noqa: E402
from scene import Scene               # noqa: E402


# --------------------------------------------------------------------------- #
# Dense sweep grids (edit for resolution).
# --------------------------------------------------------------------------- #
H_MIN = float(os.environ.get("EXP_H_MIN", "0.20"))
H_MAX = float(os.environ.get("EXP_H_MAX", "0.60"))
H_STEPS = int(os.environ.get("EXP_H_STEPS", "17"))          # ~2.5 cm spacing

PSI_MIN = float(os.environ.get("EXP_PSI_MIN", "-0.5"))
PSI_MAX = float(os.environ.get("EXP_PSI_MAX", "0.5"))
PSI_STEPS = int(os.environ.get("EXP_PSI_STEPS", "11"))      # ~0.1 rad spacing

SOLVE_TIME = float(os.environ.get("EXP_SOLVE_TIME", "12"))
DELTA = float(os.environ.get("PAYLOAD_DELTA", "0.08"))
SPACE = os.environ.get("PAYLOAD_CONSTRAINED_SPACE", "atlas")


def _pick_pair(scene):
    """Pick one representative lift-off / set-down pair, deterministically."""
    sa, sb = scene.surface("A"), scene.surface("B")
    rng = np.random.default_rng(20260612)
    lifts = T.sample_transitions(scene, sa, 200, other_surface=sb, rng=rng,
                                 support_margin=0.05)
    sets = T.sample_transitions(scene, sb, 200, other_surface=sa, rng=rng,
                                support_margin=0.05)
    if not lifts or not sets:
        raise SystemExit(
            "no liftable transitions at this case/derate; "
            "raise PAYLOAD_TORQUE_DERATE or lighten PAYLOAD_CASE"
        )
    q_lift = sorted(lifts, key=lambda q: float(np.linalg.norm(q)))[0]
    q_set = sorted(sets, key=lambda q: float(np.linalg.norm(q)))[0]
    return q_lift, q_set


def sweep_1d(scene, q_lift, q_set, case, derate):
    rows = []
    for h in np.linspace(H_MIN, H_MAX, H_STEPS):
        ok, score, info = fol.evaluate_leaf(
            scene, q_lift, q_set, float(h),
            solve_time=SOLVE_TIME, delta=DELTA, space=SPACE,
        )
        rows.append({
            "case": case, "derate": derate, "h": round(float(h), 4),
            "feasible": int(bool(ok)),
            "util_peak": info.get("util_peak", ""),
            "util_mean": info.get("util_mean", ""),
            "path_len": info.get("path_len", ""),
            "reason": info.get("reason", ""),
        })
        tag = "OK  " if ok else "FAIL"
        print(f"  1D h={h:.3f}: {tag} util_peak={info.get('util_peak',0):.3f}", flush=True)
    return rows


def sweep_2d(scene, q_lift, q_set, case, derate):
    rows = []
    for h in np.linspace(H_MIN, H_MAX, H_STEPS):
        for psi in np.linspace(PSI_MIN, PSI_MAX, PSI_STEPS):
            ok, score, info = fol.evaluate_leaf_2d(
                scene, q_lift, q_set, float(h), float(psi),
                solve_time=SOLVE_TIME, delta=DELTA, space=SPACE,
            )
            rows.append({
                "case": case, "derate": derate,
                "h": round(float(h), 4), "psi": round(float(psi), 4),
                "feasible": int(bool(ok)),
                "util_peak": info.get("util_peak", ""),
                "util_mean": info.get("util_mean", ""),
                "path_len": info.get("path_len", ""),
            })
        print(f"  2D h={h:.3f}: swept {PSI_STEPS} yaws", flush=True)
    return rows


def selected_optima(scene, q_lift, q_set, case, derate):
    """Record what the planner's own selection picks (for marking on plots)."""
    rows = []
    # 1-D continuous selection
    h1, info1 = fol._select_continuous(
        scene, q_lift, q_set, "RRTConnect", SOLVE_TIME, DELTA, SPACE,
        10.0, 1.0, False)
    if h1 is not None:
        rows.append({"case": case, "derate": derate, "method": "1d_continuous",
                     "h": round(float(h1), 4), "psi": "",
                     "util_peak": info1.get("util_peak", "")})
    # 2-D coordinate descent
    h2, psi2, info2 = fol.select_transfer_2d(
        scene, q_lift, q_set, solve_time=SOLVE_TIME, delta=DELTA, space=SPACE,
        verbose=False)
    if h2 is not None:
        rows.append({"case": case, "derate": derate, "method": "2d_coorddesc",
                     "h": round(float(h2), 4), "psi": round(float(psi2), 4),
                     "util_peak": info2.get("util_peak", "")})
    return rows


def _append_csv(path, rows, keys):
    if not rows:
        return
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main():
    case = run_panda.PAYLOAD_CASE
    derate = run_panda.PAYLOAD_TORQUE_DERATE
    scene = Scene(run_panda.CFG)
    print(f"[foliation] case={case} derate={derate:.2f} "
          f"h in [{H_MIN},{H_MAX}]x{H_STEPS}, psi in [{PSI_MIN},{PSI_MAX}]x{PSI_STEPS}")
    q_lift, q_set = _pick_pair(scene)

    r1 = sweep_1d(scene, q_lift, q_set, case, derate)
    _append_csv(OUT / "foliation_1d.csv", r1,
                ["case", "derate", "h", "feasible", "util_peak", "util_mean",
                 "path_len", "reason"])

    r2 = sweep_2d(scene, q_lift, q_set, case, derate)
    _append_csv(OUT / "foliation_2d.csv", r2,
                ["case", "derate", "h", "psi", "feasible", "util_peak",
                 "util_mean", "path_len"])

    rs = selected_optima(scene, q_lift, q_set, case, derate)
    _append_csv(OUT / "foliation_selected.csv", rs,
                ["case", "derate", "method", "h", "psi", "util_peak"])

    print(f"\nappended {len(r1)} 1-D rows, {len(r2)} 2-D rows, {len(rs)} optima")
    print(f"  -> {OUT/'foliation_1d.csv'}")
    print(f"  -> {OUT/'foliation_2d.csv'}")
    print(f"  -> {OUT/'foliation_selected.csv'}")


if __name__ == "__main__":
    main()