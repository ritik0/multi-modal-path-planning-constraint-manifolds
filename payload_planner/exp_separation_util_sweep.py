"""
exp_separation_util_sweep.py -- maps the SLIDE-LIFT-SLIDE peak-utilisation
surface over (separation x derate), on the single continuous plane with a wall.

WHY THIS, NOT THE MONO-VS-SLS VERSION
-------------------------------------
The earlier sweep also tried a MONOLITHIC direct lift as a baseline, but that
turned out to be a planning-success artifact (a single-shot RRT-Connect either
found a direct path or didn't, non-monotonically), not a clean feasibility
boundary -- so it is dropped here. What IS clean and physical is the peak torque
UTILISATION of the slide-lift-slide transport as a function of separation and
derate:

  - utilisation rises with separation (reaching farther over the wall costs
    more torque),
  - utilisation falls with derate (more budget = more headroom),
  - and in the far / low-derate corner it exceeds u=1, yet the transport still
    succeeds -- which is only possible because sliding offloads the excess. That
    crossing of u=1 is the whole point: it is where direct lifting is infeasible
    and sliding is required.

This script computes exactly that surface (successful transports only), with no
monolithic baseline, so it runs in roughly half the time of the mono-vs-sls
version.

Emits data/exp_separation_util_sweep_<case>.csv (auto-named per payload).

    for c in 2p1 3p1 4p1; do
      PAYLOAD_CASE=$c PAYLOAD_WALL=1 \
      python -B payload_planner/exp_separation_util_sweep.py \
          --n-sep 10 --sep-max 0.90 \
          --derate-min 0.40 --derate-max 0.80 --derate-step 0.10 \
          --solve-time 20 --space atlas
    done

(2p1 is already done from the earlier run; you can reuse that CSV or re-run for
consistency. 3p1 and 4p1 are the ones worth running with this faster script.)
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import time

import numpy as np

from scene import Scene
import transport
from run_panda import CFG, TABLE_A, TABLE_B
import payload_model

HERE = os.path.dirname(os.path.abspath(__file__))
START_SURF = "plane_start"
GOAL_SURF = "plane_goal"


def _plane_cfg(derate, x_lo, x_hi, y_reach, height, wall_half):
    """One continuous plane with a wall, modelled as two co-planar surfaces at
    the same height either side of the wall (start side +y, goal side -y)."""
    cfg = copy.deepcopy(CFG)
    cfg["torque_derate"] = float(derate)
    cfg["surfaces"] = [
        {"name": START_SURF, "height": float(height),
         "xy_bounds": [[float(x_lo), float(wall_half)],
                       [float(x_hi), float(y_reach)]]},
        {"name": GOAL_SURF, "height": float(height),
         "xy_bounds": [[float(x_lo), float(-y_reach)],
                       [float(x_hi), float(-wall_half)]]},
    ]
    return cfg


def _peak_utilisation(scene, path):
    if path is None or len(path) == 0:
        return float("nan")
    tau_max = np.asarray(scene.tau_max, float) * float(scene.derate)
    return float(max(
        np.max(np.abs(scene.robot.gravity(q)) / tau_max)
        for q in np.asarray(path, float)
    ))


def _path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    p = np.asarray(path, float)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _reachable_y_extent(scene, surf, x, rng, y_min=0.0, y_max=0.9, step=0.02):
    reach = y_min
    y = y_min
    while y <= y_max:
        q, ok = transport.place(scene, surf, x, y, rng=rng, n_restarts=16)
        if not ok or q is None:
            break
        reach = y
        y += step
    return reach


def _slide_lift_slide(scene, start, goal, solve_time, delta, space):
    t0 = time.perf_counter()
    res = transport.plan(
        scene, start, goal,
        planner="RRTConnect", solve_time=solve_time, delta=delta,
        constrained_space=space)
    elapsed = time.perf_counter() - t0
    path = res.full_path() if res.success else None
    return (res.success, elapsed, _peak_utilisation(scene, path),
            _path_length(path), res.n_lift, res.n_set, res.pairs_tried,
            len(res.segments) if res.success else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sep", type=int, default=10)
    ap.add_argument("--sep-min", type=float, default=0.10)
    ap.add_argument("--sep-max", type=float, default=0.90)
    ap.add_argument("--derate-min", type=float, default=0.40)
    ap.add_argument("--derate-max", type=float, default=0.80)
    ap.add_argument("--derate-step", type=float, default=0.10)
    ap.add_argument("--reach-x", type=float, default=None)
    ap.add_argument("--solve-time", type=float, default=20.0)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--space", type=str, default="atlas",
                    choices=["atlas", "projected"])
    args = ap.parse_args()

    case = payload_model.normalise_case()
    out_csv = os.path.join(HERE, "..", "data",
                           f"exp_separation_util_sweep_{case}.csv")

    ax = np.asarray(TABLE_A["xy_bounds"], float)
    bx = np.asarray(TABLE_B["xy_bounds"], float)
    x_lo = max(ax[0, 0], bx[0, 0])
    x_hi = min(ax[1, 0], bx[1, 0])
    reach_x = args.reach_x if args.reach_x is not None else 0.5 * (x_lo + x_hi)
    height = float(TABLE_A["height"])

    from run_panda import GAP_WALL
    wall_half = 0.5 * float(GAP_WALL["ysize"])
    sep_floor = 2.0 * (wall_half + 0.02)

    n_der = int(round((args.derate_max - args.derate_min) / args.derate_step)) + 1
    derates = np.round(args.derate_min + args.derate_step * np.arange(n_der), 3)
    rng = np.random.default_rng(20260711)

    y_reach_guess = 0.5 * args.sep_max + 0.05
    probe_scene = Scene(_plane_cfg(
        derates.min(), x_lo, x_hi, y_reach_guess, height, wall_half))
    probe_surf = probe_scene.surface(START_SURF)
    reach_y = _reachable_y_extent(
        probe_scene, probe_surf, reach_x, rng,
        y_min=wall_half + 0.02, y_max=y_reach_guess)
    sep_max = min(args.sep_max, 2.0 * reach_y)
    sep_min = max(args.sep_min, sep_floor)
    if sep_max <= sep_min:
        print(f"[{case}] reachable |y|={reach_y:.3f} leaves no room; "
              f"try a different --reach-x")
        return
    print(f"[{case}] reach_x={reach_x:.3f}  reachable |y|={reach_y:.3f}  "
          f"wall_half={wall_half:.3f}  => separation swept "
          f"{sep_min:.3f}..{sep_max:.3f} m  (SLS only, no mono baseline)")

    seps = np.linspace(sep_min, sep_max, args.n_sep)
    rows = []

    for rho in derates:
        scene = Scene(_plane_cfg(rho, x_lo, x_hi, reach_y, height, wall_half))
        print(f"\n=== derate {rho:0.2f} ===")
        for sep in seps:
            start = transport.Placement(START_SURF, reach_x, +0.5 * sep)
            goal = transport.Placement(GOAL_SURF, reach_x, -0.5 * sep)
            (ok, t, u, plen, n_lift, n_set, pairs, n_seg) = _slide_lift_slide(
                scene, start, goal, args.solve_time, args.delta, args.space)
            rows.append({
                "case": case, "derate": round(float(rho), 3),
                "separation": round(float(sep), 4),
                "reach_x": round(float(reach_x), 4),
                "sls_success": int(bool(ok)),
                "sls_time_s": round(float(t), 3),
                "sls_peak_u": round(float(u), 4) if u == u else "",
                "sls_path_len": round(float(plen), 4),
                "sls_n_lift": n_lift, "sls_n_set": n_set,
                "sls_pairs_tried": pairs, "sls_segments": n_seg,
            })
            print(f"  sep={sep:0.3f}  sls={int(bool(ok))} "
                  f"({t:5.1f}s, u={u:0.3f}, {n_seg} segs)")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

    ok_rows = [r for r in rows if r["sls_success"] == 1 and r["sls_peak_u"] != ""]
    print(f"\n{len(ok_rows)}/{len(rows)} cells solved.")
    if ok_rows:
        print("peak utilisation crosses u=1 (direct lift infeasible, slide "
              "required) in these cells:")
        any_over = False
        for r in ok_rows:
            if r["sls_peak_u"] > 1.0:
                any_over = True
                print(f"  derate {r['derate']}, sep {r['separation']}: "
                      f"u={r['sls_peak_u']}")
        if not any_over:
            print("  (none in this range -- all solved carries stayed under u=1)")


if __name__ == "__main__":
    main()