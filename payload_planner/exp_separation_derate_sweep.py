"""
exp_separation_derate_sweep.py -- 2-D headline experiment: separation x derate,
on a SINGLE continuous plane with a wall in the middle (the real-robot scene).

WHY A SINGLE PLANE, NOT TWO TABLES
----------------------------------
The two-table CFG in run_panda.py bounds start/goal to two narrow y-patches
either side of the wall gap. The real-robot experiment used ONE continuous table
with the wall crossing it, so start/goal can sit anywhere the ARM can reach, not
only on a predefined patch. This script rebuilds the scene as one wide plane and
lets the arm's own reach define how far apart start and goal can be placed --
exactly "stretch out as far as the arm can hold the payload flat, then come
inner and inner".

WHAT IT SWEEPS
--------------
  for each derate rho (default 0.30..0.80 step 0.10):     # sets where the
                                                          # overloaded zone is
    for each separation s (far -> near) across the wall:  # along y, symmetric
      - MONOLITHIC direct lift start->goal (no sliding): feasible?
      - SLIDE-LIFT-SLIDE (this thesis's planner): feasible?
      - times, peak torque, path length for each.

REACH IS AUTO-DISCOVERED
------------------------
At startup, at the chosen reach-x and table height, the script probes outward in
+y (flat-carry projection) until it can no longer place the payload, and caps the
largest separation at twice that reachable extent. So --sep-max is an upper
request; the actual max used is min(request, 2 * reachable_y).

The wall geometry comes from the scene (PAYLOAD_WALL / PAYLOAD_WALL_TOP). Set
PAYLOAD_CASE and PAYLOAD_WALL in the environment; derate is controlled by this
script (do NOT set PAYLOAD_TORQUE_DERATE).

Emits data/exp_separation_derate_sweep_<case>.csv (auto-named per payload).

    for c in 2p1 3p1 4p1; do
      PAYLOAD_CASE=$c PAYLOAD_WALL=1 \
      python -B payload_planner/exp_separation_derate_sweep.py \
          --n-sep 12 --sep-max 0.90 \
          --derate-min 0.30 --derate-max 0.80 --derate-step 0.10 \
          --solve-time 30 --space atlas
    done
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
    """
    A copy of CFG representing ONE continuous plane with a wall in the middle,
    modelled as TWO co-planar surfaces at the same height: a start side
    (y in [+wall_half, +y_reach]) and a goal side (y in [-y_reach, -wall_half]).

    This is physically one flat table -- both surfaces share the same height and
    abut the wall from opposite sides -- but giving the planner two named
    surfaces provides the start-side / goal-side structure transport.plan()
    expects (lift-off sampled on the start side, set-down on the goal side),
    which a single surface spanning the whole plane does not.
    """
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
    """
    Probe outward in +y from y_min (just clear of the wall) at fixed reach x
    until the payload can no longer be placed flat. Returns the largest y
    reachable. place() does multi-seed projection onto the on-surface manifold,
    so a success means the arm can hold the payload flat there.
    """
    reach = y_min
    y = y_min
    while y <= y_max:
        q, ok = transport.place(scene, surf, x, y, rng=rng, n_restarts=16)
        if not ok or q is None:
            break
        reach = y
        y += step
    return reach


def _monolithic_lift(scene, surf_start, surf_goal, start, goal,
                     solve_time, delta, space, rng):
    t0 = time.perf_counter()
    lo_s = transport.liftoff_mode(scene, surf_start)
    lo_g = transport.liftoff_mode(scene, surf_goal)
    q_start, ok_s = transport._place_feasible(scene, surf_start, start.x, start.y, lo_s, rng)
    q_goal, ok_g = transport._place_feasible(scene, surf_goal, goal.x, goal.y, lo_g, rng)
    if not (ok_s and ok_g):
        return False, time.perf_counter() - t0, float("nan"), float("nan")
    ff = transport.free_flight_mode(scene)
    seg = transport._plan_segment(
        scene.robot, ff, q_start, q_goal,
        planner="RRTConnect", solve_time=solve_time, delta=delta,
        constrained_space=space)
    ok = seg is not None and seg.success
    path = seg.path if ok else None
    elapsed = time.perf_counter() - t0
    return ok, elapsed, _peak_utilisation(scene, path), _path_length(path)


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
    ap.add_argument("--n-sep", type=int, default=12)
    ap.add_argument("--sep-min", type=float, default=0.10)
    ap.add_argument("--sep-max", type=float, default=0.90,
                    help="requested max separation; capped at 2 x reachable y")
    ap.add_argument("--derate-min", type=float, default=0.30)
    ap.add_argument("--derate-max", type=float, default=0.80)
    ap.add_argument("--derate-step", type=float, default=0.10)
    ap.add_argument("--reach-x", type=float, default=None,
                    help="fixed reach x for both endpoints; default = mid of the "
                         "two-table shared x-range from CFG")
    ap.add_argument("--solve-time", type=float, default=30.0)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--space", type=str, default="atlas",
                    choices=["atlas", "projected"])
    args = ap.parse_args()

    case = payload_model.normalise_case()
    out_csv = os.path.join(HERE, "..", "data",
                           f"exp_separation_derate_sweep_{case}.csv")

    # --- geometry derived from the existing two-table scene -------------------
    ax = np.asarray(TABLE_A["xy_bounds"], float)
    bx = np.asarray(TABLE_B["xy_bounds"], float)
    x_lo = max(ax[0, 0], bx[0, 0])
    x_hi = min(ax[1, 0], bx[1, 0])
    reach_x = args.reach_x if args.reach_x is not None else 0.5 * (x_lo + x_hi)
    height = float(TABLE_A["height"])

    # wall occupies |y| <= wall_half; surfaces must start beyond it so endpoints
    # are never placed inside the wall footprint. Half-thickness + keep-out.
    from run_panda import GAP_WALL, OBSTACLE_MARGIN
    wall_half = 0.5 * float(GAP_WALL["ysize"])  # already includes 2*margin
    # the smallest meaningful separation puts each endpoint just clear of the wall
    sep_floor = 2.0 * (wall_half + 0.02)

    n_der = int(round((args.derate_max - args.derate_min) / args.derate_step)) + 1
    derates = np.round(args.derate_min + args.derate_step * np.arange(n_der), 3)
    rng = np.random.default_rng(20260711)

    # probe reachable y-extent once, at the LEAST generous derate (smallest zone),
    # on the start-side surface, marching outward from just clear of the wall.
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
        print(f"[{case}] reachable |y|={reach_y:.3f} leaves no room between the "
              f"wall floor sep={sep_floor:.3f} and max sep={sep_max:.3f}; "
              f"try a different --reach-x")
        return
    print(f"[{case}] reach_x={reach_x:.3f}  reachable |y|={reach_y:.3f}  "
          f"wall_half={wall_half:.3f}  => separation swept "
          f"{sep_min:.3f}..{sep_max:.3f} m")

    seps = np.linspace(sep_min, sep_max, args.n_sep)
    rows = []
    threshold = {}

    for rho in derates:
        scene = Scene(_plane_cfg(rho, x_lo, x_hi, reach_y, height, wall_half))
        surf_start = scene.surface(START_SURF)
        surf_goal = scene.surface(GOAL_SURF)
        mono_ok_seps = []
        print(f"\n=== derate {rho:0.2f} ===")
        for sep in seps:
            start = transport.Placement(START_SURF, reach_x, +0.5 * sep)
            goal = transport.Placement(GOAL_SURF, reach_x, -0.5 * sep)

            mono_ok, mono_t, mono_u, mono_len = _monolithic_lift(
                scene, surf_start, surf_goal, start, goal,
                args.solve_time, args.delta, args.space, rng)
            (sls_ok, sls_t, sls_u, sls_len,
             n_lift, n_set, pairs, n_seg) = _slide_lift_slide(
                scene, start, goal, args.solve_time, args.delta, args.space)

            if mono_ok:
                mono_ok_seps.append(float(sep))

            rows.append({
                "case": case, "derate": round(float(rho), 3),
                "separation": round(float(sep), 4),
                "reach_x": round(float(reach_x), 4),
                "mono_success": int(bool(mono_ok)),
                "mono_time_s": round(float(mono_t), 3),
                "mono_peak_u": round(float(mono_u), 4) if mono_u == mono_u else "",
                "mono_path_len": round(float(mono_len), 4) if mono_len == mono_len else "",
                "sls_success": int(bool(sls_ok)),
                "sls_time_s": round(float(sls_t), 3),
                "sls_peak_u": round(float(sls_u), 4) if sls_u == sls_u else "",
                "sls_path_len": round(float(sls_len), 4) if sls_len == sls_len else "",
                "sls_n_lift": n_lift, "sls_n_set": n_set,
                "sls_pairs_tried": pairs, "sls_segments": n_seg,
            })
            print(f"  sep={sep:0.3f}  mono={int(bool(mono_ok))} "
                  f"({mono_t:5.1f}s)  sls={int(bool(sls_ok))} "
                  f"({sls_t:5.1f}s, u={sls_u:0.3f}, {n_seg} segs)")

        threshold[float(rho)] = max(mono_ok_seps) if mono_ok_seps else None

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

    print("\nthreshold separation where monolithic direct lift starts to work:")
    for rho in derates:
        t = threshold[float(rho)]
        if t is None:
            print(f"  derate {rho:0.2f}: never (slide required at all "
                  f"tested separations)")
        else:
            print(f"  derate {rho:0.2f}: direct lift works at separation <= {t:0.3f} m")
    total = len(rows)
    sls_ok = sum(r["sls_success"] for r in rows)
    print(f"\nslide-lift-slide succeeded on {sls_ok}/{total} "
          f"(separation x derate) cells")


if __name__ == "__main__":
    main()