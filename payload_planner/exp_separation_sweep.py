"""
exp_separation_sweep.py -- the headline Ch.7 experiment.

Single continuous plane with a wall in the middle (the real-robot scene). Start
and goal sit on opposite sides of the wall. We sweep their separation from far
apart to close together and, at each separation, ask two planners to solve it:

  (A) MONOLITHIC direct lift: one free-flight constrained plan from start to
      goal, no sliding, no transitions -- the object is carried the whole way.
      This is what a torque-blind constrained planner would attempt.

  (B) SLIDE-LIFT-SLIDE: the full multimodal transport planner of this thesis,
      which may slide on the plane, lift off, cross, and set down.

The story we expect (and this script measures, rather than assumes):
  - When start/goal are FAR apart, the crossing sits deep in the overloaded
    region: (A) fails, only (B) succeeds by sliding out to a torque-feasible
    lift-off. Sliding is what makes the task solvable at all.
  - As they get CLOSER, both endpoints move toward the arm's lift envelope.
    Below some separation, (A) succeeds directly -- no sliding needed.
  - (B) succeeds across the WHOLE range; (A) only below the threshold.

Emits data/exp_separation_sweep.csv with one row per (separation, planner).

Run inside the mmp2 conda env, from payload_planner/ , with the SINGLE-PLANE
wall scene enabled (see the command block in the thesis chat). Example:

    PAYLOAD_CASE=3p1 PAYLOAD_TORQUE_DERATE=0.60 PAYLOAD_WALL=1 \
    python -B payload_planner/exp_separation_sweep.py \
        --n-sep 12 --sep-min 0.10 --sep-max 0.55 --solve-time 30

The geometry (where start/goal sit for a given separation) is expressed on the
plane as two points straddling the wall in x, symmetric about the wall line, so
"separation" is the straight-line gap between them across the wall. Adjust the
--axis and --wall-x defaults to match your measured table if they differ.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from scene import Scene
import transport
from run_panda import CFG, TABLE_A, TABLE_B

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "..", "data", "exp_separation_sweep.csv")


def _peak_utilisation(scene, path):
    """Worst-joint gravity-torque utilisation over a path (free-flight sense)."""
    if path is None or len(path) == 0:
        return float("nan")
    tau_max = np.asarray(scene.tau_max, float) * float(scene.derate)
    umax = 0.0
    for q in np.asarray(path, float):
        u = np.max(np.abs(scene.robot.gravity(q)) / tau_max)
        umax = max(umax, float(u))
    return umax


def _path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    p = np.asarray(path, float)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _monolithic_lift(scene, start, goal, solve_time, delta, space, rng):
    """
    Attempt a single free-flight plan start->goal with NO sliding.

    Both endpoints must be held by the arm ALONE (torque-feasible free flight):
    we place each on its surface subject to the lift-off mode (surface pose that
    is also torque-feasible), then plan ONE free-flight segment between them. If
    either endpoint cannot be placed torque-feasibly, a monolithic lift is
    impossible by definition and we report failure without planning.

    Returns (success, elapsed, peak_u, path_len). success is None only if the
    helper wiring is wrong (should not happen with the current transport.py).
    """
    t0 = time.perf_counter()

    lo_start = transport.liftoff_mode(scene, start.surface)
    lo_goal = transport.liftoff_mode(scene, goal.surface)
    q_start, ok_s = transport._place_feasible(
        scene, start.surface, start.x, start.y, lo_start, rng)
    q_goal, ok_g = transport._place_feasible(
        scene, goal.surface, goal.x, goal.y, lo_goal, rng)
    if not (ok_s and ok_g):
        # at least one endpoint is not torque-feasible to hold freely ->
        # a no-slide direct lift cannot even start
        return False, time.perf_counter() - t0, float("nan"), float("nan")

    ff = transport.free_flight_mode(scene)
    seg = transport._plan_segment(
        scene.robot, ff, q_start, q_goal,
        planner="RRTConnect", solve_time=solve_time, delta=delta,
        constrained_space=space)
    ok = seg is not None and getattr(seg, "success", False)
    path = seg.path if ok else None
    elapsed = time.perf_counter() - t0
    return ok, elapsed, _peak_utilisation(scene, path), _path_length(path)


def _slide_lift_slide(scene, start, goal, solve_time, delta, space):
    t0 = time.perf_counter()
    res = transport.plan(
        scene, start, goal,
        planner="RRTConnect", solve_time=solve_time, delta=delta,
        constrained_space=space,
    )
    elapsed = time.perf_counter() - t0
    path = res.full_path() if res.success else None
    return (
        res.success, elapsed,
        _peak_utilisation(scene, path), _path_length(path),
        res.n_lift, res.n_set, res.pairs_tried,
        len(res.segments) if res.success else 0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sep", type=int, default=12,
                    help="number of separations to sweep")
    ap.add_argument("--sep-min", type=float, default=0.10,
                    help="smallest start-goal separation across the wall [m]")
    ap.add_argument("--sep-max", type=float, default=0.55,
                    help="largest start-goal separation across the wall [m]")
    ap.add_argument("--axis", type=float, default=0.235,
                    help="the fixed x (reach) coordinate both points share [m]")
    ap.add_argument("--wall-y", type=float, default=0.0,
                    help="y of the wall line the points straddle [m]")
    ap.add_argument("--solve-time", type=float, default=30.0)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--space", type=str, default="atlas",
                    choices=["atlas", "projected"])
    ap.add_argument("--surface", type=str, default=None,
                    help="surface name to place both endpoints on "
                         "(defaults to TABLE_A's name; single-plane scene)")
    args = ap.parse_args()

    scene = Scene(CFG)
    surface = args.surface or TABLE_A["name"]
    seps = np.linspace(args.sep_min, args.sep_max, args.n_sep)
    rng = np.random.default_rng(20260711)

    rows = []
    for sep in seps:
        # two points straddling the wall line in y, symmetric, same reach x
        y_start = args.wall_y + 0.5 * sep
        y_goal = args.wall_y - 0.5 * sep
        start = transport.Placement(surface, args.axis, y_start)
        goal = transport.Placement(surface, args.axis, y_goal)

        mono_ok, mono_t, mono_u, mono_len = _monolithic_lift(
            scene, start, goal, args.solve_time, args.delta, args.space, rng)
        (sls_ok, sls_t, sls_u, sls_len,
         n_lift, n_set, pairs, n_seg) = _slide_lift_slide(
            scene, start, goal, args.solve_time, args.delta, args.space)

        row = {
            "separation": round(float(sep), 4),
            "y_start": round(float(y_start), 4),
            "y_goal": round(float(y_goal), 4),
            "mono_success": "" if mono_ok is None else int(bool(mono_ok)),
            "mono_time_s": round(float(mono_t), 3),
            "mono_peak_u": round(float(mono_u), 4) if mono_u == mono_u else "",
            "mono_path_len": round(float(mono_len), 4) if mono_len == mono_len else "",
            "sls_success": int(bool(sls_ok)),
            "sls_time_s": round(float(sls_t), 3),
            "sls_peak_u": round(float(sls_u), 4) if sls_u == sls_u else "",
            "sls_path_len": round(float(sls_len), 4) if sls_len == sls_len else "",
            "sls_n_lift": n_lift,
            "sls_n_set": n_set,
            "sls_pairs_tried": pairs,
            "sls_segments": n_seg,
        }
        rows.append(row)
        print(
            f"sep={sep:0.3f}  "
            f"mono={'--' if mono_ok is None else int(bool(mono_ok))} "
            f"({mono_t:5.1f}s)  "
            f"sls={int(bool(sls_ok))} ({sls_t:5.1f}s, "
            f"{n_seg} segs, u={sls_u:0.3f})"
        )

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")

    # summary: the threshold separation where monolithic lift starts to succeed
    mono_success_seps = [r["separation"] for r in rows if r["mono_success"] == 1]
    if mono_success_seps:
        print(f"monolithic direct lift succeeds at separations <= "
              f"{max(mono_success_seps):0.3f} m")
    else:
        print("monolithic direct lift never succeeded in the swept range")
    sls_success = sum(1 for r in rows if r["sls_success"] == 1)
    print(f"slide-lift-slide succeeded on {sls_success}/{len(rows)} separations")


if __name__ == "__main__":
    main()