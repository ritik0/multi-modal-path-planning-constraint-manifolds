#!/usr/bin/env python3
"""
exp_freehold_map.py  --  the workspace free-hold partition (deterministic, planner-free).

WHAT THIS ANSWERS
-----------------
The separation study asked "how far apart can start and goal be before the arm
must slide?" but measured it with a stochastic planner, so the peak it reported
was a route the planner happened to find, not the demand the placement forces.
Gravity torque depends on REACH from the base, not on start-goal separation, so
two placements the same distance apart can have very different torque -- which is
why the separation curves are mushy.

This experiment removes the planner entirely.  For every (x,y) placement on the
surface plane it computes the SMALLEST peak gravity utilisation achievable over
the 1-D self-motion (null space) of the flat-carry pose:

    u_star_raw(x,y) = min_{q : payload flat at (x,y,h)} max_j |g_j(q)| / tau_max_j

This is a property of the placement, not of any trajectory.  It is also
INDEPENDENT of the derate: rho enters only as a scalar threshold, so one field
per payload answers every derate at once.  A placement is:

    free-hold at derate rho   <=>   u_star_raw(x,y) <= rho     (arm can lift here)
    must-slide at derate rho   <=>   u_star_raw(x,y) >  rho     (needs the surface)
    unreachable                <=>   no flat-carry IK converged at (x,y)

The output CSV is a dense grid; the companion plotting script turns it into the
partition map (filled contours) and the reach-collapse curve.  The threshold
reach r*(rho) = min reach at which u_star_raw first exceeds rho is the single
number the separation study was trying to produce, now on the correct axis.

DESIGN CHOICES (see the thesis discussion)
------------------------------------------
* The min over the self-motion is found the same way the PLANNER finds postures:
  multi-seed projection onto the flat-carry manifold, then greedy null-space
  descent (reusing nullspace_torque_opt.optimise_config).  This is deliberately
  the arm's own capability -- "the best posture the method could reach" -- not a
  global optimiser the planner could never match.  It is therefore an UPPER bound
  on the true free-hold region: if this says must-slide, a global IK might still
  find a holdable posture, so the reported must-slide set is conservative.  We
  state that explicitly rather than claim a global minimum.
* Grid is the full reachable annulus in the surface plane, not just the two table
  patches, so the whole partition is visible.  Placements the planner's scene
  would never use are still physically meaningful free-hold/must-slide facts.

USAGE (same env convention as the other exp_*.py, run next to run_panda.py)
--------------------------------------------------------------------------
  PAYLOAD_CASE=2p1 python -B exp_freehold_map.py --grid 121 --seeds 12
  PAYLOAD_CASE=3p1 python -B exp_freehold_map.py --grid 121 --seeds 12
  PAYLOAD_CASE=4p1 python -B exp_freehold_map.py --grid 121 --seeds 12

Writes  ../data/exp_freehold_map_<case>.csv  with columns:
  x, y, reach, reachable, u_star_raw, n_seed_ok
"""

import argparse
import csv
import os
import time

import numpy as np

from scene import Scene
from constraints import FrameConstraint, FlatCarryConstraint, Mode
from run_panda import CFG                      # the exact scene/payload config used elsewhere
from nullspace_torque_opt import optimise_config, peak_util

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.environ.get("PAYLOAD_CASE", "unknown")
OUT_CSV = os.path.join(HERE, "..", "data", f"exp_freehold_map_{CASE}.csv")


# --------------------------------------------------------------------------- #
# One placement: project the flat-carry pose from several seeds, keep the best
# (lowest peak-utilisation) posture, then null-space-descend it.  derate=1.0 so
# peak_util returns u_raw exactly.
# --------------------------------------------------------------------------- #
def freehold_at(scene, R_flat, x, y, h, seeds, rng, ns_step, ns_steps):
    robot = scene.robot
    frame = scene.controlled_frame
    tau_max = scene.tau_max
    qlo, qhi = robot.joint_limits
    qlo = np.asarray(qlo, float); qhi = np.asarray(qhi, float)

    # flat-carry pose pinned to this (x, y, h): position fully pinned, roll/pitch
    # pinned, yaw free -- identical equality bundle to the on-surface transition
    # mode, minus the surface inequalities we are deliberately not testing here.
    place = FrameConstraint(robot, frame,
                            target_position=[x, y, h], pos_mask=(1, 1, 1),
                            target_rotation=R_flat, rot_mask=(1, 1, 0),
                            name="place_flat")
    mode = Mode("place_flat", equalities=[place], inequalities=[])

    best_u = np.inf
    n_ok = 0
    qmid = 0.5 * (qlo + qhi)
    for s in range(seeds):
        seed = qmid if s == 0 else rng.uniform(qlo, qhi)
        q, ok = mode.project(seed, qlo=qlo, qhi=qhi)
        if not ok:
            continue
        # reject if projection drifted off the requested (x,y): a converged
        # projection can satisfy the equality at a different branch only if the
        # target was reachable, so a residual check on position is the reach test.
        p = robot.frame_pose(frame, q)[:3, 3]
        if abs(p[0] - x) > 1e-3 or abs(p[1] - y) > 1e-3 or abs(p[2] - h) > 1e-3:
            continue
        n_ok += 1
        # null-space descent to the lowest-torque posture at this fixed pose
        q_opt, u = optimise_config(robot, tau_max, 1.0, frame, q,
                                   step=ns_step, n_steps=ns_steps)
        if u < best_u:
            best_u = u

    reachable = n_ok > 0
    return (best_u if reachable else np.nan), reachable, n_ok


def run_map(grid, seeds, extent, ns_step, ns_steps, seed):
    scene = Scene(CFG)
    robot = scene.robot
    frame = scene.controlled_frame
    h = scene.table_height
    rng = np.random.default_rng(seed)

    # R_flat: the reference gripper-down orientation.  Recover it from a
    # successful flat projection at a near-base placement, so it matches whatever
    # convention the live scene uses rather than assuming eye(3).
    R_flat = np.eye(3)
    probe = FrameConstraint(robot, frame,
                            target_position=[0.35, 0.0, h], pos_mask=(1, 1, 1),
                            target_rotation=R_flat, rot_mask=(0, 0, 0),
                            name="probe")
    qlo, qhi = robot.joint_limits
    q0, ok = Mode("probe", equalities=[probe]).project(
        0.5 * (np.asarray(qlo, float) + np.asarray(qhi, float)),
        qlo=qlo, qhi=qhi)
    if ok:
        R_flat = robot.frame_pose(frame, q0)[:3, :3].copy()
    # keep only the "z-down" part: enforce roll/pitch = flat, leave yaw as-is.
    # (FlatCarryConstraint uses R_flat's z-axis; using the probe's full R is fine
    #  because rot_mask=(1,1,0) ignores yaw.)

    xs = np.linspace(-extent, extent, grid)
    ys = np.linspace(-extent, extent, grid)

    rows = []
    t0 = time.perf_counter()
    n_reach = 0
    for iy, y in enumerate(ys):
        for x in xs:
            r = float(np.hypot(x, y))
            u, reachable, n_ok = freehold_at(scene, R_flat, x, y, h,
                                             seeds, rng, ns_step, ns_steps)
            if reachable:
                n_reach += 1
            rows.append({
                "x": round(x, 5), "y": round(y, 5), "reach": round(r, 5),
                "reachable": int(reachable),
                "u_star_raw": ("" if not reachable else round(float(u), 5)),
                "n_seed_ok": n_ok,
            })
        dt = time.perf_counter() - t0
        done = (iy + 1) * grid
        print(f"  row {iy+1:3d}/{grid}  reachable so far {n_reach:5d}/{done:5d}  "
              f"[{dt:6.1f}s]", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=121,
                    help="grid points per axis (odd centres a sample on the base)")
    ap.add_argument("--seeds", type=int, default=12,
                    help="projection restarts per placement (more = fewer false 'unreachable')")
    ap.add_argument("--extent", type=float, default=0.90,
                    help="half-width of the square xy grid in metres")
    ap.add_argument("--ns-step", type=float, default=0.02,
                    help="null-space descent step (rad)")
    ap.add_argument("--ns-steps", type=int, default=12,
                    help="null-space descent iterations")
    ap.add_argument("--seed", type=int, default=20260718)
    args = ap.parse_args()

    if CASE == "unknown":
        raise SystemExit("set PAYLOAD_CASE=2p1|3p1|4p1 in the environment (as with the other exp_*.py)")

    print("=" * 68)
    print(f"Free-hold partition map   case={CASE}   grid={args.grid}x{args.grid}"
          f"   seeds={args.seeds}")
    print(f"  u_star_raw = min_posture max_j |g_j| / tau_max_j   (derate-independent)")
    print("=" * 68)

    rows = run_map(args.grid, args.seeds, args.extent,
                   args.ns_step, args.ns_steps, args.seed)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "reach", "reachable",
                                          "u_star_raw", "n_seed_ok"])
        w.writeheader()
        w.writerows(rows)

    reach = [r for r in rows if r["reachable"]]
    us = np.array([r["u_star_raw"] for r in reach], float)
    print("=" * 68)
    print(f"wrote {OUT_CSV}  ({len(rows)} cells, {len(reach)} reachable)")
    if len(us):
        for rho in (0.40, 0.50, 0.60, 0.70, 0.80):
            frac = float(np.mean(us > rho)) * 100.0
            print(f"  derate rho={rho:.2f}:  must-slide fraction of reachable = {frac:5.1f}%")
        print(f"  u_star_raw range over reachable set: "
              f"[{us.min():.3f}, {us.max():.3f}]")
    print("=" * 68)


if __name__ == "__main__":
    main()