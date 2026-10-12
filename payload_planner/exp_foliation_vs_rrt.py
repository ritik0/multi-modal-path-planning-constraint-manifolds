"""
exp_foliation_vs_rrt.py -- the head-to-head that decides whether the foliation
refinement earns a place in the thesis.

The supervisor's bar, stated precisely: fixing a carry height and reporting its
torque utilisation is only meaningful if the SELECTED height's peak torque is
materially LOWER than what a plain RRT-Connect free carry produces -- AND we
must report what that reduction COSTS (extra path length, extra planning time).
A refinement that lowers peak torque by a rounding error, or that lowers it only
by making the path much longer, is not worth including.

For each payload case this script plans the SAME free carry (same lift-off and
set-down endpoints) two ways, repeated over several random endpoint pairs so the
comparison is not a single lucky instance:

  (A) PLAIN: one RRT-Connect free-flight plan (PAYLOAD_FOLIATION=0 behaviour).
  (B) FOLIATION: the height-selected leaf (PAYLOAD_FOLIATION=1 behaviour).

and reports, per pair and per method: peak utilisation, path length, planning
time. The summary prints the mean peak-u reduction and the mean path-length and
time penalty, which is exactly the include/cut decision.

Emits data/exp_foliation_vs_rrt.csv (one row per case x pair x method).

Run per payload (the case is an env var, so loop it in your shell):

    for c in 2p1 3p1 4p1; do
      PAYLOAD_CASE=$c PAYLOAD_TORQUE_DERATE=0.60 PAYLOAD_WALL=1 \
      python -B payload_planner/exp_foliation_vs_rrt.py \
          --n-pairs 8 --solve-time 30 --append
    done

--append lets the three per-case runs accumulate into one CSV. The FIRST run of
the loop should NOT pass --append (it writes the header); the script handles this
by only writing a header when the file does not yet exist.
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
import payload_model

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "..", "data", "exp_foliation_vs_rrt.csv")


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


def _sample_endpoint_pairs(scene, n_pairs, rng, solve_time, delta, space):
    """
    Find n_pairs of (q_lift on A, q_set on B) that are BOTH torque-feasible
    lift-off configurations, so the free carry between them is a fair test for
    both planners. Returns a list of (q_lift, q_set).
    """
    lo_A = transport.liftoff_mode(scene, scene.surface(TABLE_A["name"]))
    lo_B = transport.liftoff_mode(scene, scene.surface(TABLE_B["name"]))
    surf_A = scene.surface(TABLE_A["name"])
    surf_B = scene.surface(TABLE_B["name"])
    (axlo, aylo), (axhi, ayhi) = np.asarray(TABLE_A["xy_bounds"], float)
    (bxlo, bylo), (bxhi, byhi) = np.asarray(TABLE_B["xy_bounds"], float)
    pairs = []
    tries = 0
    while len(pairs) < n_pairs and tries < 40 * n_pairs:
        tries += 1
        ax = float(rng.uniform(axlo + 0.03, axhi - 0.03))
        ay = float(rng.uniform(aylo + 0.03, ayhi - 0.03))
        bx = float(rng.uniform(bxlo + 0.03, bxhi - 0.03))
        by = float(rng.uniform(bylo + 0.03, byhi - 0.03))
        q_lift, ok_l = transport._place_feasible(scene, surf_A, ax, ay, lo_A, rng)
        q_set, ok_s = transport._place_feasible(scene, surf_B, bx, by, lo_B, rng)
        if ok_l and ok_s:
            pairs.append((q_lift, q_set))
    return pairs


def _plan_plain(scene, q_lift, q_set, solve_time, delta, space):
    ff = transport.free_flight_mode(scene)
    t0 = time.perf_counter()
    seg = transport._plan_segment(
        scene.robot, ff, q_lift, q_set,
        planner="RRTConnect", solve_time=solve_time, delta=delta,
        constrained_space=space)
    elapsed = time.perf_counter() - t0
    ok = seg is not None and seg.success
    path = seg.path if ok else None
    return ok, elapsed, _peak_utilisation(scene, path), _path_length(path)


def _plan_foliated(scene, q_lift, q_set, solve_time, delta, space):
    """
    Uses transport._plan_free_flight_foliated (the opt-in leaf-selecting carry).
    Falls back gracefully -- see that function's own fallback -- if the
    transfer_foliation module is unavailable.
    """
    t0 = time.perf_counter()
    seg = transport._plan_free_flight_foliated(
        scene, q_lift, q_set,
        planner="RRTConnect", solve_time=solve_time, delta=delta,
        constrained_space=space)
    elapsed = time.perf_counter() - t0
    ok = seg is not None and getattr(seg, "success", False)
    path = getattr(seg, "path", None) if ok else None
    return ok, elapsed, _peak_utilisation(scene, path), _path_length(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=8)
    ap.add_argument("--solve-time", type=float, default=30.0)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--space", type=str, default="atlas",
                    choices=["atlas", "projected"])
    ap.add_argument("--append", action="store_true",
                    help="append to the CSV instead of overwriting (for looping "
                         "over payload cases)")
    args = ap.parse_args()

    case = payload_model.normalise_case()
    scene = Scene(CFG)
    rng = np.random.default_rng(hash(case) % (2**32))

    pairs = _sample_endpoint_pairs(
        scene, args.n_pairs, rng, args.solve_time, args.delta, args.space)
    if not pairs:
        print(f"[{case}] no torque-feasible endpoint pairs found; nothing to compare")
        return

    rows = []
    for i, (q_lift, q_set) in enumerate(pairs):
        p_ok, p_t, p_u, p_len = _plan_plain(
            scene, q_lift, q_set, args.solve_time, args.delta, args.space)
        f_ok, f_t, f_u, f_len = _plan_foliated(
            scene, q_lift, q_set, args.solve_time, args.delta, args.space)
        rows.append({
            "case": case, "pair": i, "method": "plain_rrt",
            "success": int(bool(p_ok)), "peak_u": round(p_u, 4) if p_u == p_u else "",
            "path_len": round(p_len, 4), "plan_time_s": round(p_t, 3),
        })
        rows.append({
            "case": case, "pair": i, "method": "foliation",
            "success": int(bool(f_ok)), "peak_u": round(f_u, 4) if f_u == f_u else "",
            "path_len": round(f_len, 4), "plan_time_s": round(f_t, 3),
        })
        du = (p_u - f_u) if (p_u == p_u and f_u == f_u) else float("nan")
        print(f"[{case}] pair {i}: plain u={p_u:0.3f} len={p_len:0.2f}  "
              f"folia u={f_u:0.3f} len={f_len:0.2f}  "
              f"(u reduced by {du:+0.3f})")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    write_header = not (args.append and os.path.exists(OUT_CSV))
    mode = "a" if (args.append and os.path.exists(OUT_CSV)) else "w"
    with open(OUT_CSV, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"\n[{case}] wrote {OUT_CSV} (mode={mode})")

    # per-case decision summary
    both = [
        (r_p["peak_u"], r_f["peak_u"], r_p["path_len"], r_f["path_len"],
         r_p["plan_time_s"], r_f["plan_time_s"])
        for r_p, r_f in zip(rows[0::2], rows[1::2])
        if r_p["peak_u"] != "" and r_f["peak_u"] != ""
    ]
    if both:
        arr = np.array(both, float)
        du = arr[:, 0] - arr[:, 1]          # plain_u - folia_u  (positive = folia better)
        dlen = arr[:, 3] - arr[:, 2]        # folia_len - plain_len (positive = longer)
        dt = arr[:, 5] - arr[:, 4]          # folia_time - plain_time (positive = slower)
        print(f"[{case}] mean peak-u reduction from foliation: {du.mean():+0.4f} "
              f"(min {du.min():+0.4f}, max {du.max():+0.4f})")
        print(f"[{case}] mean path-length penalty:  {dlen.mean():+0.4f}")
        print(f"[{case}] mean planning-time penalty: {dt.mean():+0.2f} s")
        print(f"[{case}] DECISION HINT: include foliation only if the peak-u "
              f"reduction is worth the path/time penalty above.")


if __name__ == "__main__":
    main()