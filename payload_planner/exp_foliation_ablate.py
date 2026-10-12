"""
exp_foliation_ablate.py -- compares the golden-section search actually used
by transfer_foliation._select_continuous (Algorithm 4, Section 5.8.2)
against brute-force grid search, on the SAME real cost function
(transfer_foliation.evaluate_leaf), for a real (q_lift, q_set) pair pulled
from an actual successful transport.plan() run.

IMPORTANT -- each evaluation is real planning, not a cheap function call.
evaluate_leaf() runs up to three RRT-Connect solves (lift->leaf, cross,
leaf->set), each with its own `solve_time` budget. A "200-point grid" from
a textbook golden-section demo would take hours here. Grid sizes below are
deliberately small (default: 4, 8, 12) -- raise them only after timing a
handful of evaluations yourself.

Run:
    python -B exp_foliation_ablate.py --n-lift-set-samples 200 --solve-time 8

--n-lift-set-samples controls how many transition candidates
sample_transitions tries when looking for a usable (q_lift, q_set) pair
(passed through to transport.plan as n_tries) -- it is NOT the number of
foliation evaluations. Lower --solve-time from the thesis default (15s) for
a faster first pass; raise it back for the final numbers.

--self-test still works exactly as before (synthetic curve, NOT thesis
data) if you just want to sanity-check the search code with no planner.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from scene import Scene
import transport as T
import transfer_foliation as fol
from run_panda import CFG, TABLE_A, TABLE_B

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "..", "data", "exp_foliation_ablation.csv")


# --------------------------------------------------------------------------- #
# Real cost function: wraps transfer_foliation.evaluate_leaf into a plain
# cost(h) -> float callable, +inf for infeasible heights (matches the
# 1e6 penalty _select_continuous itself uses internally).
# --------------------------------------------------------------------------- #
def make_real_cost_fn(scene, q_lift, q_set, planner, solve_time, delta, space,
                       w_torque, w_path, verbose=False):
    def cost(h):
        ok, score, info = fol.evaluate_leaf(
            scene, q_lift, q_set, float(h),
            planner=planner, solve_time=solve_time, delta=delta, space=space,
            w_torque=w_torque, w_path=w_path,
        )
        if verbose:
            tag = "OK  " if ok else "FAIL"
            print(f"      eval h={h:.4f}: {tag} score={score if ok else float('inf'):.4f} "
                  f"({info.get('reason', '') if not ok else ''})")
        return float(score) if ok else float("inf")
    return cost


def make_synthetic_cost_fn(floor, h_lo, h_hi, k=18.0, h_knee=None):
    """--self-test only. NOT real planner output -- see module docstring."""
    if h_knee is None:
        h_knee = h_lo + 0.35 * (h_hi - h_lo)

    def cost(h):
        s = 1.0 / (1.0 + np.exp(-k * (float(h) - h_knee)))
        return floor + (0.99 - floor) * (1.0 - s)
    return cost


# --------------------------------------------------------------------------- #
# The two search strategies under comparison.
# --------------------------------------------------------------------------- #
def golden_section(cost_fn, a: float, b: float, iters: int, tol: float = 0.01):
    """Mirrors transfer_foliation._select_continuous exactly: same
    convergence break (|b-a| < tol) in addition to the iteration cap."""
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = cost_fn(c), cost_fn(d)
    n_evals = 2
    for _ in range(max(3, iters)):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = cost_fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = cost_fn(d)
        n_evals += 1
    h_best, u_best = (c, fc) if fc < fd else (d, fd)
    return h_best, u_best, n_evals


def grid_search(cost_fn, a: float, b: float, n_points: int):
    hs = np.linspace(a, b, n_points)
    us = [cost_fn(h) for h in hs]
    i = int(np.argmin(us))
    return float(hs[i]), float(us[i]), n_points


# --------------------------------------------------------------------------- #
def get_real_lift_set_pair(scene, planner, solve_time, n_tries, space):
    """Run one real transport.plan() (no foliation) and pull its accepted
    (q_lift, q_set) pair -- the same kind of pair Algorithm 4 would actually
    receive in a real run, rather than an invented one."""
    start = T.Placement(TABLE_A["name"],
                         0.5 * (TABLE_A["xy_bounds"][0][0] + TABLE_A["xy_bounds"][1][0]),
                         0.5 * (TABLE_A["xy_bounds"][0][1] + TABLE_A["xy_bounds"][1][1]))
    goal = T.Placement(TABLE_B["name"],
                        0.5 * (TABLE_B["xy_bounds"][0][0] + TABLE_B["xy_bounds"][1][0]),
                        0.5 * (TABLE_B["xy_bounds"][0][1] + TABLE_B["xy_bounds"][1][1]))
    res = T.plan(scene, start, goal, planner=planner, solve_time=solve_time,
                 n_tries=n_tries, constrained_space=space)
    if not res.success or res.lift_off is None or res.set_down is None:
        raise RuntimeError(
            f"could not obtain a real (q_lift, q_set) pair: {res.message}. "
            f"Try raising --solve-time or --n-lift-set-samples."
        )
    return res.lift_off, res.set_down


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                     help="use the synthetic stand-in cost (verifies the script, "
                          "NOT thesis data)")
    ap.add_argument("--planner", default="RRTConnect")
    ap.add_argument("--solve-time", type=float, default=8.0,
                     help="per-segment solve budget for evaluate_leaf's three "
                          "RRT-Connect calls (thesis default is 15s; lower for "
                          "a faster first pass)")
    ap.add_argument("--delta", type=float, default=0.08)
    ap.add_argument("--space", default="atlas", choices=["atlas", "projected"])
    ap.add_argument("--w-torque", type=float, default=10.0)
    ap.add_argument("--w-path", type=float, default=1.0)
    ap.add_argument("--golden-iters", type=int, default=int(
        os.environ.get("PAYLOAD_TRANSFER_ITERS", "8")),
        help="matches transfer_foliation's own PAYLOAD_TRANSFER_ITERS default (8)")
    ap.add_argument("--grid-sizes", nargs="+", type=int, default=[4, 8, 12],
                     help="kept small on purpose -- each point is real planning, "
                          "not a cheap function call")
    ap.add_argument("--n-lift-set-samples", type=int, default=200,
                     help="transition candidates tried when finding a real "
                          "(q_lift, q_set) pair (transport.plan's n_tries)")
    args = ap.parse_args()

    if args.self_test:
        print("*** --self-test: synthetic cost function, NOT real planner output ***\n")
        rows = []
        for mass, floor in [(2.1, 0.58), (3.1, 0.69), (4.1, 0.82)]:
            cost_fn = make_synthetic_cost_fn(floor, 0.20, 0.60)
            t0 = time.perf_counter()
            h_g, u_g, n_g = golden_section(cost_fn, 0.20, 0.60, iters=args.golden_iters)
            t_golden = time.perf_counter() - t0
            row = {"case": f"{mass}kg (SYNTHETIC)", "golden_h": round(h_g, 5),
                   "golden_u": round(u_g, 5), "golden_evals": n_g,
                   "golden_time_s": round(t_golden, 6)}
            for n in args.grid_sizes:
                t0 = time.perf_counter()
                h_gr, u_gr, n_gr = grid_search(cost_fn, 0.20, 0.60, n)
                row[f"grid{n}_u"] = round(u_gr, 5)
                row[f"grid{n}_gap_vs_golden"] = round(u_gr - u_g, 6)
                row[f"grid{n}_time_s"] = round(time.perf_counter() - t0, 6)
            rows.append(row)
            print(row)
        return 0

    print("building scene and finding a real (q_lift, q_set) transition pair "
          "via transport.plan() (no foliation) -- this runs the real planner "
          "and may take a while...")
    scene = Scene(CFG)
    q_lift, q_set = get_real_lift_set_pair(
        scene, args.planner, args.solve_time, args.n_lift_set_samples, args.space)
    print("got a real (q_lift, q_set) pair from transport.plan().\n")

    h_min, h_max = fol._height_bounds(scene)
    print(f"height search interval from transfer_foliation._height_bounds: "
          f"[{h_min:.3f}, {h_max:.3f}]")

    cost_fn = make_real_cost_fn(
        scene, q_lift, q_set, args.planner, args.solve_time, args.delta, args.space,
        args.w_torque, args.w_path, verbose=True,
    )

    print(f"\nrunning golden-section search ({args.golden_iters} iters cap, "
          f"same convergence rule as transfer_foliation._select_continuous)...")
    t0 = time.perf_counter()
    h_g, s_g, n_g = golden_section(cost_fn, h_min, h_max, iters=args.golden_iters)
    t_golden = time.perf_counter() - t0
    print(f"golden-section: h*={h_g:.4f}  score*={s_g:.4f}  "
          f"evals={n_g}  wall time={t_golden:.1f}s\n")

    row = {
        "golden_h": round(h_g, 5), "golden_score": round(s_g, 5),
        "golden_evals": n_g, "golden_time_s": round(t_golden, 3),
    }
    for n in args.grid_sizes:
        print(f"running grid search ({n} points)...")
        t0 = time.perf_counter()
        h_gr, s_gr, n_gr = grid_search(cost_fn, h_min, h_max, n)
        t_grid = time.perf_counter() - t0
        gap = s_gr - s_g
        print(f"grid[{n}]: h={h_gr:.4f}  score={s_gr:.4f}  "
              f"gap vs golden={gap:.4f}  wall time={t_grid:.1f}s\n")
        row[f"grid{n}_h"] = round(h_gr, 5)
        row[f"grid{n}_score"] = round(s_gr, 5)
        row[f"grid{n}_gap_vs_golden"] = round(gap, 6)
        row[f"grid{n}_time_s"] = round(t_grid, 3)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())