"""
exp_success_rate.py -- planning success rate and wall-clock time over many
random start/goal queries, comparing the atlas vs. projected constrained
state space (Section 6.4 claims atlas "reliably produced feasible carries...
where the projected representation stalled" -- this script is what actually
measures that claim instead of just asserting it).

Drop this next to run_panda.py, transport.py, scene.py, payload_model.py in
payload_planner/ and run inside the mmp2 conda env:

    python -B payload_planner/exp_success_rate.py --n-queries 40

Writes a tidy CSV (data/exp_success_rate.csv) in the same style as your other
Ch.7 data files, plus a small summary table printed to stdout that can be
transcribed directly into the thesis table.
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from scene import Scene
import transport
from run_panda import CFG, TABLE_A, TABLE_B  # reuse the exact scene config used elsewhere

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "..", "data", "exp_success_rate.csv")


def _random_placement(rng, table, margin=0.03):
    (xlo, ylo), (xhi, yhi) = np.asarray(table["xy_bounds"], dtype=float)
    x = float(rng.uniform(xlo + margin, xhi - margin))
    y = float(rng.uniform(ylo + margin, yhi - margin))
    return x, y


def run_sweep(n_queries: int, seed: int, spaces, solve_time: float) -> list[dict]:
    rng = np.random.default_rng(seed)
    scene = Scene(CFG)
    rows = []
    for space in spaces:
        n_success = 0
        for k in range(n_queries):
            sx, sy = _random_placement(rng, TABLE_A)
            gx, gy = _random_placement(rng, TABLE_B)
            start = transport.Placement(TABLE_A["name"], sx, sy)
            goal = transport.Placement(TABLE_B["name"], gx, gy)

            t0 = time.perf_counter()
            res = transport.plan(
                scene, start, goal,
                planner="RRTConnect",
                solve_time=solve_time,
                constrained_space=space,
            )
            elapsed = time.perf_counter() - t0
            n_success += int(res.success)

            joint_len = float("nan")
            if res.success:
                full = res.full_path()
                if full is not None and len(full) > 1:
                    joint_len = float(np.sum(np.linalg.norm(np.diff(full, axis=0), axis=1)))

            rows.append(dict(
                space=space,
                query=k,
                start_x=round(sx, 4), start_y=round(sy, 4),
                goal_x=round(gx, 4), goal_y=round(gy, 4),
                success=int(res.success),
                elapsed_s=round(elapsed, 4),
                pairs_tried=res.pairs_tried,
                n_lift=res.n_lift, n_set=res.n_set,
                joint_path_length=round(joint_len, 4) if joint_len == joint_len else "",
                message=res.message,
            ))
            print(f"[{space:9s}] query {k+1:3d}/{n_queries}: "
                  f"{'OK  ' if res.success else 'FAIL'}  t={elapsed:6.2f}s  "
                  f"pairs_tried={res.pairs_tried:3d}  msg={res.message}")
        print(f"== {space}: {n_success}/{n_queries} succeeded "
              f"({100.0*n_success/n_queries:.1f}%) ==\n")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-queries", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--solve-time", type=float, default=15.0,
                     help="per-segment solve budget in seconds (lower than the "
                          "thesis's 30s default so a 40-query sweep finishes in "
                          "reasonable time; raise it for the final numbers)")
    ap.add_argument("--spaces", nargs="+", default=["atlas", "projected"])
    args = ap.parse_args()

    rows = run_sweep(args.n_queries, args.seed, args.spaces, args.solve_time)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}")

    # Summary table (paste straight into the thesis table)
    print("\nsummary:")
    print(f"{'space':10s} {'success rate':>12s} {'mean time (s)':>14s} {'median time (s)':>16s}")
    for space in args.spaces:
        sub = [r for r in rows if r["space"] == space]
        succ = [r for r in sub if r["success"]]
        rate = 100.0 * len(succ) / len(sub)
        times = [r["elapsed_s"] for r in succ] or [float("nan")]
        print(f"{space:10s} {rate:11.1f}% {np.mean(times):14.2f} {np.median(times):16.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())