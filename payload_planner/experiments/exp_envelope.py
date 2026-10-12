#!/usr/bin/env python3
"""
exp_envelope.py -- operating-envelope sweep (Claims 2 and 3), v2.

Computes, for each (payload mass, torque derate) cell, the worst-case lift
utilisation over the reachable workspace and the fraction of poses that must be
slid, by evaluating the torque and support oracles directly on a grid of
placements.  Unlike v1 this does NOT parse run_panda's diagnostic stdout, so it
works at EVERY derate -- including the high-derate liftable regime where the
scenario would otherwise stop early.  That lets the utilisation-vs-derate curves
be seen crossing the feasibility line u = 1.

Outputs (into experiments/):
  envelope_cells.csv    per (case, derate): worst_lift_util, frac_must_slide,
                        frac_infeasible, regime, counts
  frontier_poses.csv    per (case, derate, region, x, y): utilisations + verdict

Run from payload_planner/:
    python3 experiments/exp_envelope.py
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
from scene import Scene               # noqa: E402


CASES = ["2p1", "3p1", "4p1"]
DERATES = [round(0.25 + 0.05 * i, 2) for i in range(14)]   # 0.25 .. 0.90
CASE_MASS = {"2p1": 2.101, "3p1": 3.101, "4p1": 4.101}


def _grid(tbl):
    (xlo, ylo), (xhi, yhi) = np.asarray(tbl["xy_bounds"], float)
    xs = np.linspace(xlo, xhi, 3)
    ys = np.linspace(ylo, yhi, 3)
    return [(float(x), float(y), float(tbl["height"])) for x in xs for y in ys]


def _build_scene(case, derate):
    os.environ["PAYLOAD_CASE"] = case
    os.environ["PAYLOAD_TORQUE_DERATE"] = f"{derate:.2f}"
    os.environ["PAYLOAD_CONSTRAINED_SPACE"] = "atlas"
    import importlib
    importlib.reload(run_panda)
    scene = Scene(run_panda.CFG)
    modes = T.build_modes(scene)
    return scene, modes


def _classify_pose(scene, modes, x, y, z):
    torque = modes["torque"]
    support = modes["support_torque"]
    qlo, qhi = scene.robot.joint_limits
    seed = 0.5 * (np.asarray(qlo, float) + np.asarray(qhi, float))
    q, ok = run_panda._project_flat(scene, x, y, z, seed)
    if not ok or q is None:
        return 0, None, None, None, "unreachable"
    util_lift = float(np.max(torque.utilisation(q)))
    sol = support.solve(q)
    util_support = float(np.max(sol["utilisation"]))
    supp_feasible = bool(sol["feasible"])
    if util_lift <= 1.0:
        verdict = "MAY-LIFT"
    elif supp_feasible:
        verdict = "SUPPORTED-SLIDE"
    else:
        verdict = "INFEASIBLE"
    return 1, util_lift, util_support, supp_feasible, verdict


def main():
    cell_rows = []
    pose_rows = []
    for case in CASES:
        for derate in DERATES:
            scene, modes = _build_scene(case, derate)
            poses = []
            for tbl in (run_panda.TABLE_A, run_panda.TABLE_B):
                region = tbl["name"]
                for (x, y, z) in _grid(tbl):
                    reach, ul, us, sf, verdict = _classify_pose(scene, modes, x, y, z)
                    poses.append({
                        "region": region, "x": x, "y": y, "reachable": reach,
                        "util_lift": ul, "util_support": us,
                        "supp_feasible": sf, "verdict": verdict,
                    })

            reach_poses = [p for p in poses if p["reachable"]]
            n = len(reach_poses)
            if n == 0:
                regime = "no_reachable_poses"
                worst = frac_slide = frac_infeas = ""
            else:
                lifts = [p["util_lift"] for p in reach_poses]
                worst = max(lifts)
                n_slide = sum(1 for p in reach_poses if p["verdict"] == "SUPPORTED-SLIDE")
                n_infeas = sum(1 for p in reach_poses if p["verdict"] == "INFEASIBLE")
                frac_slide = n_slide / n
                frac_infeas = n_infeas / n
                if n_infeas == n:
                    regime = "infeasible_everywhere"
                elif n_infeas > 0:
                    regime = "partly_infeasible"
                elif n_slide == 0:
                    regime = "liftable_everywhere"
                else:
                    regime = "has_must_slide_region"

            ws = worst if worst == "" else round(worst, 3)
            fs = frac_slide if frac_slide == "" else round(frac_slide, 2)
            print(f"[{case} d={derate:.2f}] worst_lift={ws} frac_slide={fs} -> {regime}",
                  flush=True)

            cell_rows.append({
                "case": case, "mass_kg": CASE_MASS[case], "derate": derate,
                "worst_lift_util": worst, "frac_must_slide": frac_slide,
                "frac_infeasible": frac_infeas, "regime": regime,
                "n_reachable": n,
            })
            for p in poses:
                row = {"case": case, "mass_kg": CASE_MASS[case], "derate": derate}
                row.update(p)
                pose_rows.append(row)

    with open(OUT / "envelope_cells.csv", "w", newline="") as f:
        keys = ["case", "mass_kg", "derate", "worst_lift_util", "frac_must_slide",
                "frac_infeasible", "regime", "n_reachable"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(cell_rows)

    with open(OUT / "frontier_poses.csv", "w", newline="") as f:
        keys = ["case", "mass_kg", "derate", "region", "x", "y", "reachable",
                "util_lift", "util_support", "supp_feasible", "verdict"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in pose_rows:
            w.writerow({k: r.get(k, "") for k in keys})

    print(f"\nwrote {OUT/'envelope_cells.csv'} ({len(cell_rows)} cells)")
    print(f"wrote {OUT/'frontier_poses.csv'} ({len(pose_rows)} poses)")


if __name__ == "__main__":
    main()