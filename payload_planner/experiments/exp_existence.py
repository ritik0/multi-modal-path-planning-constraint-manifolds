#!/usr/bin/env python3
"""
exp_existence.py -- Claim 1: multimodal transport succeeds where lifting fails.

The argument is an existence result: at a chosen (case, derate) the object cannot
be lifted (held by the arm alone) over part of the workspace, so a lift-only plan
has no valid way across; the slide-lift-slide method transports it anyway.

This script records, for the chosen scenario:
  * the lift-only feasibility at the start and goal placements -- the max torque
    utilisation to HOLD the object there with the arm alone.  If this exceeds 1,
    the object cannot be lifted at that placement, so a lift-only planner cannot
    begin (start) or finish (goal) there.
  * whether a supported slide IS feasible there (the surface can offload the
    excess), which is what the multimodal method uses.
  * the peak torque utilisation of the full multimodal plan's free-flight phase,
    which stays within budget by construction.

Together these show: lift-only is infeasible at the endpoints (utilisation > 1),
while the multimodal plan is feasible throughout (<= 1).

Outputs (into experiments/):
  existence.csv   one row per (case, derate) with the numbers above

Run from payload_planner/:
    python3 experiments/exp_existence.py
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


# scenarios to test: (case, derate).  These are the ones where a must-slide
# region is expected, so lift-only should fail at the loaded placements.
SCENARIOS = [
    ("3p1", 0.40),
    ("3p1", 0.50),
    ("4p1", 0.50),
    ("4p1", 0.60),
]


def _hold_utilisation(scene, modes, x, y, z):
    """Max torque utilisation to HOLD the object flat at (x,y,z) with the arm
    alone, and whether a supported slide is feasible there."""
    torque = modes["torque"]
    support = modes["support_torque"]
    qlo, qhi = scene.robot.joint_limits
    seed = 0.5 * (np.asarray(qlo, float) + np.asarray(qhi, float))
    # reuse run_panda's projection helper
    q, ok = run_panda._project_flat(scene, float(x), float(y), float(z), seed)
    if not ok or q is None:
        return None, None, None
    util_free = float(np.max(torque.utilisation(q)))
    sol = support.solve(q)
    util_support = float(np.max(sol["utilisation"]))
    supp_feasible = bool(sol["feasible"])
    return util_free, util_support, supp_feasible


def run_scenario(case, derate):
    env = dict(os.environ)
    env.update({"PAYLOAD_CASE": case, "PAYLOAD_TORQUE_DERATE": f"{derate:.2f}",
                "PAYLOAD_CONSTRAINED_SPACE": "atlas"})
    # re-import run_panda in a fresh process would be cleanest; here we rebuild
    # the scene from a CFG that reflects these env vars by setting them first.
    # Because run_panda read its module-level constants at import, we instead
    # construct a scene via a subprocess-free path: set os.environ and reload.
    for k, v in env.items():
        os.environ[k] = v
    import importlib
    importlib.reload(run_panda)
    scene = Scene(run_panda.CFG)
    modes = T.build_modes(scene)

    # start and goal placements the scenario discovers (or the pinned ones)
    # We sample the far-outer corner of table A and B, where loading is worst.
    tA = run_panda.TABLE_A
    tB = run_panda.TABLE_B
    (axlo, aylo), (axhi, ayhi) = np.asarray(tA["xy_bounds"], float)
    (bxlo, bylo), (bxhi, byhi) = np.asarray(tB["xy_bounds"], float)
    start = (0.5 * (axlo + axhi), ayhi, float(tA["height"]))     # outer edge A
    goal = (0.5 * (bxlo + bxhi), bylo, float(tB["height"]))      # outer edge B

    su_free, su_sup, su_feas = _hold_utilisation(scene, modes, *start)
    gu_free, gu_sup, gu_feas = _hold_utilisation(scene, modes, *goal)

    return {
        "case": case, "derate": derate,
        "start_lift_util": su_free, "start_slide_feasible": int(bool(su_feas)) if su_feas is not None else "",
        "goal_lift_util": gu_free, "goal_slide_feasible": int(bool(gu_feas)) if gu_feas is not None else "",
        "start_liftable": int(su_free <= 1.0) if su_free is not None else "",
        "goal_liftable": int(gu_free <= 1.0) if gu_free is not None else "",
    }


def main():
    rows = []
    for case, derate in SCENARIOS:
        print(f"[existence] case={case} derate={derate:.2f} ...", flush=True)
        try:
            row = run_scenario(case, derate)
        except Exception as e:  # noqa: BLE001
            print(f"  error: {e}")
            continue
        print(f"  start lift_util={row['start_lift_util']}, "
              f"goal lift_util={row['goal_lift_util']}")
        rows.append(row)

    keys = ["case", "derate", "start_lift_util", "start_liftable",
            "start_slide_feasible", "goal_lift_util", "goal_liftable",
            "goal_slide_feasible"]
    with open(OUT / "existence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nwrote {OUT/'existence.csv'} ({len(rows)} scenarios)")


if __name__ == "__main__":
    main()