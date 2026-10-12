#!/usr/bin/env python3
"""
sweep_nullspace.py -- does null-space posture optimisation help MORE when torque
is scarce?  Sweeps payload x derate, and for each runs the (proven, zero-drift)
null-space optimiser on the foliation carry, recording peak torque before/after.

The question: at a tight budget (low derate, heavy payload) the arm sits near its
limit -- does reconfiguring the redundant posture buy more margin there than when
there is slack?  If the reduction grows under stress, that is the headline.

Writes experiments/nullspace_sweep.csv:
  case, derate, peak_before, peak_after, reduction, max_drift_mm, feasible_after

Run from payload_planner/:
  python3 sweep_nullspace.py
"""
import os, sys, csv
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "experiments"; OUT.mkdir(exist_ok=True)

import transport as T
import transfer_foliation as fol
import nullspace_torque_opt as ns
from scene import Scene

CASES = ["2p1", "3p1", "4p1"]
# derates chosen near each payload's must-slide regime (tight = more stress)
DERATES = [0.45, 0.50, 0.55, 0.60]

def build(case, derate):
    os.environ["PAYLOAD_CASE"] = case
    os.environ["PAYLOAD_TORQUE_DERATE"] = f"{derate:.2f}"
    os.environ["PAYLOAD_WALL"] = "1"
    os.environ["PAYLOAD_WALL_TOP"] = "0.15"
    os.environ["PAYLOAD_CONSTRAINED_SPACE"] = "atlas"
    import importlib, run_panda
    importlib.reload(run_panda)
    return Scene(run_panda.CFG)

def main():
    st = float(os.environ.get("PAYLOAD_SOLVE_TIME", "15"))
    delta = 0.08; space = "atlas"
    rows = []
    for case in CASES:
        for derate in DERATES:
            try:
                scene = build(case, derate)
                sa, sb = scene.surface("A"), scene.surface("B")
                rng = np.random.default_rng(20260612)
                lifts = T.sample_transitions(scene, sa, 300, other_surface=sb, rng=rng, support_margin=0.05)
                sets = T.sample_transitions(scene, sb, 300, other_surface=sa, rng=rng, support_margin=0.05)
                if not lifts or not sets:
                    print(f"[{case} {derate}] no transitions"); continue
                q_lift = sorted(lifts, key=lambda q: float(np.linalg.norm(q)))[0]
                q_set  = sorted(sets,  key=lambda q: float(np.linalg.norm(q)))[0]
                h, info = fol._select_continuous(scene, q_lift, q_set, "RRTConnect", st, delta, space, 10.0, 0.1, False)
                if h is None:
                    print(f"[{case} {derate}] foliation failed"); continue
                carry = np.vstack([np.asarray(p, float) for _, p in info.get("segments", [])])
                _, before, after, drift = ns.optimise_path(scene, carry, verbose=False)
                red = before - after
                print(f"[{case} d={derate:.2f}] before={before:.3f} after={after:.3f} "
                      f"reduction={red:+.3f} drift={drift*1000:.2f}mm")
                rows.append({"case": case, "derate": derate,
                             "peak_before": round(before,4), "peak_after": round(after,4),
                             "reduction": round(red,4), "max_drift_mm": round(drift*1000,3),
                             "feasible_after": int(after <= 1.0)})
            except Exception as e:
                print(f"[{case} {derate}] ERROR {e!r}")
    if rows:
        with open(OUT/"nullspace_sweep.csv","w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {OUT/'nullspace_sweep.csv'} ({len(rows)} cells)")

if __name__ == "__main__":
    main()w