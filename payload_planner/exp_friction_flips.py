"""
exp_friction_flips.py -- does a richer contact model actually change any
feasibility outcome?

The thesis uses the conservative unilateral NORMAL-force support model. The
framework also implements a FORCE model (normal + Coulomb friction cone) and a
WRENCH model (full 6-D contact wrench with tipping/torsion limits). Mentioning
these richer models is only worth doing if switching to them CHANGES something
measurable -- specifically, if they flip on-surface configurations that the
normal model rejects into feasible ones (i.e. friction lets the arm slide loads
the normal model says it cannot).

This script samples on-surface configurations across the workspace at a chosen
derate, evaluates each under all three contact models, and counts how many the
normal model rejects but a richer model accepts. That flip count is the whole
decision:

  - if a meaningful fraction flips  -> friction widens the feasible set; report
    it as a short result (e.g. "friction enlarges the sliding-feasible region by
    X% at derate rho").
  - if essentially nothing flips    -> do NOT present friction as a result; keep
    the single honest sentence that the richer models exist and would only
    enlarge the feasible region, but were not evaluated.

Emits data/exp_friction_flips.csv (one row per sampled config, with the three
models' feasibility and peak utilisation) plus a printed summary.

Run per payload and/or derate:

    PAYLOAD_CASE=3p1 python -B payload_planner/exp_friction_flips.py \
        --derate 0.60 --mu 0.6 --n-samples 400
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from scene import Scene
import transport
from run_panda import CFG, TABLE_A, TABLE_B
from constraints import ContactSupportFeasibility
import payload_model

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "..", "data", "exp_friction_flips.csv")


def _sample_surface_configs(scene, n_samples, rng):
    """
    Project random on-surface placements (flat-carry on a plane) to get valid
    on-surface configurations, WITHOUT applying any support/torque inequality --
    we want configs on the plane regardless of feasibility, so the three models
    can then judge them.
    """
    configs = []
    surfaces = [TABLE_A, TABLE_B]
    tries = 0
    while len(configs) < n_samples and tries < 30 * n_samples:
        tries += 1
        table = surfaces[rng.integers(0, len(surfaces))]
        surf_obj = scene.surface(table["name"])
        (xlo, ylo), (xhi, yhi) = np.asarray(table["xy_bounds"], float)
        x = float(rng.uniform(xlo, xhi))
        y = float(rng.uniform(ylo, yhi))
        # place() projects onto the on-surface manifold (plane + flat carry);
        # we accept the projected q even if inequalities fail, since feasibility
        # is exactly what the three models are here to decide.
        q, ok = transport.place(scene, surf_obj, x, y, rng=rng, n_restarts=8)
        if ok and q is not None:
            configs.append((table["name"], x, y, q))
    return configs


def _model(scene, contact_model, mu, mu_torsion, base_radius, derate):
    return ContactSupportFeasibility.from_scene(
        scene,
        contact_model=contact_model,
        mu=mu,
        mu_torsion=mu_torsion,
        base_radius=base_radius,
        derate=derate,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derate", type=float, default=0.60)
    ap.add_argument("--mu", type=float, default=0.6,
                    help="Coulomb friction coefficient for force/wrench models")
    ap.add_argument("--mu-torsion", type=float, default=0.02)
    ap.add_argument("--base-radius", type=float, default=0.02)
    ap.add_argument("--n-samples", type=int, default=400)
    args = ap.parse_args()

    case = payload_model.normalise_case()
    scene = Scene(CFG)
    rng = np.random.default_rng(20260711)

    m_normal = _model(scene, "normal", 0.0, 0.0, 0.0, args.derate)
    m_force = _model(scene, "force", args.mu, 0.0, args.base_radius, args.derate)
    m_wrench = _model(scene, "wrench", args.mu, args.mu_torsion, args.base_radius, args.derate)

    configs = _sample_surface_configs(scene, args.n_samples, rng)
    if not configs:
        print(f"[{case}] no on-surface configs sampled; check the scene")
        return

    rows = []
    n_normal_feas = n_force_feas = n_wrench_feas = 0
    n_flip_force = n_flip_wrench = 0
    for (surf, x, y, q) in configs:
        rn = m_normal.solve(q)
        rf = m_force.solve(q)
        rw = m_wrench.solve(q)
        fn, ff, fw = rn["feasible"], rf["feasible"], rw["feasible"]
        n_normal_feas += int(fn)
        n_force_feas += int(ff)
        n_wrench_feas += int(fw)
        if (not fn) and ff:
            n_flip_force += 1
        if (not fn) and fw:
            n_flip_wrench += 1
        rows.append({
            "case": case, "surface": surf,
            "x": round(x, 4), "y": round(y, 4),
            "normal_feasible": int(fn),
            "force_feasible": int(ff),
            "wrench_feasible": int(fw),
            "normal_peak_u": round(float(np.max(rn["utilisation"])), 4),
            "force_peak_u": round(float(np.max(rf["utilisation"])), 4),
            "wrench_peak_u": round(float(np.max(rw["utilisation"])), 4),
            "flip_normal_to_force": int((not fn) and ff),
            "flip_normal_to_wrench": int((not fn) and fw),
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(configs)
    print(f"\n[{case}] derate={args.derate}  mu={args.mu}  ({n} on-surface configs)")
    print(f"  normal feasible : {n_normal_feas}/{n} ({100*n_normal_feas/n:0.1f}%)")
    print(f"  force  feasible : {n_force_feas}/{n} ({100*n_force_feas/n:0.1f}%)")
    print(f"  wrench feasible : {n_wrench_feas}/{n} ({100*n_wrench_feas/n:0.1f}%)")
    print(f"  FLIPS normal->force  (infeasible->feasible): {n_flip_force}")
    print(f"  FLIPS normal->wrench (infeasible->feasible): {n_flip_wrench}")
    print(f"  wrote {OUT_CSV}")
    print()
    if n_flip_force == 0 and n_flip_wrench == 0:
        print("  DECISION: no config flips -> do NOT present friction as a "
              "result. Keep the one-sentence mention only.")
    else:
        frac = 100 * max(n_flip_force, n_flip_wrench) / n
        print(f"  DECISION: {frac:0.1f}% of configs flip feasible under a richer "
              f"model -> worth a short result section.")


if __name__ == "__main__":
    main()