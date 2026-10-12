#!/usr/bin/env python3
"""
analyze_contact_wrench.py
=========================

Walks the *executed* trajectory and, at each supported configuration, solves the
full 6-D contact-wrench load-sharing problem

        min ||W||          s.t.   |tau_arm| = |g(q) - Jc(q)^T W| <= tau_budget
         W                        W in the contact-wrench cone

and reports how hard each physical part of the ground contact is working:

  * f_normal   - the upward push from the table        (the scalar lambda)
  * f_tangent  - the in-plane friction force           (resists sliding)
  * m_tip      - the tipping moment about a flat axis   (wants to topple the can)
  * m_twist    - the moment about the vertical          (wants to spin the can)
  * arm_util   - max_j |tau_j| / tau_max_j              (how close the arm is to
                 its derated limit -- the number the mentor wants near 1.0)

and the *usage* of each contact limit, as a fraction of what the contact can
physically supply:

  * slip_use  = |f_tangent| / (mu      * f_normal)   > 1  ->  the can slides
  * tip_use   = |m_tip|     / (r       * f_normal)   > 1  ->  the can tips over
  * twist_use = |m_twist|   / (mu_tors * f_normal)   > 1  ->  the can spins

This is read-only: it does NOT touch the plan or the exported trajectory.  It is
the picture of *why* a min-||lambda|| slide is (or is not) physically safe -- the
left-middle drawing on the whiteboard (the cylinder with the up-arrow and the
rotation curl) made quantitative.

Run:
    python analyze_contact_wrench.py
    python analyze_contact_wrench.py --plan panda_plan.npz --csv system_traj.csv \
        --stride 30 --mu 0.6 --base-radius 0.02 --mu-torsion 0.01

Needs pinocchio (same env as export_trajectory.py) + scipy.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _read_exec_csv(csv_path: Path):
    """Return (t, q) from traj_exec.csv.  Columns: time_s, q1..qN, ..."""
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r]
    idx = {name: i for i, name in enumerate(header)}
    q_cols = sorted(
        (c for c in header if c.startswith("q") and c[1:].isdigit()),
        key=lambda c: int(c[1:]),
    )
    if not q_cols:
        raise SystemExit(f"no q1..qN columns found in {csv_path} (header: {header})")
    t_i = idx.get("time_s")
    t, q = [], []
    for r in rows:
        q.append([float(r[idx[c]]) for c in q_cols])
        t.append(float(r[t_i]) if t_i is not None else float(len(t)))
    return np.asarray(t, dtype=float), np.asarray(q, dtype=float)


def _decompose(contact, normal):
    """Split a 6-vector wrench W=[f(3), m(3)] into the physical pieces."""
    W = np.asarray(contact, dtype=float).reshape(-1)
    n = np.asarray(normal, dtype=float).reshape(-1)
    n = n / (np.linalg.norm(n) + 1e-15)
    f = W[0:3] if W.size >= 3 else np.zeros(3)
    m = W[3:6] if W.size >= 6 else np.zeros(3)
    fn = float(f @ n)                      # normal force (the scalar lambda)
    f_tan = f - fn * n                     # tangential (friction) force
    mn = float(m @ n)                      # moment about normal  -> TWIST
    m_perp = m - mn * n                    # moment in the plane  -> TIP-OVER
    return {
        "f_normal": fn,
        "f_tangent": float(np.linalg.norm(f_tan)),
        "m_tip": float(np.linalg.norm(m_perp)),
        "m_twist": float(abs(mn)),
    }


def _usage(parts, mu, r, mu_tors):
    """Fraction of each contact limit used.  >1 means that mode fails."""
    fn = max(parts["f_normal"], 0.0)
    eps = 1e-9
    slip = parts["f_tangent"] / (mu * fn + eps) if mu > 0 else 0.0
    tip = parts["m_tip"] / (r * fn + eps) if r > 0 else 0.0
    twist = parts["m_twist"] / (mu_tors * fn + eps) if mu_tors > 0 else 0.0
    return slip, tip, twist


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=str(HERE / "panda_plan.npz"))
    ap.add_argument("--csv", default=str(HERE / "system_traj.csv"))
    ap.add_argument("--stride", type=int, default=30,
                    help="use every Nth executed sample (default 30 ~= 100 pts)")
    ap.add_argument("--mu", type=float, default=0.6,
                    help="steel-on-steel-ish friction coefficient")
    ap.add_argument("--base-radius", type=float, default=0.02,
                    help="cylinder base radius [m] -> tipping moment arm")
    ap.add_argument("--mu-torsion", type=float, default=0.01,
                    help="torsional friction coefficient -> twist limit")
    ap.add_argument("--support-derate", type=float, default=1.0,
                    help="shrink the torque budget the share AIMS for (0.97 caps near 97%%)")
    args = ap.parse_args(argv)

    from export_trajectory import _build_scene, _supported_surface
    from constraints import ContactSupportFeasibility

    plan_path = Path(args.plan)
    csv_path = Path(args.csv)
    if not plan_path.exists():
        raise SystemExit(f"plan not found: {plan_path}")
    if not csv_path.exists():
        raise SystemExit(f"exec csv not found: {csv_path}")

    with np.load(plan_path, allow_pickle=True) as plan_data:
        scene = _build_scene(plan_data)

    # the rich rotational model, regardless of what the plan ran with
    feas = ContactSupportFeasibility.from_scene(
        scene,
        contact_model="wrench",
        mu=args.mu,
        base_radius=args.base_radius,
        mu_torsion=args.mu_torsion,
        support_derate=args.support_derate,
    )
    normal = getattr(feas, "normal_world", np.array([0.0, 0.0, 1.0]))

    t_all, q_all = _read_exec_csv(csv_path)
    sel = range(0, len(q_all), max(1, args.stride))

    print(f"# plan : {plan_path}")
    print(f"# csv  : {csv_path}   ({len(q_all)} samples, stride {args.stride})")
    print(f"# model: wrench   mu={args.mu}  r={args.base_radius}  "
          f"mu_tors={args.mu_torsion}  support_derate={args.support_derate}")
    print(f"# contact-wrench cone:  fz>=0,  |f_tan|<=mu*fz,  "
          f"|m_tip|<=r*fz,  |m_twist|<=mu_tors*fz")
    print()
    head = (f"{'t[s]':>7} {'phase':>9} {'arm_util':>8} {'f_norm':>8} "
            f"{'f_tan':>7} {'m_tip':>7} {'m_tw':>7} "
            f"{'slip':>6} {'tip':>6} {'twist':>6} {'ok':>4}")
    print(head)
    print("-" * len(head))

    worst = {"arm_util": 0.0, "slip": 0.0, "tip": 0.0, "twist": 0.0}
    worst_at = {"tip": None, "slip": None, "twist": None}
    n_supported = 0
    n_infeasible = 0

    for k in sel:
        q = q_all[k]
        t = float(t_all[k])
        surf = _supported_surface(scene, q)
        phase = "support" if surf is not None else "flight"
        if surf is None:
            # in free flight there is no ground contact to analyse
            print(f"{t:7.2f} {phase:>9} {'-':>8} {'-':>8} {'-':>7} "
                  f"{'-':>7} {'-':>7} {'-':>6} {'-':>6} {'-':>6} {'-':>4}")
            continue

        n_supported += 1
        res = feas.solve(q)
        if not res.get("feasible", False):
            n_infeasible += 1
            print(f"{t:7.2f} {phase:>9} {'INFEAS':>8} {'-':>8} {'-':>7} "
                  f"{'-':>7} {'-':>7} {'-':>6} {'-':>6} {'-':>6} "
                  f"{res.get('reason','?')[:4]:>4}")
            continue

        arm_util = float(np.max(res["utilisation"]))
        parts = _decompose(res["contact"], normal)
        slip, tip, twist = _usage(parts, args.mu, args.base_radius, args.mu_torsion)

        worst["arm_util"] = max(worst["arm_util"], arm_util)
        for key, val in (("slip", slip), ("tip", tip), ("twist", twist)):
            if val > worst[key]:
                worst[key] = val
                worst_at[key] = t

        ok = "ok" if max(slip, tip, twist) <= 1.0 + 1e-6 else "FAIL"
        print(f"{t:7.2f} {phase:>9} {arm_util:8.4f} {parts['f_normal']:8.2f} "
              f"{parts['f_tangent']:7.2f} {parts['m_tip']:7.3f} {parts['m_twist']:7.3f} "
              f"{slip:6.2f} {tip:6.2f} {twist:6.2f} {ok:>4}")

    print()
    print("# summary")
    print(f"#   supported samples analysed : {n_supported}")
    print(f"#   infeasible (torque) samples: {n_infeasible}")
    print(f"#   worst arm utilisation      : {worst['arm_util']:.4f}"
          f"   (mentor wants this near 1.0 on the slide)")
    print(f"#   worst slip usage           : {worst['slip']:.3f}"
          f"   at t={worst_at['slip']}")
    print(f"#   worst tip usage            : {worst['tip']:.3f}"
          f"   at t={worst_at['tip']}"
          + ("   <-- CYLINDER TIPS" if worst["tip"] > 1.0 else ""))
    print(f"#   worst twist usage          : {worst['twist']:.3f}"
          f"   at t={worst_at['twist']}")
    if max(worst["slip"], worst["tip"], worst["twist"]) <= 1.0 + 1e-6:
        print("#   => every supported sample stays inside the contact-wrench cone:")
        print("#      the can neither slides, tips, nor spins while the arm slides it.")
    else:
        print("#   => at least one sample leaves the cone; the min-||W|| share is not")
        print("#      physically realisable there (object would move on the table).")


if __name__ == "__main__":
    main()