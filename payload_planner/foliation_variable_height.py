#!/usr/bin/env python3
"""
foliation_variable_height.py
============================
PROTOTYPE (opt-in, standalone) for the "true optimisation" transfer carry.

Motivation (mentor's critique)
------------------------------
The existing transfer_foliation.py picks ONE constant carry height h* (the best
leaf of a horizontal foliation) and carries the object flat across the whole gap
at that fixed height.  Two objections:
  1. a single fixed height is not practical -- the object only needs to be high
     OVER the obstacle, not for the whole crossing;
  2. picking the lowest point of a sampled curve is a selection, not a true
     optimisation.

This prototype addresses both by:
  (A) letting the carry height VARY along the path -- a height PROFILE h(s) that
      rises to clear the wall and descends afterwards, so the object is never
      higher than it needs to be; and
  (C) choosing that profile by a genuine continuous OPTIMISATION (scipy) that
      minimises the peak torque utilisation along the whole carry, subject to
      clearing the wall and connecting q_lift -> q_set.

It is fully standalone: it imports transfer_foliation and transport but does not
modify them, and nothing else imports this file.  Run it to COMPARE against the
fixed-height foliation on the same task, so we can see whether the variable
profile actually lowers peak torque before writing any of it up.

The profile is parameterised by three control heights:
    h_in   : carry height just after lift-off  (approach)
    h_peak : carry height over the wall         (clearance)
    h_out  : carry height just before set-down  (departure)
The cross is planned as rise(h_in) -> traverse(h_peak) -> settle(h_out), each a
short constant-height leaf, stitched together.  Optimising (h_in,h_peak,h_out)
lets the height track the terrain: low where it can be, high only over the wall.

Usage
-----
    import foliation_variable_height as fvh
    res = fvh.optimise_carry(scene, q_lift, q_set)      # returns profile + info

or from run_panda-style context, this file has a __main__ demo that builds the
scene from run_panda.CFG, discovers a lift/set pair, and prints a side-by-side
comparison of:  fixed-height foliation  vs  variable-height optimisation.
"""

import os
import sys
import numpy as np

import transport as T
import transfer_foliation as fol
from constraints import Mode, PlaneConstraint, TorqueFeasibility


# --------------------------------------------------------------------------- #
# Torque scoring shared with the fixed-height version, so comparisons are fair.
# --------------------------------------------------------------------------- #
def _peak_util_along(scene, path):
    """Max torque utilisation over a joint path (same metric as evaluate_leaf)."""
    torque = TorqueFeasibility(scene.robot, scene.tau_max, derate=scene.derate)
    peak = 0.0
    for q in np.asarray(path, float):
        try:
            peak = max(peak, float(np.max(torque.utilisation(q))))
        except Exception:
            continue
    return peak


def _joint_len(path):
    p = np.asarray(path, float)
    if p is None or len(p) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


# --------------------------------------------------------------------------- #
# Plan a carry that follows a 3-point height profile (h_in, h_peak, h_out).
# --------------------------------------------------------------------------- #
def evaluate_profile(scene, q_lift, q_set, h_in, h_peak, h_out,
                     planner="RRTConnect", solve_time=15.0, delta=0.08,
                     space="atlas"):
    """
    Plan lift-off -> [rise to h_in] -> [traverse near h_peak] -> [settle to
    h_out] -> set-down, letting the carry height vary through the three control
    heights.  Returns (ok, info) where info carries the peak torque utilisation
    and the stitched path.  Never raises.

    Implementation reuses transfer_foliation._project_to_height and
    transport._plan_segment so the physics/feasibility is identical to the
    fixed-height version -- only the height is allowed to change along the way.
    """
    info = {"h_in": h_in, "h_peak": h_peak, "h_out": h_out, "reason": ""}
    free_mode = T.free_flight_mode(scene)

    # Project the lift/set endpoints onto their respective approach/departure
    # heights, and the peak height for the middle traverse.
    q_in, leaf_in = fol._project_to_height(scene, q_lift, h_in, planner,
                                           solve_time, delta, space)
    if q_in is None:
        info["reason"] = "lift will not project to h_in"
        return False, info
    q_pk_a, leaf_pk = fol._project_to_height(scene, q_in, h_peak, planner,
                                             solve_time, delta, space)
    if q_pk_a is None:
        info["reason"] = "cannot project to h_peak (in)"
        return False, info
    q_pk_b, _ = fol._project_to_height(scene, q_set, h_peak, planner,
                                       solve_time, delta, space)
    if q_pk_b is None:
        info["reason"] = "cannot project to h_peak (out)"
        return False, info
    q_out, leaf_out = fol._project_to_height(scene, q_set, h_out, planner,
                                             solve_time, delta, space)
    if q_out is None:
        info["reason"] = "set will not project to h_out"
        return False, info

    # Segment chain:
    #   q_lift --(free)--> q_in            rise from table to approach height
    #   q_in   --(leaf_in @ h_in)--> q_pk_a  (optional short level move)
    #   q_pk_a --(leaf_pk @ h_peak)--> q_pk_b traverse over the wall at peak h
    #   q_pk_b --(leaf_out @ h_out)--> q_out  settle to departure height
    #   q_out  --(free)--> q_set            descend to table
    chain = [
        (free_mode, q_lift, q_in, "rise"),
        (leaf_in,   q_in,   q_pk_a, "approach_level"),
        (leaf_pk,   q_pk_a, q_pk_b, "traverse_peak"),
        (leaf_out,  q_pk_b, q_out, "depart_level"),
        (free_mode, q_out,  q_set, "descend"),
    ]
    stitched = []
    peak_util = 0.0
    total_len = 0.0
    for mode, qa, qb, tag in chain:
        seg = T._plan_segment(scene.robot, mode, qa, qb, planner,
                              solve_time, delta, space)
        if not seg.success:
            info["reason"] = f"{tag} failed: {seg.message}"
            return False, info
        path = np.asarray(seg.path, float)
        # score torque only on the parts where the object is actually carried
        # high (the leaf segments), matching the fixed-height metric which
        # scores the cross.  Rise/descend are transitions.
        if tag in ("approach_level", "traverse_peak", "depart_level"):
            peak_util = max(peak_util, _peak_util_along(scene, path))
        total_len += _joint_len(path)
        stitched.append((tag, path))

    info.update({"util_peak": peak_util, "path_len": total_len,
                 "segments": stitched})
    return True, info


# --------------------------------------------------------------------------- #
# The optimisation (C): minimise peak torque over the 3-point height profile.
# --------------------------------------------------------------------------- #
def optimise_carry(scene, q_lift, q_set,
                   planner="RRTConnect", solve_time=15.0, delta=0.08,
                   space="atlas", verbose=True):
    """
    True continuous optimisation of the carry height profile.  Minimises peak
    torque utilisation (with a light path-length penalty) over (h_in,h_peak,
    h_out) using scipy's derivative-free Nelder-Mead, starting from the flat
    profile at the fixed-height optimum so it can only improve on it.

    Returns (best_profile, best_info) or (None, None) if nothing feasible.
    """
    try:
        from scipy.optimize import minimize
    except Exception:
        print("scipy not available; cannot run the optimiser. "
              "pip install scipy")
        return None, None

    h_min, h_max = fol._height_bounds(scene)
    w_path = float(os.environ.get("PAYLOAD_W_PATH", "0.1"))
    w_torque = float(os.environ.get("PAYLOAD_W_TORQUE", "10.0"))

    # Warm start: the fixed-height optimum (so the optimiser starts from what the
    # current method already achieves and can only do better).
    h0, info0 = fol._select_continuous(
        scene, q_lift, q_set, planner, solve_time, delta, space,
        w_torque, w_path, False)
    if h0 is None:
        h0 = 0.5 * (h_min + h_max)
    x0 = np.array([h0, h0, h0], dtype=float)   # start flat at the fixed optimum

    cache = {}

    def cost(x):
        key = tuple(round(float(v), 3) for v in x)
        if key in cache:
            return cache[key]
        h_in, h_peak, h_out = [float(np.clip(v, h_min, h_max)) for v in x]
        # peak must be the highest (it clears the wall); gently enforce ordering
        h_peak = max(h_peak, h_in, h_out)
        ok, info = evaluate_profile(scene, q_lift, q_set, h_in, h_peak, h_out,
                                    planner, solve_time, delta, space)
        if not ok:
            c = 1e3   # infeasible penalty
        else:
            c = w_torque * info["util_peak"] + w_path * info["path_len"]
        cache[key] = (c, ok, info)
        if verbose:
            tag = "OK  " if ok else "FAIL"
            up = info.get("util_peak", float("nan"))
            print(f"    profile h=({h_in:.3f},{h_peak:.3f},{h_out:.3f}) "
                  f"{tag} util_peak={up:.3f} cost={c:.3f}")
        return (c, ok, info)

    def scalar_cost(x):
        return cost(x)[0]

    if verbose:
        print(f"[variable-height] warm start flat h0={h0:.3f}, "
              f"bounds=[{h_min:.3f},{h_max:.3f}]")
    res = minimize(scalar_cost, x0, method="Nelder-Mead",
                   options={"xatol": 0.01, "fatol": 0.05, "maxiter": 60})

    # pick the best feasible profile actually evaluated
    best = None
    for key, (c, ok, info) in cache.items():
        if ok and (best is None or c < best[0]):
            best = (c, key, info)
    if best is None:
        if verbose:
            print("  no feasible variable-height profile found")
        return None, None
    _, key, info = best
    if verbose:
        print(f"  BEST variable profile h=({info['h_in']:.3f},"
              f"{info['h_peak']:.3f},{info['h_out']:.3f}) "
              f"util_peak={info['util_peak']:.3f}")
    return (info["h_in"], info["h_peak"], info["h_out"]), info


# --------------------------------------------------------------------------- #
# Comparison demo: fixed-height foliation  vs  variable-height optimisation.
# --------------------------------------------------------------------------- #
def _demo():
    import run_panda
    from scene import Scene
    scene = Scene(run_panda.CFG)

    sa, sb = scene.surface("A"), scene.surface("B")
    rng = np.random.default_rng(20260612)
    lifts = T.sample_transitions(scene, sa, 300, other_surface=sb, rng=rng,
                                 support_margin=0.05)
    sets = T.sample_transitions(scene, sb, 300, other_surface=sa, rng=rng,
                                support_margin=0.05)
    if not lifts or not sets:
        raise SystemExit("no transitions at this case/derate")
    q_lift = sorted(lifts, key=lambda q: float(np.linalg.norm(q)))[0]
    q_set = sorted(sets, key=lambda q: float(np.linalg.norm(q)))[0]

    st = float(os.environ.get("PAYLOAD_SOLVE_TIME", "15"))
    delta = float(os.environ.get("PAYLOAD_DELTA", "0.08"))
    space = os.environ.get("PAYLOAD_CONSTRAINED_SPACE", "atlas")

    print("=" * 70)
    print("FIXED-HEIGHT foliation (current method)")
    print("=" * 70)
    h_fixed, info_fixed = fol._select_continuous(
        scene, q_lift, q_set, "RRTConnect", st, delta, space, 10.0, 0.1, True)
    if h_fixed is not None:
        print(f"  fixed h*={h_fixed:.3f}  util_peak={info_fixed['util_peak']:.3f}")

    print("=" * 70)
    print("VARIABLE-HEIGHT optimisation (prototype A+C)")
    print("=" * 70)
    prof, info_var = optimise_carry(scene, q_lift, q_set,
                                    solve_time=st, delta=delta, space=space,
                                    verbose=True)

    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    if h_fixed is not None:
        print(f"  fixed-height   : h={h_fixed:.3f}, "
              f"peak util={info_fixed['util_peak']:.3f}")
    if prof is not None:
        print(f"  variable-height: h=({prof[0]:.3f},{prof[1]:.3f},{prof[2]:.3f}), "
              f"peak util={info_var['util_peak']:.3f}")
    if h_fixed is not None and prof is not None:
        drop = info_fixed["util_peak"] - info_var["util_peak"]
        print(f"  peak-utilisation change: {drop:+.3f} "
              f"({'better' if drop > 0 else 'no improvement'})")


if __name__ == "__main__":
    _demo()