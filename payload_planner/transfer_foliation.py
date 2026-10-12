"""
transfer_foliation.py
=====================
OPT-IN, ADDITIVE transfer-height foliation for the payload transport planner.

Nothing in the existing pipeline imports this module, so leaving it on disk
changes NOTHING about how run_panda.py / transport.py currently behave.  You
only get the foliation behaviour if you explicitly call `plan_with_foliation`
(or set PAYLOAD_FOLIATION=1 and route through the thin wrapper at the bottom).

Idea
----
The free-carry phase is modeled as a foliation of transfer manifolds: a
continuous family of horizontal carry planes z = h.  Each leaf is the existing
free-flight mode with one extra equality pinning the payload to height h
(PlaneConstraint).  Different leaves have genuinely different torque feasibility
(g(q) changes with arm posture, which changes with carry height) and different
obstacle clearance (the leaf must clear the gap wall).  We do a coarse search
over candidate heights, then a local refinement around the best one, scoring
each feasible leaf by torque margin and path length.

This reuses transport.py's own primitives (free_flight_mode, _plan_segment,
PlaneConstraint, the torque/collision constraints).  It invents no new physics.

Usage
-----
    import transfer_foliation as fol
    res = fol.plan_with_foliation(scene, start, goal, **kwargs)

or, height candidates from the environment:
    PAYLOAD_TRANSFER_HEIGHTS="0.20,0.30,0.40,0.50"   # coarse candidate set
    PAYLOAD_TRANSFER_REFINE="1"                       # do local refine
"""

import os
import numpy as np

import transport as T
from constraints import Mode, PlaneConstraint, FlatCarryConstraint, FrameConstraint
from constraints import JointLimit, TorqueFeasibility


def _Rz(psi):
    """Rotation about world z by psi (radians)."""
    c, s = np.cos(float(psi)), np.sin(float(psi))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


# R_FLAT must match transport.R_FLAT (gripper-down flat carry).
_R_FLAT = np.diag([1.0, -1.0, -1.0])


def transfer_leaf_2d(scene, h, psi):
    """
    A 2-D foliation leaf parameterised by (height h, carry yaw psi).

    Same feasibility as free-flight (torque + collision + floor), plus:
      * PlaneConstraint  z = h          (the height leaf coordinate)
      * yaw-pinned flat carry R_FLAT@Rz(psi)   (the orientation leaf coordinate)

    Because the payload COM is off-centre, changing psi changes the gravity
    moment and therefore g(q) and the torque utilisation -- so psi is a genuine
    second optimisation dimension, not a cosmetic one.
    """
    base = T.free_flight_mode(scene)
    r = scene.robot
    frame = scene.controlled_frame
    # Replace the yaw-free flat carry with a yaw-PINNED carry at psi, and add the
    # height plane.  We rebuild equalities rather than appending so we don't keep
    # the original yaw-free flat-carry equality alongside the pinned one.
    R_target = _R_FLAT @ _Rz(psi)
    yaw_pinned_carry = FrameConstraint(
        r, frame,
        target_rotation=R_target, rot_mask=(1, 1, 1), rot_tol=0.0,
        name=f"carry_yaw{float(psi):+.2f}",
    )
    equalities = [PlaneConstraint(r, frame, float(h)), yaw_pinned_carry]
    return Mode(
        f"transfer_h{float(h):.3f}_psi{float(psi):+.2f}",
        equalities=equalities,
        inequalities=list(base.inequalities),
    )


# --------------------------------------------------------------------------- #
# Leaf construction: free-flight mode pinned to a constant carry height h.
# --------------------------------------------------------------------------- #
def transfer_plane_mode(scene, h):
    """
    One leaf of the transfer foliation: the existing free-flight feasibility
    (flat carry + torque + collision) PLUS an equality pinning the payload frame
    to the horizontal plane z = h.  Reuses transport.free_flight_mode's pieces
    so any change there (e.g. the payload-floor inequality) is inherited.
    """
    base = T.free_flight_mode(scene)            # inherits torque/collision/floor
    r = scene.robot
    frame = scene.controlled_frame
    # Add the plane equality on top of whatever equalities free_flight already has.
    equalities = list(base.equalities) + [PlaneConstraint(r, frame, float(h))]
    return Mode(
        f"transfer_h{float(h):.3f}",
        equalities=equalities,
        inequalities=list(base.inequalities),
    )


# --------------------------------------------------------------------------- #
# Candidate height set.
# --------------------------------------------------------------------------- #
def candidate_heights(scene, explicit=None, n=5):
    """
    Build a coarse list of carry heights to try.  Defaults span from just above
    the highest table to a comfortable reach height; if a gap wall is present in
    the scene config the lowest candidate is pushed above the wall top so every
    leaf at least has a chance of clearing it.
    """
    if explicit is not None:
        return [float(x) for x in explicit]

    env = os.environ.get("PAYLOAD_TRANSFER_HEIGHTS", "").strip()
    if env:
        return [float(x) for x in env.replace(",", " ").split()]

    table_top = max(float(s.height) for s in scene.surfaces)
    low = table_top + 0.10
    # If the scene carries a wall height, start above it.
    wall_top = float(scene.config.get("wall_top", 0.0) or 0.0)
    if wall_top > 0.0:
        low = max(low, wall_top + 0.05)
    high = table_top + 0.55
    return list(np.linspace(low, high, int(max(2, n))))


# --------------------------------------------------------------------------- #
# Score a single leaf: plan the three sub-segments through height h and score.
# --------------------------------------------------------------------------- #
def _project_to_height(scene, q_seed, h, planner, solve_time, delta, space):
    """
    Project q_seed onto the transfer plane at height h using the leaf mode's own
    projection, and return (q_on_leaf, leaf).  Returns (None, leaf) if the seed
    cannot be placed feasibly on the leaf.
    """
    leaf = transfer_plane_mode(scene, h)
    try:
        out = leaf.project(np.asarray(q_seed, float))
    except Exception:
        return None, leaf
    # Mode.project may return q, or (q, converged).  Normalise either form.
    if isinstance(out, tuple):
        q_on = out[0]
        converged = bool(out[1]) if len(out) > 1 else True
    else:
        q_on = out
        converged = True
    if q_on is None or not converged:
        return None, leaf
    q_on = np.asarray(q_on, float).reshape(-1)
    if q_on.shape[0] != int(scene.robot.nq):
        return None, leaf
    if not bool(leaf.feasible(q_on)):
        return None, leaf
    return q_on, leaf


def evaluate_leaf(scene, q_lift, q_set, h,
                  planner="RRTConnect", solve_time=15.0, delta=0.08,
                  space="atlas",
                  w_torque=10.0, w_path=1.0):
    """
    Try to route lift_off -> (transfer at height h) -> set_down through one leaf.
    Plans: q_lift -> q_h_in, the transfer across on the leaf, q_h_out -> q_set.
    Returns (ok, score, info).  Lower score is better.  Never raises.
    """
    info = {"h": float(h), "reason": ""}

    # The cross stays ON the leaf (plane z=h).  The lift->leaf and leaf->set
    # transitions are planned in the ORDINARY free-flight mode (no z pinning) so
    # they can rise from the lift-off config (at the table, z~0) up to the plane
    # and descend again -- and crucially so the stitched free-flight path STARTS
    # exactly at q_lift and ENDS exactly at q_set, matching the slide segments.
    free_mode = T.free_flight_mode(scene)
    q_h_in, leaf = _project_to_height(scene, q_lift, h, planner, solve_time, delta, space)
    if q_h_in is None:
        info["reason"] = "lift seed will not project to leaf"
        return False, float("inf"), info
    q_h_out, _ = _project_to_height(scene, q_set, h, planner, solve_time, delta, space)
    if q_h_out is None:
        info["reason"] = "set seed will not project to leaf"
        return False, float("inf"), info

    # lift->leaf in free mode (rise from table to plane), starting at the REAL q_lift.
    seg_lift = T._plan_segment(scene.robot, free_mode, q_lift, q_h_in,
                               planner, solve_time, delta, space)
    if not seg_lift.success:
        info["reason"] = f"lift->leaf failed: {seg_lift.message}"
        return False, float("inf"), info
    # cross ON the leaf (constant height h).
    seg_cross = T._plan_segment(scene.robot, leaf, q_h_in, q_h_out,
                                planner, solve_time, delta, space)
    if not seg_cross.success:
        info["reason"] = f"cross-on-leaf failed: {seg_cross.message}"
        return False, float("inf"), info
    # leaf->set in free mode (descend from plane to table), ending at the REAL q_set.
    seg_drop = T._plan_segment(scene.robot, free_mode, q_h_out, q_set,
                               planner, solve_time, delta, space)
    if not seg_drop.success:
        info["reason"] = f"leaf->set failed: {seg_drop.message}"
        return False, float("inf"), info

    # Path-length cost (joint space) across the three sub-segments.
    def _len(seg):
        p = np.asarray(seg.path, float)
        if p is None or len(p) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    path_len = _len(seg_lift) + _len(seg_cross) + _len(seg_drop)

    # Torque cost: TRUE max utilisation |g(q)|/(derate*tau_max) along the carry.
    # This is a continuous 0..1+ number that genuinely varies between leaves
    # (higher/extended carry postures load the arm differently), unlike the
    # feasibility residual which clamps to <=0 once feasible.  Lower = more
    # torque headroom = safer leaf, which is what we want to prefer.
    torque = TorqueFeasibility(scene.robot, scene.tau_max, derate=scene.derate)
    util_peak = 0.0
    util_sum = 0.0
    n = 0
    for q in np.asarray(seg_cross.path, float):
        try:
            u = float(np.max(torque.utilisation(q)))
        except Exception:
            continue
        util_peak = max(util_peak, u)
        util_sum += u
        n += 1
    util_mean = (util_sum / n) if n else 0.0

    # Torque-dominant score.  Weights tunable via env; defaults make torque the
    # headline and path length a mild tiebreaker.  We score peak utilisation
    # (worst-case load on the carry) plus a small mean term, and add a light
    # path-length penalty.
    w_torque = float(os.environ.get("PAYLOAD_W_TORQUE", str(w_torque)))
    w_path = float(os.environ.get("PAYLOAD_W_PATH", str(w_path)))
    w_torque_mean = float(os.environ.get("PAYLOAD_W_TORQUE_MEAN", "0.25"))

    score = (w_torque * util_peak
             + w_torque_mean * util_mean
             + w_path * path_len)
    info.update({
        "path_len": path_len,
        "util_peak": util_peak,
        "util_mean": util_mean,
        "w_torque": w_torque,
        "w_path": w_path,
        "segments": [("lift_to_leaf", seg_lift.path),
                     ("cross", seg_cross.path),
                     ("leaf_to_set", seg_drop.path)],
    })
    return True, score, info


# --------------------------------------------------------------------------- #
# Coarse-to-fine height selection.
# --------------------------------------------------------------------------- #
def select_transfer_height(scene, q_lift, q_set,
                           heights=None, refine=None,
                           planner="RRTConnect", solve_time=15.0, delta=0.08,
                           space="atlas", w_torque=10.0, w_path=1.0,
                           verbose=True):
    """
    Select a transfer height for this (q_lift, q_set) pair.

    Two modes:
      * CONTINUOUS (default, or PAYLOAD_TRANSFER_CONTINUOUS=1): optimise the
        carry height as a real scalar over the interval
        [h_min, h_max] = [obstacle_top + clearance, reach_max] using a
        golden-section search.  This is the honest continuous foliation: the
        leaf parameter ranges over a continuum and we pick the exact best h*.
      * DISCRETE (PAYLOAD_TRANSFER_CONTINUOUS=0): grid search over
        PAYLOAD_TRANSFER_HEIGHTS, optional local refine.  Kept for comparison.

    Returns (best_h, best_info) or (None, None).
    """
    continuous = os.environ.get("PAYLOAD_TRANSFER_CONTINUOUS", "1").lower() in {"1", "true", "yes", "on"}
    two_d = os.environ.get("PAYLOAD_TRANSFER_2D", "0").lower() in {"1", "true", "yes", "on"}
    if two_d and heights is None:
        h, psi, info = select_transfer_2d(scene, q_lift, q_set, planner=planner,
                                          solve_time=solve_time, delta=delta,
                                          space=space, w_torque=w_torque,
                                          w_path=w_path, verbose=verbose)
        if info is not None:
            info["psi"] = psi
        return h, info
    if continuous and heights is None:
        return _select_continuous(scene, q_lift, q_set, planner, solve_time,
                                  delta, space, w_torque, w_path, verbose)
    return _select_discrete(scene, q_lift, q_set, heights, refine, planner,
                            solve_time, delta, space, w_torque, w_path, verbose)


def _height_bounds(scene):
    """Continuous interval [h_min, h_max] for the transfer height."""
    table_top = max(float(s.height) for s in scene.surfaces)
    # lower bound: above the obstacle (wall) plus clearance, else just above table
    wall_top = float(scene.config.get("wall_top", 0.0) or 0.0)
    clearance = float(os.environ.get("PAYLOAD_TRANSFER_CLEARANCE", "0.05"))
    h_min = max(table_top + 0.10, wall_top + clearance)
    h_max = table_top + float(os.environ.get("PAYLOAD_TRANSFER_H_MAX_ABOVE", "0.55"))
    # env overrides if the user wants explicit bounds
    h_min = float(os.environ.get("PAYLOAD_TRANSFER_H_MIN", h_min))
    h_max = float(os.environ.get("PAYLOAD_TRANSFER_H_MAX", h_max))
    if h_max <= h_min:
        h_max = h_min + 0.20
    return h_min, h_max


def _select_continuous(scene, q_lift, q_set, planner, solve_time, delta, space,
                       w_torque, w_path, verbose):
    """Golden-section search for the torque-optimal carry height h*."""
    import math
    h_min, h_max = _height_bounds(scene)
    if verbose:
        print(f"  continuous foliation: searching h in [{h_min:.3f}, {h_max:.3f}]")

    cache = {}

    def cost_of(h):
        key = round(float(h), 4)
        if key in cache:
            return cache[key]
        ok, score, info = evaluate_leaf(scene, q_lift, q_set, h,
                                        planner=planner, solve_time=solve_time,
                                        delta=delta, space=space,
                                        w_torque=w_torque, w_path=w_path)
        # infeasible heights get a large but finite penalty so the optimiser
        # is pushed back toward the feasible interior rather than stalling.
        c = score if ok else 1e6
        cache[key] = (c, ok, info)
        if verbose:
            tag = "OK  " if ok else "FAIL"
            extra = (f"util_peak={info.get('util_peak',0):.3f} path_len={info.get('path_len',0):.3f}"
                     if ok else f"({info.get('reason','')})")
            print(f"    h={h:.3f}: {tag} cost={c:.3f} {extra}")
        return cache[key]

    # Golden-section search on [h_min, h_max].
    gr = (math.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    a, b = h_min, h_max
    c1 = b - gr * (b - a)
    c2 = a + gr * (b - a)
    f1 = cost_of(c1)[0]
    f2 = cost_of(c2)[0]
    iters = int(os.environ.get("PAYLOAD_TRANSFER_ITERS", "8"))
    for _ in range(max(3, iters)):
        if abs(b - a) < 0.01:
            break
        if f1 < f2:
            b, c2, f2 = c2, c1, f1
            c1 = b - gr * (b - a)
            f1 = cost_of(c1)[0]
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + gr * (b - a)
            f2 = cost_of(c2)[0]

    # best evaluated height from the cache
    best_h, best = None, None
    best_c = float("inf")
    for key, (c, ok, info) in cache.items():
        if ok and c < best_c:
            best_c, best_h, best = c, key, info
    if best_h is None:
        if verbose:
            print("  no feasible transfer height in the continuous interval")
        return None, None
    if verbose:
        print(f"  selected continuous transfer height h*={best_h:.3f} (cost={best_c:.3f})")
    return best_h, best


def _select_discrete(scene, q_lift, q_set, heights, refine, planner,
                     solve_time, delta, space, w_torque, w_path, verbose):
    hs = candidate_heights(scene, explicit=heights)
    if refine is None:
        refine = os.environ.get("PAYLOAD_TRANSFER_REFINE", "0").lower() in {"1", "true", "yes", "on"}

    best_h, best_score, best_info = None, float("inf"), None
    for h in hs:
        ok, score, info = evaluate_leaf(scene, q_lift, q_set, h,
                                        planner=planner, solve_time=solve_time,
                                        delta=delta, space=space,
                                        w_torque=w_torque, w_path=w_path)
        if verbose:
            tag = "OK  " if ok else "FAIL"
            if ok:
                print(f"  leaf h={h:.3f}: {tag} score={score:.3f} "
                      f"util_peak={info.get('util_peak',0):.3f} "
                      f"util_mean={info.get('util_mean',0):.3f} "
                      f"path_len={info.get('path_len',0):.3f}")
            else:
                print(f"  leaf h={h:.3f}: {tag} ({info.get('reason','')})")
        if ok and score < best_score:
            best_h, best_score, best_info = h, score, info

    if best_h is None:
        if verbose:
            print("  no feasible transfer leaf among coarse candidates")
        return None, None

    if refine:
        # local refinement: sample around best_h at half the coarse spacing.
        if len(hs) >= 2:
            step = 0.5 * abs(hs[1] - hs[0])
        else:
            step = 0.05
        for h in (best_h - step, best_h + step):
            if h <= max(float(s.height) for s in scene.surfaces):
                continue
            ok, score, info = evaluate_leaf(scene, q_lift, q_set, h,
                                            planner=planner, solve_time=solve_time,
                                            delta=delta, space=space,
                                            w_torque=w_torque, w_path=w_path)
            if verbose:
                tag = "OK  " if ok else "FAIL"
                if ok:
                    print(f"  refine h={h:.3f}: {tag} score={score:.3f} "
                          f"util_peak={info.get('util_peak',0):.3f} "
                          f"path_len={info.get('path_len',0):.3f}")
                else:
                    print(f"  refine h={h:.3f}: {tag} ({info.get('reason','')})")
            if ok and score < best_score:
                best_h, best_score, best_info = h, score, info

    if verbose:
        print(f"  selected transfer height h={best_h:.3f} (score={best_score:.3f})")
    return best_h, best_info


# --------------------------------------------------------------------------- #
# Demo entry point (does NOT replace transport.plan; it sits beside it).
# --------------------------------------------------------------------------- #
def demo_select(scene, q_lift, q_set, **kwargs):
    """
    Minimal demonstration: given a candidate lift-off and set-down config,
    select the best transfer height and return it with the planned sub-segments.
    Wire this in only where you explicitly want the foliation behaviour.
    """
    return select_transfer_height(scene, q_lift, q_set, **kwargs)


# =========================================================================== #
# 2-D foliation: leaves parameterised by (height h, carry yaw psi).
# =========================================================================== #
def _project_to_leaf_2d(scene, q_seed, h, psi):
    """Project q_seed onto the 2-D leaf (z=h, yaw=psi); return (q_on, leaf)."""
    leaf = transfer_leaf_2d(scene, h, psi)
    try:
        out = leaf.project(np.asarray(q_seed, float))
    except Exception:
        return None, leaf
    if isinstance(out, tuple):
        q_on = out[0]
        converged = bool(out[1]) if len(out) > 1 else True
    else:
        q_on, converged = out, True
    if q_on is None or not converged:
        return None, leaf
    q_on = np.asarray(q_on, float).reshape(-1)
    if q_on.shape[0] != int(scene.robot.nq):
        return None, leaf
    if not bool(leaf.feasible(q_on)):
        return None, leaf
    return q_on, leaf


def evaluate_leaf_2d(scene, q_lift, q_set, h, psi,
                     planner="RRTConnect", solve_time=15.0, delta=0.08,
                     space="atlas", w_torque=10.0, w_path=1.0):
    """
    Route lift -> (transfer on the (h,psi) leaf) -> set.  Lift/drop transitions
    are in free-flight mode (so the path starts at q_lift, ends at q_set); the
    cross is pinned to z=h AND yaw=psi.  Returns (ok, score, info).
    """
    info = {"h": float(h), "psi": float(psi), "reason": ""}
    free_mode = T.free_flight_mode(scene)

    q_in, leaf = _project_to_leaf_2d(scene, q_lift, h, psi)
    if q_in is None:
        info["reason"] = "lift will not project to (h,psi) leaf"
        return False, float("inf"), info
    q_out, _ = _project_to_leaf_2d(scene, q_set, h, psi)
    if q_out is None:
        info["reason"] = "set will not project to (h,psi) leaf"
        return False, float("inf"), info

    seg_lift = T._plan_segment(scene.robot, free_mode, q_lift, q_in,
                               planner, solve_time, delta, space)
    if not seg_lift.success:
        info["reason"] = f"lift->leaf failed: {seg_lift.message}"
        return False, float("inf"), info
    seg_cross = T._plan_segment(scene.robot, leaf, q_in, q_out,
                                planner, solve_time, delta, space)
    if not seg_cross.success:
        info["reason"] = f"cross failed: {seg_cross.message}"
        return False, float("inf"), info
    seg_drop = T._plan_segment(scene.robot, free_mode, q_out, q_set,
                               planner, solve_time, delta, space)
    if not seg_drop.success:
        info["reason"] = f"leaf->set failed: {seg_drop.message}"
        return False, float("inf"), info

    def _len(seg):
        p = np.asarray(seg.path, float)
        return 0.0 if (p is None or len(p) < 2) else float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    path_len = _len(seg_lift) + _len(seg_cross) + _len(seg_drop)

    torque = TorqueFeasibility(scene.robot, scene.tau_max, derate=scene.derate)
    util_peak, util_sum, n = 0.0, 0.0, 0
    for q in np.asarray(seg_cross.path, float):
        try:
            u = float(np.max(torque.utilisation(q)))
        except Exception:
            continue
        util_peak = max(util_peak, u); util_sum += u; n += 1
    util_mean = (util_sum / n) if n else 0.0

    w_torque = float(os.environ.get("PAYLOAD_W_TORQUE", str(w_torque)))
    w_path = float(os.environ.get("PAYLOAD_W_PATH", str(w_path)))
    w_torque_mean = float(os.environ.get("PAYLOAD_W_TORQUE_MEAN", "0.25"))
    score = w_torque * util_peak + w_torque_mean * util_mean + w_path * path_len
    info.update({"path_len": path_len, "util_peak": util_peak, "util_mean": util_mean,
                 "segments": [("lift_to_leaf", seg_lift.path),
                              ("cross", seg_cross.path),
                              ("leaf_to_set", seg_drop.path)]})
    return True, score, info


def _yaw_bounds():
    """Carry-yaw search interval [psi_min, psi_max] in radians."""
    lo = float(os.environ.get("PAYLOAD_TRANSFER_PSI_MIN", "-0.5"))
    hi = float(os.environ.get("PAYLOAD_TRANSFER_PSI_MAX", "0.5"))
    if hi <= lo:
        hi = lo + 0.2
    return lo, hi


def select_transfer_2d(scene, q_lift, q_set,
                       planner="RRTConnect", solve_time=15.0, delta=0.08,
                       space="atlas", w_torque=10.0, w_path=1.0, verbose=True):
    """
    Coordinate-descent over the 2-D leaf (h, psi): optimise h at psi=0, then psi
    at that h, then re-optimise h, scoring by torque utilisation + path length.
    Returns (best_h, best_psi, best_info) or (None, None, None).
    """
    import math
    h_min, h_max = _height_bounds(scene)
    psi_min, psi_max = _yaw_bounds()
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    iters = int(os.environ.get("PAYLOAD_TRANSFER_ITERS", "6"))
    cache = {}

    def cost(h, psi):
        key = (round(float(h), 4), round(float(psi), 4))
        if key in cache:
            return cache[key]
        ok, score, info = evaluate_leaf_2d(scene, q_lift, q_set, h, psi,
                                           planner=planner, solve_time=solve_time,
                                           delta=delta, space=space,
                                           w_torque=w_torque, w_path=w_path)
        c = score if ok else 1e6
        cache[key] = (c, ok, info)
        if verbose:
            tag = "OK  " if ok else "FAIL"
            extra = (f"util_peak={info.get('util_peak',0):.3f} path_len={info.get('path_len',0):.3f}"
                     if ok else f"({info.get('reason','')})")
            print(f"    (h={h:.3f}, psi={psi:+.2f}): {tag} cost={c:.3f} {extra}")
        return cache[key]

    def golden(f1d, a, b):
        """Golden-section minimise a 1-D function over [a,b], return best x."""
        c1 = b - gr * (b - a); c2 = a + gr * (b - a)
        f1 = f1d(c1); f2 = f1d(c2)
        for _ in range(max(3, iters)):
            if abs(b - a) < 0.01:
                break
            if f1 < f2:
                b, c2, f2 = c2, c1, f1; c1 = b - gr * (b - a); f1 = f1d(c1)
            else:
                a, c1, f1 = c1, c2, f2; c2 = a + gr * (b - a); f2 = f1d(c2)
        return 0.5 * (a + b)

    if verbose:
        print(f"  2-D foliation: h in [{h_min:.3f},{h_max:.3f}], "
              f"psi in [{psi_min:+.2f},{psi_max:+.2f}]")

    # coordinate descent
    psi_cur = 0.0
    h_cur = golden(lambda h: cost(h, psi_cur)[0], h_min, h_max)
    psi_cur = golden(lambda p: cost(h_cur, p)[0], psi_min, psi_max)
    h_cur = golden(lambda h: cost(h, psi_cur)[0], h_min, h_max)

    best_h, best_psi, best, best_c = None, None, None, float("inf")
    for (kh, kp), (c, ok, info) in cache.items():
        if ok and c < best_c:
            best_c, best_h, best_psi, best = c, kh, kp, info
    if best_h is None:
        if verbose:
            print("  no feasible (h,psi) leaf found")
        return None, None, None
    if verbose:
        print(f"  selected (h*={best_h:.3f}, psi*={best_psi:+.3f}) cost={best_c:.3f} "
              f"util_peak={best.get('util_peak',0):.3f}")
    return best_h, best_psi, best