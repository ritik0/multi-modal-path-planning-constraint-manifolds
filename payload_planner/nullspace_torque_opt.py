#!/usr/bin/env python3
"""
nullspace_torque_opt.py  --  Direction B prototype (opt-in, standalone).

Idea (torque-optimal redundancy resolution, Hollerbach & Suh 1987)
------------------------------------------------------------------
The 7-DOF arm is redundant for the 6-DOF payload-carrying task: for a fixed
payload pose there is a 1-D self-motion (null space of the payload-frame
Jacobian) along which the arm can reconfigure WITHOUT moving the object.
Different postures on that self-motion give different gravity-holding torques.

The probe showed peak carry torque is driven by ARM POSTURE / reach, not by
carry height -- so the right optimisation variable is the redundant posture, not
the height.  This prototype takes an EXISTING carry path (from the current
foliation) and, at each configuration, slides along the null space to the
posture that lowers the peak joint torque, keeping the object pose fixed.

It is standalone: imports transport / transfer_foliation, modifies nothing, and
is imported by nothing.  It reports the peak torque of the carry BEFORE and
AFTER null-space optimisation on the same task, so we can see whether posture
optimisation actually lowers the peak (which height could not).

Run (same env as the working case):
  PAYLOAD_CASE=4p1 PAYLOAD_TORQUE_DERATE=0.60 PAYLOAD_WALL=1 PAYLOAD_WALL_TOP=0.15 \
  PAYLOAD_CONSTRAINED_SPACE=atlas PAYLOAD_DELTA=0.08 PAYLOAD_SOLVE_TIME=15 \
  python -B nullspace_torque_opt.py
"""

import os
import numpy as np

import transport as T
import transfer_foliation as fol
from constraints import TorqueFeasibility


# --------------------------------------------------------------------------- #
# Torque helpers
# --------------------------------------------------------------------------- #
def joint_util(robot, tau_max, derate, q):
    """Per-joint gravity torque utilisation |g(q)| / (derate*tau_max)."""
    g = np.abs(np.asarray(robot.gravity(q), float).reshape(-1))
    return g / (derate * np.asarray(tau_max, float))


def peak_util(robot, tau_max, derate, q):
    return float(np.max(joint_util(robot, tau_max, derate, q)))


# --------------------------------------------------------------------------- #
# Null-space direction of the payload-frame Jacobian at q.
# --------------------------------------------------------------------------- #
def nullspace_dirs(robot, frame, q, tol=1e-6):
    """
    Columns spanning the null space of the 6xnq frame Jacobian: joint-velocity
    directions that do NOT move the payload frame (self-motion / redundancy).
    For a 7-DOF arm and 6-DOF task this is normally 1-D.
    """
    J = np.asarray(robot.frame_jacobian(frame, q), float)   # 6 x nq
    # SVD null space
    U, S, Vt = np.linalg.svd(J)
    rank = int(np.sum(S > tol))
    N = Vt[rank:].T          # nq x (nq-rank)
    return N                 # columns are null-space directions


# --------------------------------------------------------------------------- #
# Optimise one configuration: slide along the null space to lower peak torque,
# staying within joint limits and keeping the object pose fixed.
# --------------------------------------------------------------------------- #
def optimise_config(robot, tau_max, derate, frame, q,
                    scene=None, step=0.02, n_steps=12, max_passes=6):
    """
    Explore the self-motion manifold to lower peak joint torque while keeping the
    object pose EXACTLY fixed.  Each trial: take a small null-space step (which
    lowers torque but drifts the object slightly to first order), then RE-PROJECT
    onto the exact original object pose so the object does not actually move.
    Accept the trial only if it lowers peak torque and re-projection converged.
    """
    from constraints import FrameConstraint, Mode

    qlo, qhi = robot.joint_limits
    qlo = np.asarray(qlo, float); qhi = np.asarray(qhi, float)
    q = np.asarray(q, float).copy()
    best_u = peak_util(robot, tau_max, derate, q)

    # Pin the FULL object pose (position + orientation) at this config, so the
    # re-projection restores exactly where the object was.
    T0 = robot.frame_pose(frame, q)
    pin = FrameConstraint(robot, frame,
                          target_position=T0[:3, 3], pos_mask=(1, 1, 1),
                          target_rotation=T0[:3, :3], rot_mask=(1, 1, 1),
                          name="pin_obj_pose")
    pin_mode = Mode("pin_obj", equalities=[pin], inequalities=[])

    for _ in range(max_passes):
        N = nullspace_dirs(robot, frame, q)
        if N.shape[1] == 0:
            break
        improved = False
        for k in range(N.shape[1]):
            d = N[:, k]
            d = d / (np.linalg.norm(d) + 1e-12)
            for sign in (+1.0, -1.0):
                q_try = q + sign * step * d
                if np.any(q_try < qlo) or np.any(q_try > qhi):
                    continue
                # re-project onto the EXACT original object pose
                q_proj, converged = pin_mode.project(q_try)
                if not converged:
                    continue
                if np.any(q_proj < qlo) or np.any(q_proj > qhi):
                    continue
                u = peak_util(robot, tau_max, derate, q_proj)
                if u < best_u - 1e-4:
                    best_u = u
                    q = q_proj
                    improved = True
        if not improved:
            break
    return q, best_u


# --------------------------------------------------------------------------- #
# Optimise a whole carry path (sequence of configs).
# --------------------------------------------------------------------------- #
def optimise_path(scene, path, verbose=True):
    robot = scene.robot
    frame = scene.controlled_frame
    tau_max = scene.tau_max
    derate = scene.derate
    path = np.asarray(path, float)

    before = max(peak_util(robot, tau_max, derate, q) for q in path)
    out = []
    peak_after = 0.0
    for i, q in enumerate(path):
        q_opt, u = optimise_config(robot, tau_max, derate, frame, q, scene=scene)
        out.append(q_opt)
        peak_after = max(peak_after, u)
    out = np.asarray(out, float)

    # check the object pose barely moved (null-space motion should preserve it)
    def obj_xyz(q):
        return scene.robot.frame_pose(frame, q)[:3, 3]
    max_drift = max(float(np.linalg.norm(obj_xyz(a) - obj_xyz(b)))
                    for a, b in zip(path, out))

    if verbose:
        print(f"  peak util  before={before:.3f}  after={peak_after:.3f}  "
              f"({before-peak_after:+.3f})")
        print(f"  max object-pose drift from null-space motion: {max_drift*1000:.2f} mm")
    return out, before, peak_after, max_drift


# --------------------------------------------------------------------------- #
# Demo: take the current foliation carry, optimise its posture, compare.
# --------------------------------------------------------------------------- #
def main():
    import run_panda
    from scene import Scene
    scene = Scene(run_panda.CFG)
    st = float(os.environ.get("PAYLOAD_SOLVE_TIME", "15"))
    delta = float(os.environ.get("PAYLOAD_DELTA", "0.08"))
    space = os.environ.get("PAYLOAD_CONSTRAINED_SPACE", "atlas")

    sa, sb = scene.surface("A"), scene.surface("B")
    rng = np.random.default_rng(20260612)
    lifts = T.sample_transitions(scene, sa, 300, other_surface=sb, rng=rng, support_margin=0.05)
    sets = T.sample_transitions(scene, sb, 300, other_surface=sa, rng=rng, support_margin=0.05)
    q_lift = sorted(lifts, key=lambda q: float(np.linalg.norm(q)))[0]
    q_set = sorted(sets, key=lambda q: float(np.linalg.norm(q)))[0]

    # existing foliation carry (the reliable one)
    h, info = fol._select_continuous(scene, q_lift, q_set, "RRTConnect",
                                     st, delta, space, 10.0, 0.1, False)
    if h is None:
        raise SystemExit("fixed-height selection failed")
    segs = info.get("segments", [])
    carry = np.vstack([np.asarray(p, float) for _, p in segs])

    print("=" * 66)
    print(f"Null-space torque optimisation on the carry (h*={h:.3f})")
    print("=" * 66)
    out, before, after, drift = optimise_path(scene, carry, verbose=True)
    print("=" * 66)
    print("READING:")
    if after < before - 0.01 and drift < 0.02:
        print(f"  posture optimisation lowered peak torque {before:.3f}->{after:.3f}")
        print(f"  while keeping the object within {drift*1000:.1f} mm.")
        print("  -> Direction B WORKS: redundant posture is the right variable.")
    elif drift >= 0.02:
        print(f"  WARNING object drifted {drift*1000:.1f} mm -- null-space step too")
        print("  large or Jacobian handling needs a finer step; reduce step size.")
    else:
        print(f"  little/no improvement ({before:.3f}->{after:.3f}); the current")
        print("  carry postures may already be near torque-optimal.")
    print("=" * 66)


if __name__ == "__main__":
    main()