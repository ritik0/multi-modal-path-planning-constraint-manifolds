"""
verify.py
=========
End-to-end sanity check of the new approach, using only Pinocchio (no PyBullet).

It shows, for a general scene:
  1. the payload really is in g(q)            (dynamics check)
  2. the IK-free projection puts the object on the table where we ask
  3. the lift-off inequality C(q) <= 0 flips from "must stay on table" to
     "may fly free" as the object is brought closer in -- the transition
     boundary is discovered, not hand-placed
  4. many valid lift-off transition configs exist on that boundary

Run:  python -B verify.py
"""

from pathlib import Path

import numpy as np

from scene import Scene, FRANKA_TAU_MAX

HERE = Path(__file__).resolve().parent     # so paths work from any CWD
from constraints import (PlaneConstraint, FlatCarryConstraint, FrameConstraint,
                         TorqueFeasibility, Mode, sample_transition)

np.set_printoptions(precision=3, suppress=True)


def build_modes(scene):
    r = scene.robot
    f = scene.controlled_frame
    torque = TorqueFeasibility(r, scene.tau_max, derate=scene.derate)

    # MODE A: object supported on the table -> stays on plane, held flat.
    #         (torque is NOT required here: the table carries the load.)
    on_table = Mode("on_table",
                    equalities=[PlaneConstraint(r, f, scene.table_height),
                                FlatCarryConstraint(r, f)])

    # MODE B: object in free flight -> held flat, and the arm must be able to
    #         hold it alone everywhere -> torque feasibility is active.
    free_flight = Mode("free_flight",
                       equalities=[FlatCarryConstraint(r, f)],
                       inequalities=[torque])
    return on_table, free_flight, torque


def main():
    cfg = {
        "urdf": str(HERE / "urdf" / "fer_franka_hand_payload.urdf"),
        "controlled_frame": "payload",
        "tau_max": FRANKA_TAU_MAX,
        "torque_derate": 0.85,
        "table_height": 0.30,
    }
    scene = Scene(cfg)
    r = scene.robot
    on_table, free_flight, torque = build_modes(scene)

    print("=" * 64)
    print("1) payload in g(q)?  gravity torque at a neutral-ish reach")
    q_seed = np.array([0, -0.3, 0, -1.8, 0, 1.6, 0.0])
    print("   g(q) =", r.gravity(q_seed), "Nm   (includes the 4 kg link)")

    print("=" * 64)
    print("2-3) sweep reach distance x; project object onto the table at")
    print("     (x, 0, table_h) held flat, then read the lift-off test C(q)")
    print()
    print("     x[m]   proj_ok   util_max   C(q)=util-1   verdict")
    R_flat = np.diag([1., -1., -1.])          # tool z pointing down (a flat grasp)
    boundary_x = None
    prev_sign = None
    for x in np.linspace(0.35, 0.75, 17):
        place = Mode("place",
                     equalities=[FrameConstraint(r, scene.controlled_frame,
                                                 target_position=[x, 0.0, scene.table_height],
                                                 pos_mask=(1, 1, 1),
                                                 target_rotation=R_flat, rot_mask=(1, 1, 1))])
        q, ok = place.project(q_seed)
        if not ok:
            print(f"     {x:4.2f}   no        --         --           (unreachable)")
            continue
        util = float(np.max(torque.utilisation(q)))
        C = util - 1.0
        verdict = "MAY LIFT" if C <= 0 else "stay on table"
        print(f"     {x:4.2f}   yes       {util:5.3f}      {C:+6.3f}       {verdict}")
        if prev_sign is not None and (prev_sign <= 0) and (C > 0):
            boundary_x = x
        prev_sign = C
        q_seed = q                              # warm-start next projection

    print()
    if boundary_x is not None:
        print(f"   --> lift-off boundary is near x = {boundary_x:.2f} m")
        print("       (closer in: arm holds it -> free flight; farther out: must slide)")

    print("=" * 64)
    print("4) many valid lift-off transitions on the on_table <-> free_flight")
    print("   intersection (projected, not hand-placed):")
    seed = np.array([0, -0.4, 0, -2.0, 0, 1.8, 0.0])
    trans = sample_transition(on_table, free_flight, seed, n_tries=40, perturb=0.25)
    print(f"   found {len(trans)} valid transition configs")
    for q in trans[:4]:
        p = r.frame_pose(scene.controlled_frame, q)[:3, 3]
        print(f"     object at {p}   margin={torque.margin(q):+.3f}")


if __name__ == "__main__":
    main()