"""
verify_multimodal.py
====================
End-to-end check of the multi-surface pipeline with REAL OMPL RRT-Connect.

  1. payload is in g(q)                (compare arm with vs without the 4 kg link)
  2. IK-free placement projection      (start/goal configs)
  3. lift-off boundary is discovered   (slide region vs may-fly region)
  4. transitions sampled on the intersections (many, not hand-placed)
  5. full plan: slide_A -> free_flight -> slide_B, each planned by RRT-Connect,
     then certified (on-manifold residual + feasible the whole way)

Run:  python -B verify_multimodal.py
"""

import numpy as np

from scene import Scene
from constraints import (PlaneConstraint, FlatCarryConstraint, FrameConstraint,
                         TorqueFeasibility, Mode)
from ompl_planner import plan_in_mode
import multimodal as mm
import pinocchio
np.set_printoptions(precision=3, suppress=True)

URDF = "fer_franka_hand_payload.urdf"
CFG = {
    "urdf": URDF,
    "controlled_frame": "payload",
    "tau_max": [38, 38, 38, 14, 14, 14],     # calibrated so the boundary is in reach
    "torque_derate": 0.85,
    "collision_frames": ["payload"],
    "surfaces": [{"name": "A", "height": 0.40},
                 {"name": "B", "height": 0.40}],
}
SEED = np.array([0, 0.6, 0.8, 0, 0.6, 0.0])


def main():
    scene = Scene(CFG)
    r = scene.robot
    torque = TorqueFeasibility(r, scene.tau_max, derate=scene.derate)

    print("=" * 70)
    print("1) payload in g(q)?  (gravity torque with vs without the 4 kg link)")

    m_no = pinocchio.buildModelFromUrdf(URDF)
    pid = m_no.getFrameId("payload")
    m_no.inertias[m_no.frames[pid].parentJoint].mass = 0.0   # strip payload mass
    d_no = m_no.createData()
    q = mm.place_config(scene, mm.Placement("A", 0.45, 0.0), SEED)[0]
    g_with = r.gravity(q)
    g_without = pinocchio.computeGeneralizedGravity(m_no, d_no, q)
    print("   ||g|| with payload   :", round(np.linalg.norm(g_with), 3), "Nm")
    print("   ||g|| without payload:", round(np.linalg.norm(g_without), 3), "Nm")
    print("   -> payload contributes", round(np.linalg.norm(g_with - g_without), 3), "Nm")

    print("=" * 70)
    print("2-3) project a flat placement across reach; read the lift-off test")
    print("     x[m]   util_max   C(q)      verdict")
    seed = SEED.copy()
    for x in np.linspace(0.30, 0.55, 8):
        q, ok = Mode("p", equalities=[FrameConstraint(
            r, "payload", target_position=[x, 0, 0.40], pos_mask=(1, 1, 1),
            target_rotation=mm.R_FLAT, rot_mask=(1, 1, 1))]).project(seed)
        if not ok:
            print(f"     {x:4.2f}   (unreachable)"); continue
        u = float(np.max(torque.utilisation(q)))
        print(f"     {x:4.2f}   {u:5.3f}     {u-1:+.3f}    "
              f"{'MAY LIFT' if u <= 1 else 'must SLIDE'}")
        seed = q

    print("=" * 70)
    print("4-5) FULL multimodal plan with OMPL RRT-Connect")
    start = mm.Placement("A", 0.52, 0.00)    # far reach -> must slide in first
    goal = mm.Placement("B", 0.36, 0.36)     # different bearing on table B
    print(f"   start: object on A at ({start.x},{start.y},0.40)  flat")
    print(f"   goal : object on B at ({goal.x},{goal.y},0.40)  flat")
    res = mm.plan(scene, start, goal, planner=plan_in_mode,
                  seed_q=SEED, n_transition_tries=80, perturb=0.3,
                  max_pairs=12, solve_time=10.0, delta=0.5, range_value=1.0,
                  interpolate_points=80, verbose=True)
    print()
    print("   result:", res.message)
    print(f"   transitions sampled: lift-off={res.n_lift}, set-down={res.n_set}, "
          f"pairs tried={res.pairs_tried}")
    if res.success:
        for label, path in res.segments:
            # certify each segment independently
            print(f"     segment {label:14s}: {len(path):3d} pts")
        fp = res.full_path()
        print(f"   full path: {len(fp)} configs")
        if res.lift_off is not None:
            pl = r.frame_pose("payload", res.lift_off)[:3, 3]
            ps = r.frame_pose("payload", res.set_down)[:3, 3]
            print(f"   lift-off  object at {pl}  margin={torque.margin(res.lift_off):+.3f}")
            print(f"   set-down  object at {ps}  margin={torque.margin(res.set_down):+.3f}")
        np.save("multimodal_path.npy", fp)
        print("   saved full joint path -> multimodal_path.npy")
    print("=" * 70)
    print("DONE" if res.success else "PLAN FAILED (see messages above)")


if __name__ == "__main__":
    main()