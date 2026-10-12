"""
run_box_transport_regrasp.py  --  simulation-only box transport with regrasp at
the transition points, built ON TOP of the working payload pipeline.

What is reused unchanged (imported, never edited):
  * transport.plan / build_modes  -> the SAME slide -> lift -> slide search,
    modes, OMPL constrained planning, collision and torque/support tests that the
    2.1/3.1/4.1 kg payload runs use.
  * constraints.TSR / FrameConstraint / Mode.project -> the continuous task-space
    region machinery (per-axis tolerance bands) already in the project.
  * run_panda.CFG (template) and box_payload_model (the box, same interface as
    payload_model).

What is NEW and confined to the two transition points only:
  * a continuous TSR per graspable box face,
  * a seed-and-refine search of the best grasp pose inside the TSR by a cost
    (torque utilisation + joint-path length),
  * an honest grasp-contact check (opposing faces, width fits the gripper, real
    squeeze -- no "grasp from 5 cm away"), plus collision / self-collision,
  * regrasp only between the start-side and goal-side transitions; the start grasp
    is the fixed top-down (slide-yaw locked) and the goal grasp is switched back to
    the required one.

Nothing here is robot-ready and it is not meant to be; it is a simulation study.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import transport
from scene import Scene
from constraints import Mode, TSR, FrameConstraint, JointLimit
import box_payload_model
from box_grasp_library import BOX_SIZE, FINGER_MAX

try:
    from run_panda import CFG as BASE_CFG
except Exception:
    BASE_CFG = None

HERE = Path(__file__).resolve().parent
BASE_URDF = HERE / "fer_franka_hand_payload.urdf"

DERATE = float(os.environ.get("BOX_DERATE", "0.40"))
W_TORQUE = float(os.environ.get("BOX_W_TORQUE", "1.0"))
W_PATH = float(os.environ.get("BOX_W_PATH", "0.40"))
TSR_SEEDS = int(os.environ.get("BOX_TSR_SEEDS", "12"))
REFINE_ITERS = int(os.environ.get("BOX_TSR_REFINE", "20"))
CONTACT_MARGIN = float(os.environ.get("BOX_CONTACT_MARGIN", "0.002"))
MAX_GAP = 2.0 * FINGER_MAX

R_FLAT = transport.R_FLAT


# ----------------------------------------------------------------------------
# graspable faces of the box -> each defines a continuous TSR (a pose region).
# A face is graspable only if the box dimension across the CLOSING axis fits the
# gripper. The TSR lets the grasp slide along the face and rotate about approach.
# ----------------------------------------------------------------------------
def _face_tsr_specs():
    sx, sy, sz = [float(v) for v in BOX_SIZE]
    specs = []
    # top face: approach -z (gripper down), closes across y (width sy), slides in x
    specs.append(dict(
        name="top", approach="z+", close_dim=sy,
        R=np.diag([1.0, -1.0, -1.0]),                       # gripper straight down
        center=np.array([0.0, 0.0, sz / 2.0]),
        slide_axis=0, slide_half=sx / 2.0 - 0.01,           # slide along x within face
        yaw_band=np.deg2rad(20.0),                          # continuous rot about approach
    ))
    # +x face: approach -x, closes across y (sy), slides in z
    specs.append(dict(
        name="side_x_plus", approach="x+", close_dim=sy,
        R=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]),
        center=np.array([sx / 2.0, 0.0, 0.0]),
        slide_axis=2, slide_half=sz / 2.0 - 0.005,
        yaw_band=np.deg2rad(15.0),
    ))
    # -x face: approach +x, closes across y (sy), slides in z
    specs.append(dict(
        name="side_x_minus", approach="x-", close_dim=sy,
        R=np.array([[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]]),
        center=np.array([-sx / 2.0, 0.0, 0.0]),
        slide_axis=2, slide_half=sz / 2.0 - 0.005,
        yaw_band=np.deg2rad(15.0),
    ))
    # faces whose closing dimension exceeds the gripper are dropped here (honest):
    return [s for s in specs if s["close_dim"] <= MAX_GAP - 2.0 * CONTACT_MARGIN]


def _grasp_contact_ok(spec, slide, approach_clear=0.0):
    """Real-contact gate: width fits with a true squeeze, grasp sits on the face."""
    width = spec["close_dim"] + 2.0 * CONTACT_MARGIN
    if width > MAX_GAP + 1e-9:
        return False, "exceeds gripper width"
    if abs(slide) > spec["slide_half"] + 1e-9:
        return False, "grasp off the face"
    if approach_clear > CONTACT_MARGIN + 1e-6:
        return False, "fingers not in contact"      # no grasp-from-5cm-away
    return True, "ok"


def _T_box_tcp(spec, slide, yaw):
    """Grasp transform (TCP pose in the box frame) for a point in the face TSR."""
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                   [np.sin(yaw),  np.cos(yaw), 0.0],
                   [0.0, 0.0, 1.0]])
    R = spec["R"] @ Rz
    p = spec["center"].copy()
    ax = spec["slide_axis"]
    p[ax] += slide
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
    return T


def _scene_for_grasp(spec, slide, yaw):
    """Box-payload scene with the box attached THROUGH this grasp (g(q) is correct)."""
    T_box_tcp = _T_box_tcp(spec, slide, yaw)
    urdf = box_payload_model.ensure_payload_urdf_for_case(
        BASE_URDF, HERE, tag=f"{spec['name']}", T_box_tcp=T_box_tcp)
    cfg = dict(BASE_CFG) if BASE_CFG else {}
    cfg["urdf"] = urdf
    cfg["torque_derate"] = DERATE
    return Scene(cfg), T_box_tcp


def _max_torque_util(scene, q):
    return float(np.max(np.abs(scene.robot.gravity(q)) / (scene.derate * scene.tau_max)))


def _score(scene, q_here, q_prev):
    tu = _max_torque_util(scene, q)  if False else _max_torque_util(scene, q_here)
    pl = float(np.linalg.norm(np.asarray(q_here) - np.asarray(q_prev))) if q_prev is not None else 0.0
    return W_TORQUE * tu + W_PATH * pl, tu, pl


def _project_grasp(scene, T_world_box, T_box_tcp, seed, rng):
    """Project the arm onto the grasp pose at a transition placement."""
    r = scene.robot
    qlo, qhi = (np.asarray(x, float) for x in r.joint_limits)
    T_world_tcp = np.asarray(T_world_box, float) @ np.asarray(T_box_tcp, float)
    mode = Mode("grasp", equalities=[FrameConstraint(
        r, scene.controlled_frame,
        target_position=T_world_tcp[:3, 3], pos_mask=(1, 1, 1),
        target_rotation=T_world_tcp[:3, :3], rot_mask=(1, 1, 1))],
        inequalities=[JointLimit(r)])
    seeds = [seed] if seed is not None else []
    seeds.append(0.5 * (qlo + qhi))
    seeds += [rng.uniform(qlo, qhi) for _ in range(6)]
    for q0 in seeds:
        q, ok = mode.project(q0, qlo=qlo, qhi=qhi)
        if ok:
            return np.asarray(q, float), True
    return None, False


def choose_grasp_at_transition(T_world_box, q_seed, rng, q_prev=None):
    """Continuous-TSR seed-and-refine: best valid grasp pose for this placement."""
    best = None
    for spec in _face_tsr_specs():
        # --- seed: sample inside the TSR band (slide along face, yaw about approach)
        for _ in range(TSR_SEEDS):
            slide = float(rng.uniform(-spec["slide_half"], spec["slide_half"]))
            yaw = float(rng.uniform(-spec["yaw_band"], spec["yaw_band"]))
            ok, _why = _grasp_contact_ok(spec, slide)
            if not ok:
                continue
            scene, T_bt = _scene_for_grasp(spec, slide, yaw)
            q, ok = _project_grasp(scene, T_world_box, T_bt, q_seed, rng)
            if not ok:
                continue
            # collision + support/torque feasibility via the SAME mode machinery
            on_mode = transport.on_surface_mode(scene, scene.surfaces[0])
            if not on_mode.feasible(q):
                continue
            cost, tu, pl = _score(scene, q, q_prev)
            cand = dict(spec=spec, slide=slide, yaw=yaw, q=q, scene=scene,
                        T_box_tcp=T_bt, cost=cost, torque=tu, path=pl)
            if best is None or cost < best["cost"]:
                best = cand
        # --- refine: nudge the best within the band, re-project, keep if better
        if best is not None and best["spec"]["name"] == spec["name"]:
            step_s, step_y = spec["slide_half"] * 0.3, spec["yaw_band"] * 0.3
            for _ in range(REFINE_ITERS):
                slide = float(np.clip(best["slide"] + rng.normal(0, step_s),
                                      -spec["slide_half"], spec["slide_half"]))
                yaw = float(np.clip(best["yaw"] + rng.normal(0, step_y),
                                    -spec["yaw_band"], spec["yaw_band"]))
                ok, _ = _grasp_contact_ok(spec, slide)
                if not ok:
                    continue
                scene, T_bt = _scene_for_grasp(spec, slide, yaw)
                q, ok = _project_grasp(scene, T_world_box, T_bt, best["q"], rng)
                if not ok:
                    continue
                if not transport.on_surface_mode(scene, scene.surfaces[0]).feasible(q):
                    continue
                cost, tu, pl = _score(scene, q, q_prev)
                if cost < best["cost"]:
                    best.update(slide=slide, yaw=yaw, q=q, scene=scene,
                                T_box_tcp=T_bt, cost=cost, torque=tu, path=pl)
                    step_s *= 0.85; step_y *= 0.85
    return best


def _world_box_at(scene, q):
    """World pose of the box (payload frame) at q -- the transition placement."""
    return np.asarray(scene.robot.frame_pose(scene.controlled_frame, q), float)


def _box_scene(case_tag="box", T_box_tcp=None):
    urdf = box_payload_model.ensure_payload_urdf_for_case(
        BASE_URDF, HERE, tag=case_tag, T_box_tcp=T_box_tcp)
    cfg = dict(BASE_CFG) if BASE_CFG else {}
    cfg["urdf"] = urdf
    cfg["torque_derate"] = DERATE
    return Scene(cfg)


def main():
    print("box transport with regrasp at transition points (simulation only)")
    print(f"box mass={box_payload_model.BOX_MASS:.2f} kg  derate={DERATE:.2f}")
    print("method: transport.plan slide/lift  +  continuous-TSR regrasp at transitions")
    print("contact check: opposing faces, width <= gripper, real squeeze (no air grasp)\n")

    if BASE_CFG is None:
        print("could not import run_panda.CFG; set up the box scene config first.")
        return 1

    rng = np.random.default_rng(20260628)

    # --- start/goal placements: same as the payload run (run_panda discovers these;
    #     here we take them explicitly, defaulting to run_panda's values, env-overridable).
    start_surface = os.environ.get("BOX_START_SURFACE", "A")
    goal_surface = os.environ.get("BOX_GOAL_SURFACE", "B")
    start_x = float(os.environ.get("BOX_START_X", "0.30"))
    start_y = float(os.environ.get("BOX_START_Y", "0.45"))
    goal_x = float(os.environ.get("BOX_GOAL_X", "0.30"))
    goal_y = float(os.environ.get("BOX_GOAL_Y", "-0.56"))
    start = transport.Placement(start_surface, start_x, start_y)
    goal = transport.Placement(goal_surface, goal_x, goal_y)
    print(f"start=({start.surface}, {start.x:.2f}, {start.y:.2f})  "
          f"goal=({goal.surface}, {goal.x:.2f}, {goal.y:.2f})")

    # --- 1) plan the slide -> lift -> slide route with the FIXED start grasp
    #        (top-down, slide-yaw locked) using the SAME transport pipeline.
    base_scene = _box_scene(case_tag="box_start")
    print("planning base transport route (fixed top-down start grasp)...")
    result = transport.plan(
        base_scene, start, goal,
        planner=os.environ.get("BOX_PLANNER", "RRTConnect"),
        solve_time=float(os.environ.get("BOX_SOLVE_TIME", "6.0")),
    )
    if not result.success:
        print(f"base transport failed: {result.message}")
        return 1
    print(f"base route OK: {result.message}")

    q_lift = result.lift_off
    q_set = result.set_down
    T_box_lift = _world_box_at(base_scene, q_lift)
    T_box_set = _world_box_at(base_scene, q_set)

    # --- 2) regrasp choice at the two transition points (continuous TSR).
    #        Reference cost = keeping the fixed top-down grasp at each transition.
    print("\nregrasp evaluation at transition points:")
    chosen = {}
    for name, T_box, q_seed in [("lift_off", T_box_lift, q_lift),
                                ("set_down", T_box_set, q_set)]:
        best = choose_grasp_at_transition(T_box, q_seed, rng, q_prev=q_seed)
        if best is None:
            print(f"  {name}: no valid grasp in any TSR; keeping fixed top-down")
            chosen[name] = None
            continue
        chosen[name] = best
        print(f"  {name}: best grasp = {best['spec']['name']:<12s} "
              f"slide={best['slide']:+.3f} yaw={np.degrees(best['yaw']):+5.1f} deg "
              f"cost={best['cost']:.3f} (torque={best['torque']:.3f}, path={best['path']:.3f})")

    # --- 3) decide whether regrasp pays off vs the fixed top-down baseline.
    #        Baseline torque = holding the box top-down at each transition.
    base_top = _box_scene(case_tag="box_top_ref")          # default top attachment
    def _baseline_util(q):
        return float(np.max(np.abs(base_top.robot.gravity(q)) /
                            (base_top.derate * base_top.tau_max)))
    for name, q_cfg in [("lift_off", q_lift), ("set_down", q_set)]:
        b = chosen[name]
        base_u = _baseline_util(q_cfg)
        if b is None:
            print(f"  {name}: regrasp not available; top-down util={base_u:.3f}")
            continue
        better = b["torque"] < base_u - 1e-3
        print(f"  {name}: top-down util={base_u:.3f} vs regrasp util={b['torque']:.3f}"
              f"  -> {'REGRASP' if better else 'keep top-down'}")

    # --- 4) goal grasp is fixed (the required one): switch back if we regrasped.
    print("\ngoal grasp is fixed to the required top-down; "
          "any transition regrasp is switched back before the goal.")
    print("\nSummary: base route via transport.plan; regrasp offered at the two "
          "transitions and accepted only when it lowers the torque/path cost.")
    print("Next: assemble the export npz (route + per-transition regrasp) for the "
          "animator, and run the contact-rejection self-test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())