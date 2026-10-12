"""
regrasp_plan.py  --  simulation-only multi-grasp regrasp demo.

Idea (sim only, never on the robot): the object is carried between a series of
STABLE PLACEMENTS ("stations" -- the transition points of the multimodal
system).  Between two stations the object is held in one grasp (a TRANSFER).
At a station the object is set down and the gripper is free to RE-GRASP it in a
different way before the next transfer.  Which grasp is used on each transfer is
CHOSEN by a cost (gravity-torque of holding the box in that grasp + joint-path
length + a small penalty for switching), so a regrasp only happens when it pays
off -- e.g. when the next placement needs an orientation the current grasp can't
reach, or when a different grasp holds the box at lower torque.

Manifold view (for the writeup): each grasp defines a TRANSFER mode -- a
constraint manifold where the box is rigidly fixed to the gripper at that grasp
transform.  A regrasp is a TRANSITION between two transfer modes, and it can only
happen on their shared boundary: a configuration where the box is at a stable
placement AND both the releasing and the acquiring grasp are valid.  This is the
classic transit/transfer manipulation-planning structure, expressed in the same
constraint-manifold language as the main thesis.

Reuses the already-working helpers in run_box_grasp_tsr.py and the grasp library;
gives the box mass via box_payload.  Touches no robot-project file.

Run:
    python -B regrasp_plan.py
    python -B animate_regrasp.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from scene import Scene
from run_panda import CFG
from constraints import (
    CollisionConstraint,
    FrameConstraint,
    JointLimit,
    Mode,
    TorqueFeasibility,
    WorkspaceBoundary,
)
from ompl_planner import plan_in_mode
from box_grasp_library import (
    grasp_library,
    BOX_SIZE,
    BOX_CENTER_WORLD,
    BOX_TARGET_CENTER_WORLD,
    TABLE_XY_BOUNDS,
    TABLE_HEIGHT,
    make_T,
    rpy_to_matrix,
)
import box_payload
from box_demo_collision import BoxDemoCollisionChecker, BoxDemoCollisionError

# Reuse the tested seed helper from the single-grasp demo (read-only import).
from run_box_grasp_tsr import _seed_set

HERE = Path(__file__).resolve().parent
BASE_URDF = HERE / "fer_franka_hand_payload.urdf"   # base with payload_joint + payload link
OUT = HERE / "box_regrasp_plan.npz"

FINGER_MAX = 0.04
OPEN_W = FINGER_MAX
SWITCH_PENALTY = float(os.environ.get("REGRASP_SWITCH_PENALTY", "0.15"))
W_TORQUE = float(os.environ.get("REGRASP_W_TORQUE", "1.0"))
W_PATH = float(os.environ.get("REGRASP_W_PATH", "0.40"))


# ---------------------------------------------------------------------------
# Scored OMPL regrasp planner.
# ---------------------------------------------------------------------------

TCP_FRAME = "fer_hand_tcp"
TRANSFER_PLANNER = os.environ.get("BOX_REGRASP_PLANNER", os.environ.get("PAYLOAD_PLANNER_NAME", "RRTConnect"))
TRANSFER_SOLVE_TIME = float(os.environ.get("BOX_REGRASP_SOLVE_TIME", "6.0"))
TRANSFER_DELTA = float(os.environ.get("BOX_REGRASP_DELTA", "0.12"))
TRANSFER_SPACE = os.environ.get("BOX_REGRASP_CONSTRAINED_SPACE", "projected").strip().lower()
TRANSFER_INTERPOLATE_POINTS = int(os.environ.get("BOX_REGRASP_INTERPOLATE_POINTS", "200"))
SHORTCUT_ITERS = int(os.environ.get("BOX_REGRASP_SHORTCUT_ITERS", "80"))
SHORTCUT_STEP = float(os.environ.get("BOX_REGRASP_SHORTCUT_STEP", "0.10"))
N_REGRASP_PLACEMENTS = int(os.environ.get("BOX_REGRASP_PLACEMENTS", "6"))
MAX_ROUTE_TRIALS = int(os.environ.get("BOX_REGRASP_MAX_ROUTE_TRIALS", "80"))
OPEN_RAMP_STEPS = int(os.environ.get("BOX_REGRASP_OPEN_CLOSE_STEPS", "14"))
TABLE_THICKNESS = float(os.environ.get("BOX_DEMO_TABLE_THICKNESS", "0.025"))
CARRY_CLEARANCE = float(os.environ.get("BOX_REGRASP_CARRY_CLEARANCE", "0.005"))
W_CLEAR = float(os.environ.get("REGRASP_W_CLEAR", "0.20"))
REGRASP_TARGET_YAW_DEG = float(os.environ.get("BOX_REGRASP_TARGET_YAW_DEG", "0.0"))


def _box_points_local():
    half = 0.5 * np.asarray(BOX_SIZE, dtype=float)
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append([sx * half[0], sy * half[1], sz * half[2]])
    pts.append([0.0, 0.0, 0.0])
    return np.asarray(pts, dtype=float)


def _table_obstacle():
    (xlo, ylo), (xhi, yhi) = np.asarray(TABLE_XY_BOUNDS, dtype=float).reshape(2, 2)
    top = TABLE_HEIGHT - float(os.environ.get("BOX_REGRASP_TABLE_PLANNING_CLEARANCE", "0.003"))
    T = np.eye(4, dtype=float)
    T[:3, 3] = [0.5 * (xlo + xhi), 0.5 * (ylo + yhi), top - 0.5 * TABLE_THICKNESS]
    return {
        "type": "box",
        "name": "box_demo_table_solid",
        "T": T,
        "xsize": float(xhi - xlo),
        "ysize": float(yhi - ylo),
        "zsize": float(TABLE_THICKNESS),
    }


def _base_obstacle():
    T = np.eye(4, dtype=float)
    T[:3, 3] = [0.0, 0.0, 0.09]
    return {
        "type": "cylinder",
        "name": "robot_base_proxy",
        "T": T,
        "radius": float(os.environ.get("BOX_DEMO_BASE_RADIUS", "0.16")),
        "height": float(os.environ.get("BOX_DEMO_BASE_HEIGHT", "0.18")),
    }


def _scene_cfg(urdf, *, controlled_frame, held):
    cfg = dict(CFG)
    cfg["urdf"] = str(urdf)
    cfg["controlled_frame"] = str(controlled_frame)
    cfg["surfaces"] = [
        {
            "name": "table",
            "height": float(TABLE_HEIGHT),
            "xy_bounds": np.asarray(TABLE_XY_BOUNDS, dtype=float).tolist(),
        }
    ]
    cfg["table_collision_margin"] = 0.0
    cfg["obstacles"] = [_table_obstacle(), _base_obstacle()]
    frames = [
        "fer_link1",
        "fer_link2",
        "fer_link3",
        "fer_link4",
        "fer_link5",
        "fer_link6",
        "fer_link7",
        "fer_link8",
        "fer_hand",
        "fer_hand_tcp",
        "fer_leftfinger",
        "fer_rightfinger",
    ]
    if held:
        frames.append("payload")
        cfg["collision_frame_points"] = {"payload": _box_points_local()}
    else:
        cfg["collision_frame_points"] = {}
    cfg["collision_frames"] = frames
    cfg["self_collision_frames"] = [
        "fer_link1",
        "fer_link2",
        "fer_link3",
        "fer_link4",
        "fer_link5",
        "fer_link6",
        "fer_link7",
        "fer_link8",
        "fer_hand",
    ] + (["payload"] if held else [])
    cfg["self_collision_min_distance"] = float(os.environ.get("BOX_REGRASP_SELF_COLLISION_MIN_DISTANCE", "0.035"))
    cfg["torque_derate"] = float(os.environ.get("BOX_REGRASP_TORQUE_DERATE", cfg.get("torque_derate", 0.85)))
    return cfg


def _collision(scene):
    return CollisionConstraint(
        scene.robot,
        scene.collision_frames,
        scene.obstacles,
        frame_points=scene.collision_frame_points,
        self_collision_frames=scene.self_collision_frames,
        self_collision_min_distance=scene.self_collision_min_distance,
    )


def _transfer_mode(scene, grasp_name):
    half_z = 0.5 * float(np.asarray(BOX_SIZE, dtype=float)[2])
    return Mode(
        f"transfer_{grasp_name}",
        inequalities=[
            JointLimit(scene.robot),
            WorkspaceBoundary(
                scene.robot,
                "payload",
                [-1.0e3, -1.0e3, TABLE_HEIGHT + half_z + CARRY_CLEARANCE],
                [1.0e3, 1.0e3, 1.0e3],
                name=f"payload_carry_clearance_{grasp_name}",
            ),
            _collision(scene),
            TorqueFeasibility(scene.robot, scene.tau_max, derate=scene.derate),
        ],
    )


def _transit_mode(scene):
    return Mode(
        "transit_empty_gripper",
        inequalities=[
            JointLimit(scene.robot),
            _collision(scene),
        ],
    )


_HELD_SCENES = {}
_EMPTY_SCENE = None


def _held_scene(grasp):
    key = grasp.name
    if key not in _HELD_SCENES:
        urdf = box_payload.ensure_box_urdf_for_grasp(BASE_URDF, HERE, grasp.name, grasp.T_box_tcp)
        _HELD_SCENES[key] = Scene(_scene_cfg(urdf, controlled_frame="payload", held=True))
    return _HELD_SCENES[key]


def _empty_scene():
    global _EMPTY_SCENE
    if _EMPTY_SCENE is None:
        _EMPTY_SCENE = Scene(_scene_cfg(HERE / "fer_franka_hand_render.urdf", controlled_frame=TCP_FRAME, held=False))
    return _EMPTY_SCENE


def _seed_list(scene, extra=()):
    seeds = [np.asarray(q, dtype=float) for q in extra if q is not None]
    seeds.extend(np.asarray(q, dtype=float) for q in _seed_set(scene))
    return seeds


def _project_frame_pose_once(scene, frame, T_target, seeds, name, inequalities=(), *, exact=True):
    if exact:
        pos_tol = [0.0, 0.0, 0.0]
        rot_tol = [0.0, 0.0, 0.0]
        project_tol = 1.0e-7
        max_iter = 160
    else:
        pos_tol = [0.006, 0.006, 0.006]
        rot_tol = np.deg2rad([8.0, 8.0, 12.0])
        project_tol = 1.0e-5
        max_iter = 120
    mode = Mode(
        name,
        equalities=[
            FrameConstraint(
                scene.robot,
                frame,
                target_position=np.asarray(T_target, dtype=float)[:3, 3],
                pos_mask=(1, 1, 1),
                pos_tol=pos_tol,
                target_rotation=np.asarray(T_target, dtype=float)[:3, :3],
                rot_mask=(1, 1, 1),
                rot_tol=rot_tol,
                name=f"{name}_{frame}",
            )
        ],
        inequalities=[JointLimit(scene.robot), *inequalities],
    )
    qlo, qhi = scene.robot.joint_limits
    for seed in seeds:
        q, ok = mode.project(seed, max_iter=max_iter, tol=project_tol, qlo=qlo, qhi=qhi)
        if ok and mode.feasible(q):
            return np.asarray(q, dtype=float), True
    return None, False


def _project_frame_pose(scene, frame, T_target, seeds, name, inequalities=()):
    q, ok = _project_frame_pose_once(
        scene, frame, T_target, seeds, name + "_exact", inequalities, exact=True
    )
    if ok:
        return q, True
    return _project_frame_pose_once(
        scene, frame, T_target, seeds, name + "_relaxed", inequalities, exact=False
    )


def _world_box(center, yaw_deg=0.0):
    R = rpy_to_matrix([0.0, 0.0, np.deg2rad(float(yaw_deg))])
    c = np.asarray(center, dtype=float).reshape(3).copy()
    c[2] = TABLE_HEIGHT + 0.5 * float(BOX_SIZE[2])
    return make_T(R, c)


def _initial_pose():
    return _world_box(BOX_CENTER_WORLD, 0.0)


def _target_pose():
    return _world_box(BOX_TARGET_CENTER_WORLD, REGRASP_TARGET_YAW_DEG)


def _carry_pose(T_box):
    T = np.asarray(T_box, dtype=float).copy()
    T[:3, 3] = T[:3, 3] + np.array([0.0, 0.0, CARRY_CLEARANCE], dtype=float)
    return T


def _sample_regrasp_placements():
    rng = np.random.default_rng(20260626)
    (xlo, ylo), (xhi, yhi) = np.asarray(TABLE_XY_BOUNDS, dtype=float).reshape(2, 2)
    margin = 0.07
    poses = []
    yaw_choices = sorted(set([0.0, REGRASP_TARGET_YAW_DEG, 90.0, -90.0]))
    for _ in range(max(0, N_REGRASP_PLACEMENTS)):
        x = float(rng.uniform(xlo + margin, xhi - margin))
        y = float(rng.uniform(ylo + margin, yhi - margin))
        yaw = float(rng.choice(yaw_choices))
        poses.append(_world_box([x, y, TABLE_HEIGHT + 0.5 * BOX_SIZE[2]], yaw))
    return poses


def _hold_finger_width_new(grasp):
    gap = float(grasp.object_width + 2.0 * grasp.contact_margin)
    return float(min(FINGER_MAX, 0.5 * gap))


_hold_finger_width = _hold_finger_width_new


def _feasible_grasps_new():
    return [g for g in grasp_library() if g.required_width <= 2.0 * FINGER_MAX + 1.0e-9]


def _joint_path_length(path):
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _path_valid(mode, path, tol=1.0e-4):
    return all(mode.valid(np.asarray(q, dtype=float), tol=tol) for q in np.asarray(path, dtype=float))


def _projected_shortcut(mode, q_a, q_b):
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    dist = float(np.linalg.norm(q_b - q_a))
    n = max(2, int(np.ceil(dist / max(SHORTCUT_STEP, 1.0e-6))) + 1)
    out = [q_a]
    q_prev = q_a
    for alpha in np.linspace(0.0, 1.0, n)[1:-1]:
        seed = (1.0 - float(alpha)) * q_a + float(alpha) * q_b
        q_proj, ok = mode.project(seed, max_iter=80, tol=1.0e-5)
        if not ok or not mode.valid(q_proj, tol=1.0e-4):
            return None
        q_proj = np.asarray(q_proj, dtype=float)
        if np.linalg.norm(q_proj - q_prev) > 1.0e-8:
            out.append(q_proj)
            q_prev = q_proj
    out.append(q_b)
    candidate = np.asarray(out, dtype=float)
    if not _path_valid(mode, candidate):
        return None
    return candidate


def _reprojected_shortcut_path(mode, path):
    best = np.asarray(path, dtype=float)
    if len(best) <= 2 or SHORTCUT_ITERS <= 0:
        return best
    rng = np.random.default_rng(20260627)
    best_len = _joint_path_length(best)
    for _ in range(max(0, SHORTCUT_ITERS)):
        if len(best) <= 2:
            break
        i = int(rng.integers(0, len(best) - 2))
        j = int(rng.integers(i + 2, len(best)))
        shortcut = _projected_shortcut(mode, best[i], best[j])
        if shortcut is None:
            continue
        candidate = np.vstack([best[:i], shortcut, best[j + 1 :]])
        candidate_len = _joint_path_length(candidate)
        if candidate_len + 1.0e-6 < best_len and _path_valid(mode, candidate):
            best = candidate
            best_len = candidate_len
    return best


def _torque_util(scene, path):
    vals = []
    tau = np.asarray(scene.tau_max, dtype=float) * float(scene.derate)
    for q in np.asarray(path, dtype=float):
        vals.append(float(np.max(np.abs(scene.robot.gravity(q)) / tau)))
    return float(max(vals)) if vals else float("inf")


def _clearance_penalty(checker, scene, path):
    penalty = 0.0
    for q in np.asarray(path, dtype=float)[:: max(1, len(path) // 20)]:
        for frame in ("fer_link6", "fer_link7", "fer_link8", "fer_hand", "fer_hand_tcp"):
            try:
                p = np.asarray(scene.robot.frame_pose(frame, q)[:3, 3], dtype=float)
            except Exception:
                continue
            if checker._point_near_table_xy(p, margin=0.08):
                clearance = float(p[2] - TABLE_HEIGHT)
                penalty += max(0.0, 0.05 - clearance)
    return float(penalty)


def _plan_segment(scene, mode, q0, q1):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    raw = plan_in_mode(
        scene.robot,
        mode,
        q0,
        q1,
        planner_name=TRANSFER_PLANNER,
        solve_time=TRANSFER_SOLVE_TIME,
        delta=TRANSFER_DELTA,
        space=TRANSFER_SPACE,
        simplify=False,
        interpolate_points=0,
    )
    simplified = plan_in_mode(
        scene.robot,
        mode,
        q0,
        q1,
        planner_name=TRANSFER_PLANNER,
        solve_time=TRANSFER_SOLVE_TIME,
        delta=TRANSFER_DELTA,
        space=TRANSFER_SPACE,
        simplify=True,
        interpolate_points=TRANSFER_INTERPOLATE_POINTS,
    )
    if simplified.success and simplified.path is not None:
        before = np.asarray(simplified.path, dtype=float)
        after = _reprojected_shortcut_path(mode, before)
        if _path_valid(mode, after):
            simplified.path = after
        simplified.raw_report_path_length = (
            _joint_path_length(raw.path)
            if raw.success and raw.path is not None
            else float("nan")
        )
        simplified.ompl_simplified_path_length = _joint_path_length(before)
        simplified.final_path_length = _joint_path_length(np.asarray(simplified.path, dtype=float))
        simplified.raw_report_success = bool(raw.success and raw.path is not None)
    return simplified


def _plan_transfer(grasp, T_a, T_b, checker):
    scene = _held_scene(grasp)
    mode = _transfer_mode(scene, grasp.name)
    T_a_carry = _carry_pose(T_a)
    T_b_carry = _carry_pose(T_b)
    q_a, ok_a = _project_frame_pose(
        scene, "payload", T_a_carry, _seed_list(scene), f"{grasp.name}_from", inequalities=mode.inequalities
    )
    if not ok_a:
        return None, "from placement projection failed"
    q_b, ok_b = _project_frame_pose(
        scene, "payload", T_b_carry, _seed_list(scene, [q_a]), f"{grasp.name}_to", inequalities=mode.inequalities
    )
    if not ok_b:
        return None, "to placement projection failed"
    result = _plan_segment(scene, mode, q_a, q_b)
    if not result.success or result.path is None:
        return None, f"OMPL transfer failed: {result.message}"
    path = np.asarray(result.path, dtype=float)
    cost_path = _joint_path_length(path)
    cost_torque = _torque_util(scene, path)
    cost_clear = _clearance_penalty(checker, scene, path)
    return {
        "scene": scene,
        "mode": mode,
        "path": path,
        "q_start": q_a,
        "q_goal": q_b,
        "T_carry_start": T_a_carry,
        "T_carry_goal": T_b_carry,
        "raw_path_len": float(getattr(result, "raw_report_path_length", np.nan)),
        "ompl_path_len": float(getattr(result, "ompl_simplified_path_length", cost_path)),
        "path_len": cost_path,
        "torque": cost_torque,
        "clear": cost_clear,
        "cost": W_PATH * cost_path + W_TORQUE * cost_torque + W_CLEAR * cost_clear,
    }, "ok"


def _plan_transit(T_box, grasp_from, q_from, grasp_to, checker):
    scene = _empty_scene()
    mode = _transit_mode(scene)
    T_tcp_to = np.asarray(T_box, dtype=float) @ np.asarray(grasp_to.T_box_tcp, dtype=float)
    q_to, ok = _project_frame_pose(
        scene, TCP_FRAME, T_tcp_to, _seed_list(scene, [q_from]), f"transit_to_{grasp_to.name}", inequalities=mode.inequalities
    )
    if not ok:
        return None, "next grasp projection failed"
    if not mode.feasible(q_from):
        return None, "release configuration is not transit-feasible"
    result = _plan_segment(scene, mode, q_from, q_to)
    if not result.success or result.path is None:
        return None, f"OMPL transit failed: {result.message}"
    path = np.asarray(result.path, dtype=float)
    cost_path = _joint_path_length(path)
    cost_clear = _clearance_penalty(checker, scene, path)
    switch = 0.0 if grasp_from.name == grasp_to.name else SWITCH_PENALTY
    return {
        "scene": scene,
        "mode": mode,
        "path": path,
        "q_start": q_from,
        "q_goal": q_to,
        "raw_path_len": float(getattr(result, "raw_report_path_length", np.nan)),
        "ompl_path_len": float(getattr(result, "ompl_simplified_path_length", cost_path)),
        "path_len": cost_path,
        "torque": 0.0,
        "clear": cost_clear,
        "switch": switch,
        "cost": W_PATH * cost_path + W_CLEAR * cost_clear + switch,
    }, "ok"


def _candidate_routes(checker):
    initial = _initial_pose()
    target = _target_pose()
    grasps = _feasible_grasps_new()
    candidates = []
    failures = []

    for g in grasps:
        seg, reason = _plan_transfer(g, initial, target, checker)
        if seg is None:
            failures.append(f"direct {g.name}: {reason}")
            continue
        route = {
            "kind": "direct",
            "grasps": [g],
            "placements": [initial, target],
            "segments": [("transfer", g, seg)],
            "cost": seg["cost"],
            "regrasp_flags": [False],
        }
        ok, reason = _route_audits_ok(route, checker)
        if ok:
            candidates.append(route)
        else:
            failures.append(f"direct {g.name}: {reason}")

    trials = 0
    for T_mid in _sample_regrasp_placements():
        for g1 in grasps:
            if trials >= MAX_ROUTE_TRIALS:
                break
            first, reason = _plan_transfer(g1, initial, T_mid, checker)
            if first is None:
                failures.append(f"regrasp first {g1.name}: {reason}")
                continue
            for g2 in grasps:
                if trials >= MAX_ROUTE_TRIALS:
                    break
                trials += 1
                transit, reason = _plan_transit(T_mid, g1, first["q_goal"], g2, checker)
                if transit is None:
                    failures.append(f"regrasp transit {g1.name}>{g2.name}: {reason}")
                    continue
                second, reason = _plan_transfer(g2, T_mid, target, checker)
                if second is None:
                    failures.append(f"regrasp second {g2.name}: {reason}")
                    continue
                total = first["cost"] + transit["cost"] + second["cost"]
                route = {
                    "kind": "one_regrasp",
                    "grasps": [g1, g2],
                    "placements": [initial, T_mid, target],
                    "segments": [("transfer", g1, first), ("transit", None, transit), ("transfer", g2, second)],
                    "cost": total,
                    "regrasp_flags": [False, g1.name != g2.name],
                }
                ok, reason = _route_audits_ok(route, checker)
                if ok:
                    candidates.append(route)
                else:
                    failures.append(f"regrasp route {g1.name}>{g2.name}: {reason}")
    return candidates, failures


class _Frames:
    def __init__(self):
        self.q = []
        self.w = []
        self.attached = []
        self.box = []
        self.state = []
        self.labels = []
        self.scenes = []
        self.modes = []
        self.grasps = []

    def add(self, q, w, attached, T_box, state, label, scene, mode, grasp):
        self.q.append(np.asarray(q, dtype=float))
        self.w.append(float(w))
        self.attached.append(bool(attached))
        self.box.append(np.asarray(T_box, dtype=float))
        self.state.append(str(state))
        self.labels.append(str(label))
        self.scenes.append(scene)
        self.modes.append(mode)
        self.grasps.append(grasp)


def _append_gripper_ramp(F, q, T_box, scene, mode, grasp, w0, w1, label):
    for w in np.linspace(float(w0), float(w1), max(2, OPEN_RAMP_STEPS)):
        F.add(q, w, False, T_box, "resting", label, scene, mode, grasp)


def _build_frames(route):
    F = _Frames()
    open_w = float(FINGER_MAX)
    placements = route["placements"]
    active_grasp = None
    placement_index = 0

    for kind, grasp, seg in route["segments"]:
        if kind == "transit":
            T_static = placements[placement_index]
            _append_gripper_ramp(F, seg["q_start"], T_static, seg["scene"], seg["mode"], active_grasp, _hold_finger_width_new(active_grasp), open_w, f"release_{active_grasp.name}")
            for q in seg["path"]:
                F.add(q, open_w, False, T_static, "resting", "transit_regrasp", seg["scene"], seg["mode"], None)
            continue

        placement_from = placements[placement_index]
        placement_to = placements[placement_index + 1]
        if active_grasp is None or active_grasp.name != grasp.name:
            _append_gripper_ramp(F, seg["q_start"], placement_from, seg["scene"], seg["mode"], grasp, open_w, _hold_finger_width(grasp), f"close_{grasp.name}")
        hold_w = _hold_finger_width(grasp)
        T_grasp_box = np.asarray(seg["scene"].robot.frame_pose("payload", seg["q_start"]), dtype=float)
        F.add(seg["q_start"], hold_w, True, T_grasp_box, "attached", f"grasp_{grasp.name}", seg["scene"], seg["mode"], grasp)
        for q in seg["path"]:
            T_box = np.asarray(seg["scene"].robot.frame_pose("payload", q), dtype=float)
            F.add(q, hold_w, True, T_box, "attached", f"carry_{grasp.name}", seg["scene"], seg["mode"], grasp)
        F.add(seg["q_goal"], hold_w, True, placement_to, "attached", f"arrive_{grasp.name}", seg["scene"], seg["mode"], grasp)
        active_grasp = grasp
        placement_index += 1

    if active_grasp is not None:
        final = placements[-1]
        last_seg = route["segments"][-1][2]
        _append_gripper_ramp(F, last_seg["q_goal"], final, last_seg["scene"], last_seg["mode"], active_grasp, _hold_finger_width_new(active_grasp), open_w, f"release_{active_grasp.name}")
    return F


def _audit_path(F):
    reasons = []
    for i, (q, mode, label) in enumerate(zip(F.q, F.modes, F.labels)):
        if mode is not None and not mode.valid(q, tol=1.0e-4):
            reasons.append(f"sample {i}, phase={label}: mode {mode.name} invalid")
            if len(reasons) >= 8:
                break
    return len(reasons) == 0, reasons


def _collision_audit(F, checker):
    violations = []
    for i, (q, w, T_box, state, label) in enumerate(zip(F.q, F.w, F.box, F.state, F.labels)):
        ok, reason = checker.sample_valid(q, w, T_box, state, label)
        if not ok:
            violations.append(f"sample {i}, phase={label}: {reason}")
            if len(violations) >= 12:
                break
    return len(violations) == 0, violations


def _collision_audit_self_test(F, checker):
    contact_prefixes = ("grasp_", "carry_", "arrive_")
    for q, w, T_box, state, label in zip(F.q, F.w, F.box, F.state, F.labels):
        if not any(str(label).startswith(prefix) for prefix in contact_prefixes):
            continue

        normal_ok, normal_reason = checker.sample_valid(q, w, T_box, state, label)
        T_bad = np.asarray(T_box, dtype=float).copy()
        z_min = float(np.min(T_bad[:3, 3][2] + T_bad[:3, :3][2, :] @ _box_points_local().T))
        T_bad[:3, 3] = T_bad[:3, 3] + np.array(
            [0.0, 0.0, checker.table_height - 0.005 - z_min],
            dtype=float,
        )
        bad_ok, bad_reason = checker.sample_valid(
            q,
            w,
            T_bad,
            state,
            f"{label}_table_penetration_selftest",
        )
        table_rejected = (not bad_ok) and ("table" in str(bad_reason).lower())
        table_bounds = np.asarray(checker.table_xy_bounds, dtype=float).reshape(-1, 2, 2)[0]
        (xlo, ylo), (xhi, yhi) = table_bounds
        forced_link6 = {
            "fer_link6": np.array(
                [0.5 * (xlo + xhi), 0.5 * (ylo + yhi), checker.table_height - 0.005],
                dtype=float,
            )
        }
        arm_ok, arm_reason = checker.robot_table_proxy_points_valid(
            forced_link6,
            f"{label}_forearm_table_selftest",
        )
        arm_rejected = (not arm_ok) and ("table" in str(arm_reason).lower())
        return {
            "normal_ok": bool(normal_ok),
            "normal_reason": str(normal_reason),
            "table_rejected": bool(table_rejected),
            "table_reason": str(bad_reason),
            "arm_rejected": bool(arm_rejected),
            "arm_reason": str(arm_reason),
        }

    return {
        "normal_ok": False,
        "normal_reason": "no grasp contact sample found",
        "table_rejected": False,
        "table_reason": "no grasp contact sample found",
        "arm_rejected": False,
        "arm_reason": "no grasp contact sample found",
    }


def _print_phase_summary(F):
    print("per-phase plan summary:")
    seen = []
    for label in F.labels:
        if label not in seen:
            seen.append(label)
    for label in seen:
        idx = [i for i, lab in enumerate(F.labels) if lab == label]
        widths = np.asarray([F.w[i] for i in idx], dtype=float)
        attached = np.asarray([F.attached[i] for i in idx], dtype=bool)
        follow_errors = []
        for i in idx:
            if not F.attached[i] or F.scenes[i] is None or F.grasps[i] is None:
                continue
            try:
                T_payload = np.asarray(F.scenes[i].robot.frame_pose("payload", F.q[i]), dtype=float)
            except Exception:
                continue
            follow_errors.append(float(np.linalg.norm(T_payload[:3, 3] - F.box[i][:3, 3])))
        if follow_errors:
            max_follow = max(follow_errors)
            follows = "PASS" if max_follow <= 1.0e-6 else f"FAIL max_err={max_follow:.5f}m"
        else:
            follows = "n/a"
        print(
            f"  {label:22s} n={len(idx):4d} "
            f"finger_gap_cm={2.0 * widths.min() * 100.0:.2f}..{2.0 * widths.max() * 100.0:.2f} "
            f"attached_any={bool(np.any(attached))} attached_all={bool(np.all(attached))} "
            f"box_follows_gripper={follows}"
        )


def _route_audits_ok(route, checker):
    frames = _build_frames(route)
    path_ok, path_reasons = _audit_path(frames)
    grasp_ok, grasp_reasons = _grasp_relation_audit(frames)
    collision_ok, collision_reasons = _collision_audit(frames, checker)
    if not path_ok:
        return False, path_reasons[0] if path_reasons else "path audit failed"
    if not grasp_ok:
        return False, grasp_reasons[0] if grasp_reasons else "grasp relation audit failed"
    if not collision_ok:
        return False, collision_reasons[0] if collision_reasons else "collision audit failed"
    return True, "ok"


def _grasp_relation_audit(F):
    reasons = []
    for i, (q, T_box, attached, scene, grasp, label) in enumerate(zip(F.q, F.box, F.attached, F.scenes, F.grasps, F.labels)):
        if not attached or grasp is None or scene is None:
            continue
        T_tcp = np.asarray(scene.robot.frame_pose(TCP_FRAME, q), dtype=float)
        T_expected_box = T_tcp @ np.linalg.inv(np.asarray(grasp.T_box_tcp, dtype=float))
        pos = float(np.linalg.norm(T_expected_box[:3, 3] - T_box[:3, 3]))
        if pos > 0.01:
            reasons.append(f"sample {i}, phase={label}: held box/TCP relation error {pos:.4f} m")
            if len(reasons) >= 8:
                break
    return len(reasons) == 0, reasons


def _save_route(F, route, path_ok, collision_ok):
    chosen = [g.name for g in route["grasps"]]
    segment_costs = [float(seg["cost"]) for _, _, seg in route["segments"]]
    segment_raw_path_lengths = [float(seg.get("raw_path_len", np.nan)) for _, _, seg in route["segments"]]
    segment_ompl_path_lengths = [float(seg.get("ompl_path_len", np.nan)) for _, _, seg in route["segments"]]
    segment_final_path_lengths = [float(seg.get("path_len", np.nan)) for _, _, seg in route["segments"]]
    np.savez(
        OUT,
        plan_version=np.asarray("regrasp_multimodal_ompl_v1"),
        plan_valid=np.asarray(bool(path_ok and collision_ok)),
        path_audit_pass=np.asarray(bool(path_ok)),
        collision_audit_pass=np.asarray(bool(collision_ok)),
        collision_audit_violation_count=np.asarray(0 if collision_ok else 1),
        path=np.asarray(F.q, dtype=float),
        labels=np.asarray(F.labels).astype(str),
        finger_widths=np.asarray(F.w, dtype=float),
        object_attached=np.asarray(F.attached, dtype=bool),
        contact_reached=np.asarray(F.attached, dtype=bool),
        box_poses=np.asarray(F.box, dtype=float),
        box_pose_state=np.asarray(F.state).astype(str),
        box_size=np.asarray(BOX_SIZE, dtype=float),
        T_world_box_initial=_initial_pose(),
        T_world_box_target=_target_pose(),
        T_box_tcp=np.asarray(route["grasps"][0].T_box_tcp, dtype=float),
        table_xy_bounds=np.asarray(TABLE_XY_BOUNDS, dtype=float),
        table_height=np.asarray(float(TABLE_HEIGHT)),
        table_thickness=np.asarray(float(TABLE_THICKNESS)),
        selected_grasp=np.asarray("regrasp:" + ">".join(chosen)),
        selected_grasp_sequence=np.asarray(chosen),
        segment_costs=np.asarray(segment_costs, dtype=float),
        segment_raw_path_lengths=np.asarray(segment_raw_path_lengths, dtype=float),
        segment_ompl_path_lengths=np.asarray(segment_ompl_path_lengths, dtype=float),
        segment_final_path_lengths=np.asarray(segment_final_path_lengths, dtype=float),
        total_cost=np.asarray(float(route["cost"])),
        regrasp_flags=np.asarray(route["regrasp_flags"], dtype=bool),
        collision_model_note=np.asarray(
            "CollisionConstraint/collision.py planning and post audit, plus explicit box contact checks"
        ),
        gripper_hold_finger_width=np.asarray(_hold_finger_width_new(route["grasps"][0])),
    )


def main():
    print("multimodal box regrasp planner (simulation only)")
    print("method: scored grasp/placement search + OMPL plan_in_mode segments")
    print(
        f"planner={TRANSFER_PLANNER}, solve_time={TRANSFER_SOLVE_TIME:.2f}s, "
        f"space={TRANSFER_SPACE}, ompl_simplify=True, shortcut_iters={SHORTCUT_ITERS}, "
        f"carry_clearance={CARRY_CLEARANCE:.4f}m"
    )
    print(f"box mass={box_payload.BOX_MASS:.3f} kg, feasible grasps={[g.name for g in _feasible_grasps_new()]}")

    try:
        audit_scene = _empty_scene()
        checker = BoxDemoCollisionChecker(
            box_size=BOX_SIZE,
            table_xy_bounds=TABLE_XY_BOUNDS,
            table_height=TABLE_HEIGHT,
            scene=audit_scene,
            table_thickness=TABLE_THICKNESS,
            box_table_tol=float(os.environ.get("BOX_DEMO_BOX_TABLE_TOL", "0.0015")),
        )
    except BoxDemoCollisionError as exc:
        print(f"collision checker setup failed: {exc}")
        return 1

    candidates, failures = _candidate_routes(checker)
    if not candidates:
        print("No route candidates survived projection/planning.")
        for line in failures[:20]:
            print("  failed:", line)
        checker.close()
        return 1

    candidates.sort(key=lambda c: c["cost"])
    selected = candidates[0]
    print("candidate summary:")
    for rank, cand in enumerate(candidates[:8], start=1):
        route = ">".join(g.name for g in cand["grasps"])
        print(f"  {rank:2d}. {cand['kind']:11s} grasps={route:32s} cost={cand['cost']:.4f}")
    print("selected route:")
    print(f"  kind={selected['kind']}")
    print(f"  grasp sequence={'>'.join(g.name for g in selected['grasps'])}")
    for idx, (kind, grasp, seg) in enumerate(selected["segments"]):
        gname = "empty" if grasp is None else grasp.name
        print(
            f"  segment {idx}: {kind:8s} {gname:14s} "
            f"cost={seg['cost']:.4f} "
            f"path_raw={seg['raw_path_len']:.4f} "
            f"path_ompl={seg['ompl_path_len']:.4f} "
            f"path_final={seg['path_len']:.4f} "
            f"torque={seg['torque']:.4f} clear={seg['clear']:.4f}"
        )
    print(f"  regrasp decisions={selected['regrasp_flags']}")

    frames = _build_frames(selected)
    _print_phase_summary(frames)
    selftest = _collision_audit_self_test(frames, checker)
    print("collision audit self-test:")
    print(
        "  normal grasp contact: "
        f"{'PASS' if selftest['normal_ok'] else 'FAIL'}"
        + ("" if selftest["normal_ok"] else f" ({selftest['normal_reason']})")
        )
    print(
        "  box 5mm into table: "
        f"{'PASS' if selftest['table_rejected'] else 'FAIL'}"
        + (f" ({selftest['table_reason']})" if selftest["table_rejected"] else "")
    )
    print(
        "  fer_link6 proxy 5mm below table: "
        f"{'PASS' if selftest['arm_rejected'] else 'FAIL'}"
        + (f" ({selftest['arm_reason']})" if selftest["arm_rejected"] else "")
    )
    path_ok, path_reasons = _audit_path(frames)
    grasp_ok, grasp_reasons = _grasp_relation_audit(frames)
    collision_ok, collision_reasons = _collision_audit(frames, checker)
    path_ok = bool(path_ok and grasp_ok)
    selftest_ok = bool(selftest["normal_ok"] and selftest["table_rejected"] and selftest["arm_rejected"])
    collision_ok = bool(collision_ok and selftest_ok)
    print("path audit:", "PASS" if path_ok else "FAIL")
    for reason in (path_reasons + grasp_reasons)[:10]:
        print("  ", reason)
    print("collision audit:", "PASS" if collision_ok else "FAIL")
    for reason in collision_reasons[:10]:
        print("  ", reason)

    _save_route(frames, selected, path_ok, collision_ok)
    checker.close()
    if not (path_ok and collision_ok):
        print(f"saved audited invalid plan for debugging only: {OUT}")
        return 1
    print(f"saved valid regrasp plan: {OUT}")
    print("animate with: python -B animate_regrasp.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
