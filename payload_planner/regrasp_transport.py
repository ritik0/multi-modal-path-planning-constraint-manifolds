"""
Simulation-only multimodal box transport with regrasp choices.

This extends the transport.py separated-table structure with a grasp dimension:
each candidate chooses slide/lift transition placements and a grasp for the
table-A slide, free-flight transfer, and table-B slide.  If adjacent grasps
differ, the route inserts release -> transit -> close at the same stable box
placement.  Robot-project files are imported read-only.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from scene import Scene
from run_panda import CFG
from constraints import (
    CollisionConstraint,
    ContactSupportFeasibility,
    FlatCarryConstraint,
    FrameConstraint,
    JointLimit,
    Mode,
    PlaneConstraint,
    TorqueFeasibility,
    WorkspaceBoundary,
)
from box_grasp_library import BOX_SIZE, grasp_library, make_T, rpy_to_matrix
import box_payload
from box_demo_collision import BoxDemoCollisionChecker, BoxDemoCollisionError
from ompl_planner import plan_in_mode
from regrasp_plan import (
    FINGER_MAX,
    OPEN_W,
    SWITCH_PENALTY,
    TRANSFER_DELTA,
    TRANSFER_PLANNER,
    TRANSFER_SOLVE_TIME,
    TRANSFER_SPACE,
    W_CLEAR,
    W_PATH,
    W_TORQUE,
    _box_points_local,
    _collision_audit,
    _collision_audit_self_test,
    _grasp_relation_audit,
    _hold_finger_width,
    _joint_path_length,
    _path_valid,
    _print_phase_summary,
    _seed_list,
)
import transport


HERE = Path(__file__).resolve().parent
BASE_URDF = HERE / "fer_franka_hand_payload.urdf"
OUT = HERE / "box_regrasp_transport_plan.npz"

TCP_FRAME = "fer_hand_tcp"
BOX_FLAT_R = np.eye(3, dtype=float)
TABLE_HEIGHT = 0.0
TABLE_THICKNESS = float(os.environ.get("BOX_DEMO_TABLE_THICKNESS", "0.025"))
CARRY_CLEARANCE = float(os.environ.get("BOX_REGRASP_TRANSPORT_CARRY_CLEARANCE", "0.005"))
OPEN_RAMP_STEPS = int(os.environ.get("BOX_REGRASP_OPEN_CLOSE_STEPS", "14"))
N_TRANSITIONS = int(os.environ.get("BOX_REGRASP_TRANSPORT_TRANSITIONS", "24"))
PAIR_BUDGET = int(os.environ.get("BOX_REGRASP_TRANSPORT_PAIR_BUDGET", str(N_TRANSITIONS * N_TRANSITIONS)))
COMBO_BUDGET = int(os.environ.get("BOX_REGRASP_TRANSPORT_COMBO_BUDGET", "160"))
SUPPORT_MARGIN = float(os.environ.get("BOX_REGRASP_TRANSPORT_SUPPORT_MARGIN", "0.04"))
REGRASP_TRANSPORT_YAW_DEG = float(os.environ.get("BOX_REGRASP_TRANSPORT_YAW_DEG", "0.0"))

TABLE_A = {
    "name": "A",
    "height": TABLE_HEIGHT,
    "xy_bounds": [[0.09, 0.20], [0.38, 0.58]],
}
TABLE_B = {
    "name": "B",
    "height": TABLE_HEIGHT,
    "xy_bounds": [[0.09, -0.72], [0.38, -0.20]],
}
START_CENTER = np.array(
    [
        float(os.environ.get("BOX_TRANSPORT_START_X", "0.30")),
        float(os.environ.get("BOX_TRANSPORT_START_Y", "0.45")),
        TABLE_HEIGHT + 0.5 * BOX_SIZE[2],
    ],
    dtype=float,
)
GOAL_CENTER = np.array(
    [
        float(os.environ.get("BOX_TRANSPORT_GOAL_X", "0.30")),
        float(os.environ.get("BOX_TRANSPORT_GOAL_Y", "-0.56")),
        TABLE_HEIGHT + 0.5 * BOX_SIZE[2],
    ],
    dtype=float,
)


@dataclass
class Segment:
    kind: str
    grasp: object | None
    scene: Scene
    mode: Mode
    path: np.ndarray
    q_start: np.ndarray
    q_goal: np.ndarray
    T_static: np.ndarray | None
    cost: float
    path_len: float
    raw_path_len: float
    ompl_path_len: float
    torque: float
    clear: float
    switch: float = 0.0


@dataclass
class Failure:
    pair_index: int
    combo: str
    segment: str
    phase: str
    cause: str
    detail: str
    progress: int

    def line(self) -> str:
        return (
            f"pair={self.pair_index:03d} combo={self.combo} "
            f"segment={self.segment} phase={self.phase} "
            f"cause={self.cause}: {self.detail}"
        )


def _surface_bounds():
    return np.asarray([TABLE_A["xy_bounds"], TABLE_B["xy_bounds"]], dtype=float)


def _combined_bounds():
    bounds = _surface_bounds().reshape(-1, 2)
    return np.asarray(
        [
            [float(np.min(bounds[:, 0])), float(np.min(bounds[:, 1]))],
            [float(np.max(bounds[:, 0])), float(np.max(bounds[:, 1]))],
        ],
        dtype=float,
    )


def _table_obstacles():
    obstacles = []
    for table in (TABLE_A, TABLE_B):
        (xlo, ylo), (xhi, yhi) = np.asarray(table["xy_bounds"], dtype=float).reshape(2, 2)
        top = float(table["height"]) - float(os.environ.get("BOX_REGRASP_TABLE_PLANNING_CLEARANCE", "0.003"))
        T = np.eye(4, dtype=float)
        T[:3, 3] = [0.5 * (xlo + xhi), 0.5 * (ylo + yhi), top - 0.5 * TABLE_THICKNESS]
        obstacles.append(
            {
                "type": "box",
                "name": f"box_transport_table_{table['name']}",
                "T": T,
                "xsize": float(xhi - xlo),
                "ysize": float(yhi - ylo),
                "zsize": float(TABLE_THICKNESS),
            }
        )
    return obstacles


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
    cfg["surfaces"] = [TABLE_A, TABLE_B]
    cfg["table_collision_margin"] = 0.0
    cfg["obstacles"] = [*_table_obstacles(), _base_obstacle()]
    cfg["support_contact_local"] = [0.0, 0.0, -0.5 * float(BOX_SIZE[2])]
    cfg["support_normal_world"] = [0.0, 0.0, 1.0]
    cfg["support_contact_model"] = os.environ.get("BOX_REGRASP_SUPPORT_CONTACT_MODEL", "normal")
    cfg["support_force_max"] = float(os.environ.get("BOX_REGRASP_SUPPORT_FORCE_MAX", "inf"))
    cfg["support_derate"] = float(os.environ.get("BOX_REGRASP_SUPPORT_DERATE", "1.0"))
    cfg["torque_derate"] = float(os.environ.get("BOX_REGRASP_TORQUE_DERATE", cfg.get("torque_derate", 0.85)))
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
    return cfg


_HELD_SCENES: dict[str, Scene] = {}
_EMPTY_SCENE: Scene | None = None


def _held_scene(grasp):
    if grasp.name not in _HELD_SCENES:
        urdf = box_payload.ensure_box_urdf_for_grasp(BASE_URDF, HERE, grasp.name, grasp.T_box_tcp)
        _HELD_SCENES[grasp.name] = Scene(_scene_cfg(urdf, controlled_frame="payload", held=True))
    return _HELD_SCENES[grasp.name]


def _empty_scene():
    global _EMPTY_SCENE
    if _EMPTY_SCENE is None:
        _EMPTY_SCENE = Scene(_scene_cfg(HERE / "fer_franka_hand_render.urdf", controlled_frame=TCP_FRAME, held=False))
    return _EMPTY_SCENE


def _collision(scene):
    return CollisionConstraint(
        scene.robot,
        scene.collision_frames,
        scene.obstacles,
        frame_points=scene.collision_frame_points,
        self_collision_frames=scene.self_collision_frames,
        self_collision_min_distance=scene.self_collision_min_distance,
    )


def _support_torque(scene):
    return ContactSupportFeasibility.from_scene(scene)


def _inset_bounds(surface, margin):
    (xlo, ylo), (xhi, yhi) = np.asarray(surface.xy_bounds, dtype=float).reshape(2, 2)
    return float(xlo + margin), float(ylo + margin), float(xhi - margin), float(yhi - margin)


def _surface_mode(scene, surface, grasp_name):
    xlo, ylo, xhi, yhi = _inset_bounds(surface, SUPPORT_MARGIN)
    zc = float(surface.height) + 0.5 * float(BOX_SIZE[2])
    return Mode(
        f"slide_{surface.name}_{grasp_name}",
        equalities=[
            PlaneConstraint(scene.robot, "payload", zc),
            FlatCarryConstraint(scene.robot, "payload", BOX_FLAT_R),
        ],
        inequalities=[
            JointLimit(scene.robot),
            WorkspaceBoundary(scene.robot, "payload", [xlo, ylo, -1.0e3], [xhi, yhi, 1.0e3]),
            _collision(scene),
            _support_torque(scene),
        ],
    )


def _liftoff_mode(scene, surface, grasp_name):
    xlo, ylo, xhi, yhi = _inset_bounds(surface, SUPPORT_MARGIN)
    zc = float(surface.height) + 0.5 * float(BOX_SIZE[2])
    return Mode(
        f"lift_{surface.name}_{grasp_name}",
        equalities=[
            PlaneConstraint(scene.robot, "payload", zc),
            FlatCarryConstraint(scene.robot, "payload", BOX_FLAT_R),
        ],
        inequalities=[
            JointLimit(scene.robot),
            WorkspaceBoundary(scene.robot, "payload", [xlo, ylo, -1.0e3], [xhi, yhi, 1.0e3]),
            _collision(scene),
            TorqueFeasibility(scene.robot, scene.tau_max, derate=scene.derate),
        ],
    )


def _free_mode(scene, grasp_name):
    return Mode(
        f"free_flight_{grasp_name}",
        equalities=[FlatCarryConstraint(scene.robot, "payload", BOX_FLAT_R)],
        inequalities=[
            JointLimit(scene.robot),
            TorqueFeasibility(scene.robot, scene.tau_max, derate=scene.derate),
            _collision(scene),
        ],
    )


def _transit_mode(scene):
    return Mode(
        "transit_empty_gripper",
        inequalities=[JointLimit(scene.robot), _collision(scene)],
    )


def _world_box(center, yaw_deg=0.0):
    T = make_T(rpy_to_matrix([0.0, 0.0, np.deg2rad(float(yaw_deg))]), center)
    T[2, 3] = TABLE_HEIGHT + 0.5 * float(BOX_SIZE[2])
    return T


def _box_on_surface(surface, x, y, yaw_deg=0.0):
    return _world_box([float(x), float(y), float(surface.height) + 0.5 * float(BOX_SIZE[2])], yaw_deg)


def _carry_pose(T_box):
    T = np.asarray(T_box, dtype=float).copy()
    T[:3, 3] += np.array([0.0, 0.0, CARRY_CLEARANCE], dtype=float)
    return T


def _project_pose(scene, mode, T_box, seeds, name, *, relaxed=True):
    qlo, qhi = scene.robot.joint_limits
    modes = [
        (0.0, 0.0, 1.0e-7, 180),
        (0.006, np.deg2rad(8.0), 1.0e-5, 140),
    ] if relaxed else [(0.0, 0.0, 1.0e-7, 180)]
    for pos_tol, rot_tol, project_tol, max_iter in modes:
        project_mode = Mode(
            name,
            equalities=[
                FrameConstraint(
                    scene.robot,
                    "payload",
                    target_position=np.asarray(T_box, dtype=float)[:3, 3],
                    pos_mask=(1, 1, 1),
                    pos_tol=[pos_tol] * 3,
                    target_rotation=np.asarray(T_box, dtype=float)[:3, :3],
                    rot_mask=(1, 1, 1),
                    rot_tol=[rot_tol] * 3,
                    name=f"{name}_payload",
                )
            ],
            inequalities=[JointLimit(scene.robot), *mode.inequalities],
        )
        for seed in seeds:
            q, ok = project_mode.project(seed, max_iter=max_iter, tol=project_tol, qlo=qlo, qhi=qhi)
            if ok and mode.feasible(q):
                return np.asarray(q, dtype=float), True
    return None, False


def _project_tcp(scene, mode, T_tcp, seeds, name):
    qlo, qhi = scene.robot.joint_limits
    project_mode = Mode(
        name,
        equalities=[
            FrameConstraint(
                scene.robot,
                TCP_FRAME,
                target_position=np.asarray(T_tcp, dtype=float)[:3, 3],
                pos_mask=(1, 1, 1),
                pos_tol=[0.006] * 3,
                target_rotation=np.asarray(T_tcp, dtype=float)[:3, :3],
                rot_mask=(1, 1, 1),
                rot_tol=np.deg2rad([8.0, 8.0, 12.0]),
                name=f"{name}_{TCP_FRAME}",
            )
        ],
        inequalities=[JointLimit(scene.robot), *mode.inequalities],
    )
    for seed in seeds:
        q, ok = project_mode.project(seed, max_iter=140, tol=1.0e-5, qlo=qlo, qhi=qhi)
        if ok and mode.feasible(q):
            return np.asarray(q, dtype=float), True
    return None, False


def _path_cost(scene, path, checker):
    path_len = _joint_path_length(path)
    torque = _max_torque_util(scene, path)
    clear = _clearance_penalty(checker, scene, path)
    return path_len, torque, clear, W_PATH * path_len + W_TORQUE * torque + W_CLEAR * clear


def _is_internal_error_text(text):
    low = str(text).lower()
    return (
        "internal error" in low
        or "backend failed" in low
        or "object does not support" in low
        or "ompl._ompl" in low
        or "traceback" in low
        or "attributeerror" in low
        or "typeerror" in low
    )


def _classify_reason(reason):
    text = str(reason)
    low = text.lower()
    if _is_internal_error_text(low):
        return "internal error"
    if "torque" in low or "util" in low:
        return "torque-infeasible"
    if "box penetrates table" in low or "box-table" in low:
        return "collision: box-table"
    if "robot-table" in low or ("collision" in low and "table" in low):
        return "collision: robot-table"
    if "robot-base" in low or "box-base" in low or ("collision" in low and "base" in low):
        return "collision: robot-base"
    if "collision" in low:
        return "collision"
    if "width" in low:
        return "grasp-width"
    if "from-placement" in low or "from placement" in low or "lift-off placement" in low:
        return "from-placement projection failed"
    if "to-placement" in low or "to placement" in low or "set-down projection" in low or "goal placement" in low:
        return "to-placement projection failed"
    if "start placement" in low:
        return "from-placement projection failed"
    if "projection" in low or "project" in low:
        return "projection failed"
    if "ompl" in low or "planner" in low or "path" in low:
        return "OMPL plan failed"
    if "mode" in low or "valid" in low or "feasible" in low:
        return "mode-infeasible"
    return "other"


def _failure(pair_index, combo, segment, phase, reason, progress):
    return Failure(
        pair_index=int(pair_index),
        combo=str(combo),
        segment=str(segment),
        phase=str(phase),
        cause=_classify_reason(reason),
        detail=str(reason),
        progress=int(progress),
    )


def _print_failure_summary(failures, *, evaluated_pairs, valid_count):
    print("failure diagnostics:")
    print(f"  evaluated_pairs={evaluated_pairs} valid_routes={valid_count} failures={len(failures)}")
    counts = {}
    by_segment = {}
    for f in failures:
        counts[f.cause] = counts.get(f.cause, 0) + 1
        key = (f.segment, f.cause)
        by_segment[key] = by_segment.get(key, 0) + 1
    if counts:
        print("  by cause:")
        for cause, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {cause:28s} {count:5d}")
        print("  by segment/cause:")
        for (segment, cause), count in sorted(by_segment.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {segment:16s} {cause:28s} {count:5d}")
    else:
        print("  no failures recorded")
    if failures:
        best = max(failures, key=lambda f: f.progress)
        print("  best partial progress:")
        print(f"    stage={best.progress} {best.line()}")
        print("  first failure examples:")
        for f in failures[:20]:
            print(f"    {f.line()}")


def _print_plain_fallback_diagnostic(failures, valid_count):
    print("plain slide/lift fallback:")
    print(f"  valid_routes={valid_count} failures={len(failures)}")
    torque_failures = [f for f in failures if f.cause == "torque-infeasible"]
    if torque_failures:
        best = max(torque_failures, key=lambda f: f.progress)
        print(f"  torque-limited example: {best.line()}")
    elif failures:
        best = max(failures, key=lambda f: f.progress)
        print(f"  best plain partial progress: stage={best.progress} {best.line()}")


def _max_torque_util(scene, path):
    tau = np.asarray(scene.tau_max, dtype=float) * float(scene.derate)
    vals = [float(np.max(np.abs(scene.robot.gravity(q)) / tau)) for q in np.asarray(path, dtype=float)]
    return float(max(vals)) if vals else 0.0


def _estimate_endpoint_util(scene, qs):
    try:
        return _max_torque_util(scene, np.asarray(qs, dtype=float))
    except Exception:
        return float("nan")


def _endpoint_torque_detail(scene, qs):
    try:
        tau = np.asarray(scene.tau_max, dtype=float) * float(scene.derate)
        best_util = -np.inf
        best_joint = -1
        for q in np.asarray(qs, dtype=float):
            util = np.abs(scene.robot.gravity(q)) / tau
            j = int(np.argmax(util))
            if float(util[j]) > best_util:
                best_util = float(util[j])
                best_joint = j + 1
        return f"max_torque_util={best_util:.3f} at joint {best_joint}"
    except Exception:
        return "max_torque_util=nan"


def _clearance_penalty(checker, scene, path):
    penalty = 0.0
    for q in np.asarray(path, dtype=float)[:: max(1, len(path) // 20)]:
        for frame in ("fer_link6", "fer_link7", "fer_link8", "fer_hand", "fer_hand_tcp"):
            try:
                p = np.asarray(scene.robot.frame_pose(frame, q)[:3, 3], dtype=float)
            except Exception:
                continue
            if checker._point_near_table_xy(p, margin=0.08):
                penalty += max(0.0, 0.05 - float(p[2] - TABLE_HEIGHT))
    return float(penalty)


def _make_segment(kind, grasp, scene, mode, q0, q1, checker, T_static=None, switch=0.0):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    res = plan_in_mode(
        scene.robot,
        mode,
        q0,
        q1,
        planner_name=TRANSFER_PLANNER,
        solve_time=TRANSFER_SOLVE_TIME,
        delta=TRANSFER_DELTA,
        space=TRANSFER_SPACE,
    )
    if not res.success or res.path is None:
        return None, str(res.message)
    path = np.asarray(res.path, dtype=float)
    if not _path_valid(mode, path):
        return None, "simplified path failed mode validity"
    path_len, torque, clear, cost = _path_cost(scene, path, checker)
    cost += float(switch)
    return Segment(
        kind=kind,
        grasp=grasp,
        scene=scene,
        mode=mode,
        path=path,
        q_start=np.asarray(q0, dtype=float),
        q_goal=np.asarray(q1, dtype=float),
        T_static=None if T_static is None else np.asarray(T_static, dtype=float),
        cost=cost,
        path_len=path_len,
        raw_path_len=float(getattr(res, "raw_report_path_length", np.nan)),
        ompl_path_len=float(getattr(res, "ompl_simplified_path_length", path_len)),
        torque=torque,
        clear=clear,
        switch=float(switch),
    ), "ok"


def _sample_transitions(surface, rng, n):
    xlo, ylo, xhi, yhi = _inset_bounds(surface, SUPPORT_MARGIN)
    out = []
    other_y = -0.46 if surface.name == "A" else 0.39
    for _ in range(max(0, int(n))):
        x = float(rng.uniform(xlo, xhi))
        edge_y = ylo if abs(ylo - other_y) < abs(yhi - other_y) else yhi
        y = float(np.clip(edge_y + rng.normal(0.0, 0.08), ylo, yhi))
        out.append(_box_on_surface(surface, x, y, REGRASP_TRANSPORT_YAW_DEG))
    return out


def _feasible_grasps():
    return [g for g in grasp_library() if g.required_width <= 2.0 * FINGER_MAX + 1.0e-9]


def _plan_regrasp(T_box, q_from, grasp_from, grasp_to, checker):
    if grasp_from.name == grasp_to.name:
        scene = _held_scene(grasp_to)
        mode = _liftoff_mode(scene, scene.surface("A"), grasp_to.name)
        return [], q_from, 0.0, "same grasp"
    scene_to = _held_scene(grasp_to)
    empty = _empty_scene()
    mode_t = _transit_mode(empty)
    T_tcp_to = np.asarray(T_box, dtype=float) @ np.asarray(grasp_to.T_box_tcp, dtype=float)
    q_to, ok = _project_tcp(empty, mode_t, T_tcp_to, _seed_list(empty, [q_from]), f"regrasp_to_{grasp_to.name}")
    if not ok:
        return None, None, 0.0, "regrasp transit projection failed"
    transit, reason = _make_segment(
        "transit_regrasp",
        None,
        empty,
        mode_t,
        q_from,
        q_to,
        checker,
        T_static=T_box,
        switch=SWITCH_PENALTY,
    )
    if transit is None:
        return None, None, 0.0, reason
    # q_to is also the seed for the next held-scene projection at this placement.
    return [transit], q_to, transit.cost, "ok"


def _build_candidate(pair, grasps, checker, pair_index=0, failures=None, *, no_regrasp_only=False):
    T_start, T_lift, T_set, T_goal = pair
    candidates = []
    combos = 0
    for g_a in grasps:
        combo_a = g_a.name
        scene_a = _held_scene(g_a)
        surf_a = scene_a.surface("A")
        mode_a = _surface_mode(scene_a, surf_a, g_a.name)
        mode_lift_a = _liftoff_mode(scene_a, surf_a, g_a.name)
        q_start, ok = _project_pose(scene_a, mode_a, T_start, _seed_list(scene_a), f"start_{g_a.name}")
        if not ok:
            if failures is not None:
                failures.append(_failure(pair_index, combo_a, "slide_A", "start", "start placement projection failed", 1))
            continue
        q_lift_a, ok = _project_pose(scene_a, mode_lift_a, T_lift, _seed_list(scene_a, [q_start]), f"lift_A_{g_a.name}")
        if not ok:
            if failures is not None:
                failures.append(_failure(pair_index, combo_a, "lift", "from-placement", "lift-off placement projection failed", 2))
            continue
        slide_a, reason = _make_segment("slide_A", g_a, scene_a, mode_a, q_start, q_lift_a, checker)
        if slide_a is None:
            if failures is not None:
                failures.append(_failure(pair_index, combo_a, "slide_A", "plan", f"OMPL plan failed: {reason}", 3))
            continue
        for g_f in grasps:
            if no_regrasp_only and g_f.name != g_a.name:
                continue
            if combos >= COMBO_BUDGET:
                break
            combo_af = f"{g_a.name}>{g_f.name}"
            re_a, q_seed_f, re_cost_a, reason = _plan_regrasp(T_lift, q_lift_a, g_a, g_f, checker)
            if re_a is None:
                if failures is not None:
                    failures.append(_failure(pair_index, combo_af, "transit_regrasp", "lift-transition", reason, 4))
                continue
            scene_f = _held_scene(g_f)
            surf_a_f = scene_f.surface("A")
            surf_b_f = scene_f.surface("B")
            mode_lift_f = _liftoff_mode(scene_f, surf_a_f, g_f.name)
            mode_set_f = _liftoff_mode(scene_f, surf_b_f, g_f.name)
            mode_free = _free_mode(scene_f, g_f.name)
            q_lift_f, ok = _project_pose(scene_f, mode_lift_f, T_lift, _seed_list(scene_f, [q_seed_f]), f"lift_A_free_{g_f.name}")
            if not ok:
                if failures is not None:
                    failures.append(_failure(pair_index, combo_af, "lift", "from-placement", "free-flight lift projection failed", 5))
                continue
            q_set_f, ok = _project_pose(scene_f, mode_set_f, T_set, _seed_list(scene_f, [q_lift_f]), f"set_B_free_{g_f.name}")
            if not ok:
                if failures is not None:
                    failures.append(_failure(pair_index, combo_af, "set", "to-placement", "free-flight set-down projection failed", 6))
                continue
            free, reason = _make_segment("free_flight", g_f, scene_f, mode_free, q_lift_f, q_set_f, checker)
            if free is None:
                if failures is not None:
                    detail = f"OMPL plan failed: {reason}; {_endpoint_torque_detail(scene_f, [q_lift_f, q_set_f])}"
                    failures.append(_failure(pair_index, combo_af, "free", "plan", detail, 7))
                continue
            for g_b in grasps:
                if no_regrasp_only and g_b.name != g_f.name:
                    continue
                combos += 1
                combo = f"{g_a.name}>{g_f.name}>{g_b.name}"
                re_b, q_seed_b, re_cost_b, reason = _plan_regrasp(T_set, q_set_f, g_f, g_b, checker)
                if re_b is None:
                    if failures is not None:
                        failures.append(_failure(pair_index, combo, "transit_regrasp", "set-transition", reason, 8))
                    continue
                scene_b = _held_scene(g_b)
                surf_b = scene_b.surface("B")
                mode_b = _surface_mode(scene_b, surf_b, g_b.name)
                mode_set_b = _liftoff_mode(scene_b, surf_b, g_b.name)
                q_set_b, ok = _project_pose(scene_b, mode_set_b, T_set, _seed_list(scene_b, [q_seed_b]), f"set_B_{g_b.name}")
                if not ok:
                    if failures is not None:
                        failures.append(_failure(pair_index, combo, "set", "to-placement", "slide-B set-down projection failed", 9))
                    continue
                q_goal, ok = _project_pose(scene_b, mode_b, T_goal, _seed_list(scene_b, [q_set_b]), f"goal_{g_b.name}")
                if not ok:
                    if failures is not None:
                        failures.append(_failure(pair_index, combo, "slide_B", "goal", "goal placement projection failed", 10))
                    continue
                slide_b, reason = _make_segment("slide_B", g_b, scene_b, mode_b, q_set_b, q_goal, checker)
                if slide_b is None:
                    if failures is not None:
                        failures.append(_failure(pair_index, combo, "slide_B", "plan", f"OMPL plan failed: {reason}", 11))
                    continue
                segments = [slide_a, *re_a, free, *re_b, slide_b]
                total = float(sum(seg.cost for seg in segments))
                candidates.append(
                    {
                        "segments": segments,
                        "grasps": [g_a, g_f, g_b],
                        "placements": [T_start, T_lift, T_set, T_goal],
                        "cost": total,
                    }
                )
    return candidates


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


def _append_ramp(F, q, T_box, scene, mode, grasp, w0, w1, label):
    for w in np.linspace(float(w0), float(w1), max(2, OPEN_RAMP_STEPS)):
        F.add(q, w, False, T_box, "resting", label, scene, mode, grasp)


def _frames(route):
    F = _Frames()
    held = None
    for seg in route["segments"]:
        if seg.kind == "transit_regrasp":
            if held is not None:
                _append_ramp(F, seg.q_start, seg.T_static, seg.scene, seg.mode, held, _hold_finger_width(held), OPEN_W, f"release_{held.name}")
            for q in seg.path:
                F.add(q, OPEN_W, False, seg.T_static, "resting", "transit_regrasp", seg.scene, seg.mode, None)
            held = None
            continue
        grasp = seg.grasp
        if held is None or held.name != grasp.name:
            T_static = seg.T_static if seg.T_static is not None else np.asarray(seg.scene.robot.frame_pose("payload", seg.q_start), dtype=float)
            _append_ramp(F, seg.q_start, T_static, seg.scene, seg.mode, grasp, OPEN_W, _hold_finger_width(grasp), f"close_{grasp.name}")
            F.add(seg.q_start, _hold_finger_width(grasp), True, np.asarray(seg.scene.robot.frame_pose("payload", seg.q_start), dtype=float), "attached", f"grasp_{grasp.name}", seg.scene, seg.mode, grasp)
        width = _hold_finger_width(grasp)
        for q in seg.path:
            T_box = np.asarray(seg.scene.robot.frame_pose("payload", q), dtype=float)
            F.add(q, width, True, T_box, "attached", seg.kind, seg.scene, seg.mode, grasp)
        held = grasp
    if held is not None and route["segments"]:
        last = route["segments"][-1]
        T_final = route["placements"][-1]
        _append_ramp(F, last.q_goal, T_final, last.scene, last.mode, held, _hold_finger_width(held), OPEN_W, f"release_{held.name}")
    return F


def _audit_path(F):
    reasons = []
    for i, (q, mode, label) in enumerate(zip(F.q, F.modes, F.labels)):
        if mode is not None and not mode.valid(q, tol=1.0e-4):
            reasons.append(f"sample {i}, phase={label}: mode {mode.name} invalid")
            if len(reasons) >= 8:
                break
    return len(reasons) == 0, reasons


def _route_ok(route, checker):
    F = _frames(route)
    path_ok, path_reasons = _audit_path(F)
    grasp_ok, grasp_reasons = _grasp_relation_audit(F)
    collision_ok, collision_reasons = _collision_audit(F, checker)
    return bool(path_ok and grasp_ok and collision_ok), (path_reasons + grasp_reasons + collision_reasons), F


def _save(F, route, path_ok, collision_ok):
    segment_costs = [seg.cost for seg in route["segments"]]
    np.savez(
        OUT,
        plan_version=np.asarray("regrasp_transport_v1"),
        plan_valid=np.asarray(bool(path_ok and collision_ok)),
        path_audit_pass=np.asarray(bool(path_ok)),
        collision_audit_pass=np.asarray(bool(collision_ok)),
        path=np.asarray(F.q, dtype=float),
        labels=np.asarray(F.labels).astype(str),
        finger_widths=np.asarray(F.w, dtype=float),
        object_attached=np.asarray(F.attached, dtype=bool),
        contact_reached=np.asarray(F.attached, dtype=bool),
        box_poses=np.asarray(F.box, dtype=float),
        box_pose_state=np.asarray(F.state).astype(str),
        box_size=np.asarray(BOX_SIZE, dtype=float),
        T_world_box_initial=route["placements"][0],
        T_world_box_target=route["placements"][-1],
        table_xy_bounds=_combined_bounds(),
        table_surfaces_bounds=np.asarray([TABLE_A["xy_bounds"], TABLE_B["xy_bounds"]], dtype=float),
        table_surfaces_heights=np.asarray([TABLE_A["height"], TABLE_B["height"]], dtype=float),
        table_height=np.asarray(float(TABLE_HEIGHT)),
        selected_grasp=np.asarray("transport:" + ">".join(g.name for g in route["grasps"])),
        selected_grasp_sequence=np.asarray([g.name for g in route["grasps"]]),
        selected_segment_kinds=np.asarray([seg.kind for seg in route["segments"]]),
        segment_costs=np.asarray(segment_costs, dtype=float),
        segment_path_lengths=np.asarray([seg.path_len for seg in route["segments"]], dtype=float),
        segment_raw_path_lengths=np.asarray([seg.raw_path_len for seg in route["segments"]], dtype=float),
        segment_ompl_path_lengths=np.asarray([seg.ompl_path_len for seg in route["segments"]], dtype=float),
        segment_torque_utils=np.asarray([seg.torque for seg in route["segments"]], dtype=float),
        total_cost=np.asarray(float(route["cost"])),
        gripper_hold_finger_width=np.asarray(_hold_finger_width(route["grasps"][0])),
    )


def main():
    print("box regrasp transport planner (simulation only)")
    print("method: transport.py slide/lift transition search + grasp/regrasp dimension")
    print(f"box mass={box_payload.BOX_MASS:.3f} kg")
    print(f"surfaces: A={TABLE_A['xy_bounds']} B={TABLE_B['xy_bounds']}")
    print(f"start={START_CENTER.tolist()} goal={GOAL_CENTER.tolist()}")
    grasps = _feasible_grasps()
    print(f"candidate grasps={[g.name for g in grasps]}")

    audit_scene = _empty_scene()
    checker = BoxDemoCollisionChecker(
        box_size=BOX_SIZE,
        table_xy_bounds=_surface_bounds(),
        table_height=TABLE_HEIGHT,
        scene=audit_scene,
        table_thickness=TABLE_THICKNESS,
        box_table_tol=float(os.environ.get("BOX_DEMO_BOX_TABLE_TOL", "0.0015")),
    )
    rng = np.random.default_rng(20260627)
    T_start = _world_box(START_CENTER, REGRASP_TRANSPORT_YAW_DEG)
    T_goal = _world_box(GOAL_CENTER, REGRASP_TRANSPORT_YAW_DEG)
    scene0 = _held_scene(grasps[0])
    surface_a = scene0.surface("A")
    surface_b = scene0.surface("B")
    lifts = _sample_transitions(surface_a, rng, N_TRANSITIONS)
    sets = _sample_transitions(surface_b, rng, N_TRANSITIONS)
    pairs = []
    for i, T_lift in enumerate(lifts):
        for j, T_set in enumerate(sets):
            pairs.append((float(np.linalg.norm(T_lift[:3, 3] - T_set[:3, 3])), i, j))
    pairs.sort(key=lambda item: item[0])
    print(f"sampled transitions: lift={len(lifts)} set={len(sets)} candidate_pairs={len(pairs)}")

    candidates = []
    failures = []
    considered_pairs = pairs[: max(1, min(PAIR_BUDGET, len(pairs)))]
    for pair_index, (_, i, j) in enumerate(considered_pairs):
        pair = (T_start, lifts[i], sets[j], T_goal)
        for route in _build_candidate(pair, grasps, checker, pair_index, failures):
            ok, reasons, _ = _route_ok(route, checker)
            if ok:
                candidates.append(route)
            elif reasons:
                combo = ">".join(g.name for g in route["grasps"])
                failures.append(_failure(pair_index, combo, "audit", "certify", reasons[0], 12))
    _print_failure_summary(failures, evaluated_pairs=len(considered_pairs), valid_count=len(candidates))

    if not candidates:
        print("No regrasp-enabled route survived; trying plain slide/lift fallback (same grasp throughout).")
        plain_candidates = []
        plain_failures = []
        for pair_index, (_, i, j) in enumerate(considered_pairs):
            pair = (T_start, lifts[i], sets[j], T_goal)
            for route in _build_candidate(
                pair,
                grasps,
                checker,
                pair_index,
                plain_failures,
                no_regrasp_only=True,
            ):
                ok, reasons, _ = _route_ok(route, checker)
                if ok:
                    plain_candidates.append(route)
                elif reasons:
                    combo = ">".join(g.name for g in route["grasps"])
                    plain_failures.append(_failure(pair_index, combo, "audit", "certify", reasons[0], 12))
        _print_failure_summary(plain_failures, evaluated_pairs=len(considered_pairs), valid_count=len(plain_candidates))
        _print_plain_fallback_diagnostic(plain_failures, len(plain_candidates))
        candidates = plain_candidates
        failures = plain_failures
    if not candidates:
        print("No valid route survived.")
        return 1

    candidates.sort(key=lambda r: r["cost"])
    selected = candidates[0]
    print("selected sequence:")
    print("  grasps:", ">".join(g.name for g in selected["grasps"]))
    print("  modes :", " -> ".join(seg.kind for seg in selected["segments"]))
    for idx, seg in enumerate(selected["segments"]):
        gname = "empty" if seg.grasp is None else seg.grasp.name
        print(
            f"  segment {idx}: {seg.kind:16s} grasp={gname:14s} "
            f"cost={seg.cost:.4f} raw={seg.raw_path_len:.4f} "
            f"ompl={seg.ompl_path_len:.4f} final={seg.path_len:.4f} "
            f"torque={seg.torque:.4f} clear={seg.clear:.4f} switch={seg.switch:.4f}"
        )
    F = _frames(selected)
    _print_phase_summary(F)
    selftest = _collision_audit_self_test(F, checker)
    print("collision audit self-test:")
    print("  normal grasp contact:", "PASS" if selftest["normal_ok"] else f"FAIL ({selftest['normal_reason']})")
    print("  box 5mm into table:", "PASS" if selftest["table_rejected"] else f"FAIL ({selftest['table_reason']})")
    print("  fer_link6 proxy 5mm below table:", "PASS" if selftest["arm_rejected"] else f"FAIL ({selftest['arm_reason']})")
    path_ok, path_reasons = _audit_path(F)
    grasp_ok, grasp_reasons = _grasp_relation_audit(F)
    collision_ok, collision_reasons = _collision_audit(F, checker)
    selftest_ok = bool(selftest["normal_ok"] and selftest["table_rejected"] and selftest["arm_rejected"])
    path_ok = bool(path_ok and grasp_ok)
    collision_ok = bool(collision_ok and selftest_ok)
    print("path audit:", "PASS" if path_ok else "FAIL")
    for reason in (path_reasons + grasp_reasons)[:8]:
        print("  ", reason)
    print("collision audit:", "PASS" if collision_ok else "FAIL")
    for reason in collision_reasons[:8]:
        print("  ", reason)
    _save(F, selected, path_ok, collision_ok)
    if not (path_ok and collision_ok):
        print(f"saved invalid debug plan -> {OUT}")
        return 1
    print(f"saved valid regrasp transport plan -> {OUT}")
    print("animate with: python -B animate_regrasp.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
