from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scene import Scene
from constraints import FrameConstraint, JointLimit, Mode
from run_panda import CFG
from box_demo_collision import BoxDemoCollisionChecker, BoxDemoCollisionError
from box_grasp_library import (
    BOX_CENTER_WORLD,
    BOX_SIZE,
    BOX_TARGET_CENTER_WORLD,
    CARRY_HEIGHT,
    FINGER_MAX,
    LIFT_HEIGHT,
    TABLE_HEIGHT,
    TABLE_XY_BOUNDS,
    GraspCandidate,
    T_world_box,
    T_world_box_target,
    grasp_library,
    sampled_T_box_tcp,
)


HERE = Path(__file__).resolve().parent
OUT = HERE / "box_grasp_tsr_plan.npz"
BASE_RENDER_URDF = HERE / "fer_franka_hand_render.urdf"
GRIP_RENDER_URDF = HERE / "fer_franka_hand_render_prismatic.urdf"
TCP_FRAME = "fer_hand_tcp"
FINGER_JOINTS = ["fer_finger_joint1", "fer_finger_joint2"]
TORQUE_WEIGHT = float(os.environ.get("BOX_GRASP_TORQUE_WEIGHT", "0.25"))
WIDTH_WEIGHT = float(os.environ.get("BOX_GRASP_WIDTH_WEIGHT", "0.10"))
SIDE_GRASP_BONUS = float(os.environ.get("BOX_GRASP_SIDE_BONUS", "0.00"))
CENTERING_WEIGHT = float(os.environ.get("BOX_GRASP_CENTERING_WEIGHT", "20.0"))
MAX_TOP_CENTERING_ERROR = float(os.environ.get("BOX_GRASP_MAX_TOP_CENTERING_ERROR", "0.004"))
GRASP_POS_TOL_STRICT = float(os.environ.get("BOX_GRASP_STRICT_POS_TOL", "0.005"))
GRASP_ROT_TOL_STRICT_DEG = float(os.environ.get("BOX_GRASP_STRICT_ROT_TOL_DEG", "6.0"))
PLACE_POS_TOL_STRICT = float(os.environ.get("BOX_PLACE_STRICT_POS_TOL", "0.005"))
PLACE_ROT_TOL_STRICT_DEG = float(os.environ.get("BOX_PLACE_STRICT_ROT_TOL_DEG", "6.0"))
ATTACH_AUDIT_POS_WARN = float(os.environ.get("BOX_ATTACH_AUDIT_POS_WARN", "0.010"))
ATTACH_AUDIT_ROT_WARN_DEG = float(os.environ.get("BOX_ATTACH_AUDIT_ROT_WARN_DEG", "8.0"))
TABLE_THICKNESS = float(os.environ.get("BOX_DEMO_TABLE_THICKNESS", "0.025"))
TABLE_CLEARANCE = float(os.environ.get("BOX_DEMO_TABLE_CLEARANCE", "0.002"))
BOX_TABLE_TOL = float(os.environ.get("BOX_DEMO_BOX_TABLE_TOL", "0.003"))
ROBOT_TABLE_CLEARANCE = float(os.environ.get("BOX_DEMO_ROBOT_TABLE_CLEARANCE", "0.012"))
GRASP_CONTACT_POS_MAX = float(os.environ.get("BOX_DEMO_GRASP_CONTACT_POS_MAX", "0.008"))
GRASP_CONTACT_ROT_MAX_DEG = float(os.environ.get("BOX_DEMO_GRASP_CONTACT_ROT_MAX_DEG", "8.0"))
BOX_DEMO_USE_OMPL = os.environ.get("BOX_DEMO_USE_OMPL", "1") not in {
    "0",
    "false",
    "False",
    "no",
    "NO",
}
BOX_DEMO_OMPL_SOLVE_TIME = float(os.environ.get("BOX_DEMO_OMPL_SOLVE_TIME", "3.0"))
BOX_DEMO_OMPL_DELTA = float(os.environ.get("BOX_DEMO_OMPL_DELTA", "0.08"))
OMPL_FALLBACK_WIRED = False
ROBOT_TABLE_CHECK_FRAMES = [
    "fer_link6",
    "fer_link7",
    "fer_link8",
    "fer_hand",
    "fer_hand_tcp",
    "fer_leftfinger",
    "fer_rightfinger",
]


@dataclass
class CandidateResult:
    grasp: GraspCandidate
    success: bool
    q_pre: np.ndarray | None
    q_grasp: np.ndarray | None
    q_lift: np.ndarray | None
    score: float
    reason: str
    max_torque_util: float | None = None
    joint_length: float | None = None
    sample_id: int | None = None
    T_box_tcp: np.ndarray | None = None
    q_carry: np.ndarray | None = None
    q_place: np.ndarray | None = None
    centering_error: float | None = None
    grasp_pos_err: float | None = None
    grasp_rot_err_deg: float | None = None
    lift_pos_err: float | None = None
    lift_rot_err_deg: float | None = None
    carry_pos_err: float | None = None
    carry_rot_err_deg: float | None = None
    place_pos_err: float | None = None
    place_rot_err_deg: float | None = None


def make_prismatic_render_urdf(src: Path, dst: Path) -> Path:
    tree = ET.parse(src)
    root = tree.getroot()
    switched = []
    for joint in root.findall("joint"):
        if joint.attrib.get("name") in FINGER_JOINTS:
            joint.set("type", "prismatic")
            if joint.find("axis") is None:
                ET.SubElement(joint, "axis", {"xyz": "0 1 0"})
            if joint.find("limit") is None:
                ET.SubElement(
                    joint,
                    "limit",
                    {
                        "effort": "100",
                        "lower": "0.0",
                        "upper": "0.04",
                        "velocity": "0.2",
                    },
                )
            switched.append(joint.attrib.get("name"))
    if len(switched) != len(FINGER_JOINTS):
        missing = set(FINGER_JOINTS) - set(switched)
        raise RuntimeError(f"finger joints not found in URDF: {sorted(missing)}")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def _make_collision_checker(scene):
    return BoxDemoCollisionChecker(
        box_size=BOX_SIZE,
        table_xy_bounds=TABLE_XY_BOUNDS,
        table_height=TABLE_HEIGHT,
        robot_urdf=None,
        scene_robot=scene.robot,
        table_thickness=TABLE_THICKNESS,
        robot_table_margin=float(os.environ.get("BOX_DEMO_ROBOT_TABLE_MARGIN", "0.002")),
        box_table_tol=float(os.environ.get("BOX_DEMO_BOX_TABLE_TOL", "0.001")),
    )


def _pybullet_audit_path(checker, path, finger_widths, box_poses, box_pose_state, labels):
    violations = []
    for i, (q, w, T_box, state, label) in enumerate(
        zip(path, finger_widths, box_poses, box_pose_state, labels)
    ):
        ok, reason = checker.sample_valid(q, w, T_box, str(state), str(label))
        if not ok:
            violations.append((i, str(label), reason))
    return violations


def _print_pybullet_audit(violations):
    print("collision audit:")
    print(f"  violations: {len(violations)}")
    for i, label, reason in violations[:10]:
        print(f"  sample {i}, phase={label}: {reason}")


def _move_stale_plan():
    if OUT.exists():
        bad = OUT.with_suffix(".failed_or_stale.npz")
        OUT.replace(bad)
        print(f"Moved stale previous plan to: {bad}")


def _make_scene():
    # Copy CFG so this demo cannot mutate the working payload planner config.
    cfg = dict(CFG)
    return Scene(cfg)


def _seed_set(scene):
    qlo, qhi = scene.robot.joint_limits
    qlo = np.asarray(qlo, dtype=float)
    qhi = np.asarray(qhi, dtype=float)
    rng = np.random.default_rng(20260623)
    seeds = [0.5 * (qlo + qhi)]
    ready = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
    if ready.shape[0] == scene.robot.nq:
        seeds.append(np.minimum(np.maximum(ready, qlo), qhi))
        seeds.extend(
            [
                np.minimum(np.maximum(ready + delta, qlo), qhi)
                for delta in (
                    np.array([0.0, -0.20, 0.0, -0.20, 0.0, 0.20, 0.0]),
                    np.array([0.25, -0.10, -0.25, 0.15, 0.20, 0.0, -0.15]),
                    np.array([-0.25, 0.10, 0.25, -0.15, -0.20, 0.0, 0.15]),
                )
            ]
        )
    seeds.extend(rng.uniform(qlo, qhi) for _ in range(6))
    return seeds


def _project_tcp_pose(scene, T_target, seed, name, *, exact=True):
    if exact:
        pos_tol = [0.0, 0.0, 0.0]
        rot_tol = [0.0, 0.0, 0.0]
        project_tol = 1.0e-7
        max_iter = 160
    else:
        pos_tol = [0.004, 0.004, 0.004]
        rot_tol = np.deg2rad([6.0, 6.0, 6.0])
        project_tol = 1.0e-5
        max_iter = 100
    mode = Mode(
        name,
        equalities=[
            FrameConstraint(
                scene.robot,
                TCP_FRAME,
                target_position=T_target[:3, 3],
                pos_mask=(1, 1, 1),
                pos_tol=pos_tol,
                target_rotation=T_target[:3, :3],
                rot_mask=(1, 1, 1),
                rot_tol=rot_tol,
                name=f"{name}_tcp_pose",
            )
        ],
        inequalities=[JointLimit(scene.robot)],
    )
    qlo, qhi = scene.robot.joint_limits
    q, ok = mode.project(seed, max_iter=max_iter, tol=project_tol, qlo=qlo, qhi=qhi)
    if not ok:
        return None, False
    if not mode.feasible(q):
        return None, False
    return np.asarray(q, dtype=float), True


def _try_project(scene, T_target, seeds, name, *, exact=True):
    for seed in seeds:
        q, ok = _project_tcp_pose(scene, T_target, seed, name, exact=exact)
        if ok and q is not None:
            return q, True
    return None, False


def _polish_tcp_pose(scene, q_seed, T_target, name):
    q = np.asarray(q_seed, dtype=float)
    for _ in range(3):
        q2, ok = _project_tcp_pose(scene, T_target, q, name + "_polish", exact=True)
        if not ok or q2 is None:
            break
        q = q2
    return q


def _finger_from_total_gap(gap: float) -> float:
    return 0.5 * float(gap)


def _total_gap_from_finger(finger_w: float) -> float:
    return 2.0 * float(finger_w)


def _contact_gap_for_grasp(grasp) -> float:
    return float(grasp.object_width + 2.0 * grasp.contact_margin)


def _open_finger_width() -> float:
    return float(FINGER_MAX)


def _contact_finger_width(grasp) -> float:
    gap = _contact_gap_for_grasp(grasp)
    return float(np.clip(_finger_from_total_gap(gap), 0.0, FINGER_MAX))


def _rotation_error_angle(R_actual, R_target) -> float:
    R_actual = np.asarray(R_actual, dtype=float).reshape(3, 3)
    R_target = np.asarray(R_target, dtype=float).reshape(3, 3)
    R_err = R_actual @ R_target.T
    c = 0.5 * (np.trace(R_err) - 1.0)
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.arccos(c))


def _pose_error(scene, q, T_target):
    T_actual = np.asarray(scene.robot.frame_pose(TCP_FRAME, q), dtype=float)
    T_target = np.asarray(T_target, dtype=float)
    pos_err = float(np.linalg.norm(T_actual[:3, 3] - T_target[:3, 3]))
    rot_err = _rotation_error_angle(T_actual[:3, :3], T_target[:3, :3])
    return pos_err, rot_err


def _axis_angle_deg(a, b):
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    a = a / max(np.linalg.norm(a), 1.0e-12)
    b = b / max(np.linalg.norm(b), 1.0e-12)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(c)))


def _print_grasp_alignment(scene, q, T_target, label):
    T_actual = np.asarray(scene.robot.frame_pose(TCP_FRAME, q), dtype=float)
    T_target = np.asarray(T_target, dtype=float)
    pos_err, rot_err = _pose_error(scene, q, T_target)
    ax = T_actual[:3, 0]
    ay = T_actual[:3, 1]
    az = T_actual[:3, 2]
    tx = T_target[:3, 0]
    ty = T_target[:3, 1]
    tz = T_target[:3, 2]

    print(f"{label} pose audit:")
    print(f"  pos_err={pos_err:.5f} m, rot_err={np.rad2deg(rot_err):.3f} deg")
    print(f"  TCP x-axis angle to target x-axis: {_axis_angle_deg(ax, tx):.2f} deg")
    print(f"  TCP y-axis angle to target y-axis: {_axis_angle_deg(ay, ty):.2f} deg")
    print(f"  TCP z-axis angle to target z-axis: {_axis_angle_deg(az, tz):.2f} deg")
    print(f"  TCP z-axis angle to world down: {_axis_angle_deg(az, [0, 0, -1]):.2f} deg")


def _grasp_centering_error(grasp, T_box_tcp):
    """Return how badly the grasp is shifted along the finger closing direction."""
    T_box_tcp = np.asarray(T_box_tcp, dtype=float)
    if grasp.name == "top":
        return abs(float(T_box_tcp[1, 3]))
    return 0.0


def _smoothstep(s):
    s = np.asarray(s, dtype=float)
    return s * s * (3.0 - 2.0 * s)


def _interp(a, b, n):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ss = _smoothstep(np.linspace(0.0, 1.0, int(n)))
    return np.asarray([(1.0 - s) * a + s * b for s in ss], dtype=float)


def _interp_pose_translation(T_a, T_b, n):
    T_a = np.asarray(T_a, dtype=float)
    T_b = np.asarray(T_b, dtype=float)
    ss = _smoothstep(np.linspace(0.0, 1.0, int(n)))
    out = []
    for s in ss:
        T = np.eye(4, dtype=float)
        T[:3, :3] = T_a[:3, :3]
        T[:3, 3] = (1.0 - s) * T_a[:3, 3] + s * T_b[:3, 3]
        out.append(T)
    return np.asarray(out, dtype=float)


def _repeat_pose(T, n):
    return np.repeat(np.asarray(T, dtype=float).reshape(1, 4, 4), int(n), axis=0)


def _box_pose_from_tcp(scene, q, T_box_tcp):
    T_tcp = np.asarray(scene.robot.frame_pose(TCP_FRAME, q), dtype=float)
    return T_tcp @ np.linalg.inv(np.asarray(T_box_tcp, dtype=float))


def _box_corners_local():
    half = 0.5 * np.asarray(BOX_SIZE, dtype=float)
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append([sx * half[0], sy * half[1], sz * half[2]])
    pts.append([0.0, 0.0, 0.0])
    return np.asarray(pts, dtype=float)


def _transform_points(T, pts):
    T = np.asarray(T, dtype=float)
    pts = np.asarray(pts, dtype=float).reshape((-1, 3))
    return np.asarray([T[:3, 3] + T[:3, :3] @ p for p in pts], dtype=float)


def _inside_table_xy(p, margin=0.0):
    (xlo, ylo), (xhi, yhi) = np.asarray(TABLE_XY_BOUNDS, dtype=float).reshape(2, 2)
    return (
        xlo + margin <= p[0] <= xhi - margin
        and ylo + margin <= p[1] <= yhi - margin
    )


def _box_table_valid(T_box, *, state, phase):
    pts = _transform_points(T_box, _box_corners_local())
    z_min = float(np.min(pts[:, 2]))
    center = np.asarray(T_box[:3, 3], dtype=float)

    if z_min < TABLE_HEIGHT - BOX_TABLE_TOL:
        return False, f"box penetrates table: z_min={z_min:.4f}, table={TABLE_HEIGHT:.4f}, phase={phase}"

    if state in {"initial", "target", "release", "resting"}:
        if abs(z_min - TABLE_HEIGHT) > 0.006:
            return False, f"resting box not on table: z_min={z_min:.4f}, table={TABLE_HEIGHT:.4f}, phase={phase}"

    if not _inside_table_xy(center, margin=0.005):
        return False, f"box center outside safe table area: center={center}, phase={phase}"

    return True, "ok"


def _robot_table_valid(scene, q, *, phase):
    hits = []
    for frame in ROBOT_TABLE_CHECK_FRAMES:
        try:
            p = np.asarray(scene.robot.frame_pose(frame, q)[:3, 3], dtype=float)
        except Exception:
            continue
        if _inside_table_xy(p, margin=-0.08) and p[2] < TABLE_HEIGHT + ROBOT_TABLE_CLEARANCE:
            hits.append((frame, p.copy()))

    if hits:
        frame, p = hits[0]
        return False, f"robot/table proxy collision: {frame} at {np.round(p, 4)}, phase={phase}"

    return True, "ok"


def _grasp_relation_valid(scene, q, T_box, T_box_tcp, *, phase):
    T_expected_tcp = np.asarray(T_box, dtype=float) @ np.asarray(T_box_tcp, dtype=float)
    T_actual_tcp = np.asarray(scene.robot.frame_pose(TCP_FRAME, q), dtype=float)
    pos = float(np.linalg.norm(T_actual_tcp[:3, 3] - T_expected_tcp[:3, 3]))
    rot = float(np.rad2deg(_rotation_error_angle(T_actual_tcp[:3, :3], T_expected_tcp[:3, :3])))
    if pos > GRASP_CONTACT_POS_MAX or rot > GRASP_CONTACT_ROT_MAX_DEG:
        return False, f"object not in gripper: tcp-box mismatch pos={pos:.4f}, rot={rot:.2f}, phase={phase}"
    return True, "ok"


def _top_grasp_gap_valid(T_box, T_box_tcp, grasp, *, phase):
    if grasp.name != "top":
        return True, "ok"

    z_tcp_box = float(np.asarray(T_box_tcp, dtype=float)[2, 3])
    z_top = float(BOX_SIZE[2] / 2.0)
    if abs(z_tcp_box - z_top) > 0.006:
        return False, f"top grasp TCP not on top face: z_tcp_box={z_tcp_box:.4f}, z_top={z_top:.4f}, phase={phase}"

    y_tcp_box = float(np.asarray(T_box_tcp, dtype=float)[1, 3])
    if abs(y_tcp_box) > 0.004:
        return False, f"top grasp off center in closing direction: y={y_tcp_box:.4f}, phase={phase}"

    return True, "ok"


def _sample_valid(scene, q, T_box, T_box_tcp, grasp, *, box_state, phase):
    box_state = str(box_state)
    ok, reason = _robot_table_valid(scene, q, phase=phase)
    if not ok:
        return False, reason

    ok, reason = _box_table_valid(T_box, state=box_state, phase=phase)
    if not ok:
        return False, reason

    if box_state == "attached":
        ok, reason = _grasp_relation_valid(scene, q, T_box, T_box_tcp, phase=phase)
        if not ok:
            return False, reason

        ok, reason = _top_grasp_gap_valid(T_box, T_box_tcp, grasp, phase=phase)
        if not ok:
            return False, reason

    return True, "ok"


def _checked_interp_segment(scene, q_a, q_b, T_box_a, T_box_b, T_box_tcp, grasp, *, box_state, label, n=80):
    qs = _interp(q_a, q_b, n)
    if box_state == "attached":
        Ts = np.asarray([_box_pose_from_tcp(scene, q, T_box_tcp) for q in qs], dtype=float)
    else:
        Ts = _interp_pose_translation(T_box_a, T_box_b, n)
    for i, (q, T) in enumerate(zip(qs, Ts)):
        ok, reason = _sample_valid(scene, q, T, T_box_tcp, grasp, box_state=box_state, phase=label)
        if not ok:
            if BOX_DEMO_USE_OMPL and not OMPL_FALLBACK_WIRED:
                return None, None, f"{label} invalid at sample {i}/{len(qs)}: {reason}; OMPL fallback not wired: checked interpolation only"
            return None, None, f"{label} invalid at sample {i}/{len(qs)}: {reason}"
    return qs, Ts, "ok"


def _checked_hold_segment(scene, q, T_box, T_box_tcp, grasp, *, box_state, label, n=1):
    qs = np.repeat(np.asarray(q, dtype=float).reshape(1, -1), int(n), axis=0)
    if box_state == "attached":
        Ts = np.asarray([_box_pose_from_tcp(scene, qi, T_box_tcp) for qi in qs], dtype=float)
    else:
        Ts = _repeat_pose(T_box, int(n))
    for i, (qi, Ti) in enumerate(zip(qs, Ts)):
        ok, reason = _sample_valid(scene, qi, Ti, T_box_tcp, grasp, box_state=box_state, phase=label)
        if not ok:
            return None, None, f"{label} invalid at sample {i}/{len(qs)}: {reason}"
    return qs, Ts, "ok"


def _audit_full_demo_path(scene, path, box_poses, box_pose_state, T_box_tcp, grasp, labels):
    counts = {
        "robot_table": 0,
        "box_table": 0,
        "grasp_relation": 0,
        "other": 0,
    }
    first_reasons = []

    for i, (q, T_box, state, label) in enumerate(zip(path, box_poses, box_pose_state, labels)):
        ok, reason = _sample_valid(
            scene,
            q,
            T_box,
            T_box_tcp,
            grasp,
            box_state=str(state),
            phase=str(label),
        )
        if ok:
            continue

        text = f"sample {i}, phase={label}: {reason}"
        first_reasons.append(text)

        if "robot/table" in reason:
            counts["robot_table"] += 1
        elif "box" in reason and "table" in reason:
            counts["box_table"] += 1
        elif "gripper" in reason or "grasp" in reason:
            counts["grasp_relation"] += 1
        else:
            counts["other"] += 1

    passed = sum(counts.values()) == 0
    print("full path audit:")
    print(f"  robot/table violations : {counts['robot_table']}")
    print(f"  box/table violations   : {counts['box_table']}")
    print(f"  grasp relation errors  : {counts['grasp_relation']}")
    print(f"  other violations       : {counts['other']}")
    for reason in first_reasons[:8]:
        print("  first violation:", reason)
    print(f"  result: {'PASS' if passed else 'FAIL'}")
    return passed, counts, first_reasons


def _joint_length(path):
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _max_torque_util(scene, path):
    try:
        tau_max = np.asarray(scene.derate * scene.tau_max, dtype=float)
        vals = []
        for q in np.asarray(path, dtype=float):
            vals.append(float(np.max(np.abs(scene.robot.gravity(q)) / tau_max)))
        return float(max(vals)) if vals else float("inf")
    except Exception:
        return float("nan")


def _score_sample(grasp, joint_len, max_util, centering_error):
    width_fraction = float(grasp.required_width / max(_total_gap_from_finger(FINGER_MAX), 1.0e-6))
    torque_penalty = TORQUE_WEIGHT * max_util if np.isfinite(max_util) else 0.0
    side_adjustment = -SIDE_GRASP_BONUS if grasp.name.startswith("side_") else 0.0
    centering_penalty = CENTERING_WEIGHT * float(centering_error)
    return float(
        joint_len
        + torque_penalty
        + WIDTH_WEIGHT * width_fraction
        + centering_penalty
        + side_adjustment
    )


def _failed_sample(grasp, sample_id, T_box_tcp, reason, centering_error=None):
    return CandidateResult(
        grasp=grasp,
        success=False,
        q_pre=None,
        q_grasp=None,
        q_lift=None,
        score=float("inf"),
        reason=reason,
        sample_id=sample_id,
        T_box_tcp=T_box_tcp,
        centering_error=centering_error,
    )


def evaluate_candidate(scene, grasp, seeds, checker, sample_results=None):
    sample_results = [] if sample_results is None else sample_results
    if grasp.required_width > _total_gap_from_finger(FINGER_MAX) + 1.0e-9:
        res = CandidateResult(
            grasp,
            False,
            None,
            None,
            None,
            float("inf"),
            (
                f"required width {grasp.required_width:.3f} exceeds "
                f"gripper max gap {_total_gap_from_finger(FINGER_MAX):.3f}"
            ),
        )
        sample_results.append(res)
        return res

    T_box_initial = T_world_box()
    T_box_target = T_world_box_target()
    T_box_lift = T_box_initial.copy()
    T_box_lift[:3, 3] += np.array([0.0, 0.0, CARRY_HEIGHT], dtype=float)
    T_box_carry = T_box_target.copy()
    T_box_carry[:3, 3] += np.array([0.0, 0.0, CARRY_HEIGHT], dtype=float)
    T_box_place = T_box_target.copy()

    best_result = None
    last_failure = "no samples evaluated"

    for sample_id, T_box_tcp in enumerate(sampled_T_box_tcp(grasp)):
        centering_error = _grasp_centering_error(grasp, T_box_tcp)
        if grasp.name == "top" and centering_error > MAX_TOP_CENTERING_ERROR:
            last_failure = f"top grasp off-center by {centering_error:.4f} m"
            sample_results.append(
                _failed_sample(grasp, sample_id, T_box_tcp, last_failure, centering_error)
            )
            continue

        T_grasp = T_box_initial @ T_box_tcp
        T_lift = T_box_lift @ T_box_tcp
        T_carry = T_box_carry @ T_box_tcp
        T_place = T_box_place @ T_box_tcp

        T_pre = T_grasp.copy()
        T_pre[:3, 3] = T_pre[:3, 3] + np.asarray(grasp.pregrasp_offset_world, dtype=float)

        q_pre, ok = _try_project(
            scene, T_pre, seeds, f"pregrasp_{grasp.name}_{sample_id}", exact=False
        )
        if not ok:
            last_failure = "pregrasp projection failed"
            res = _failed_sample(grasp, sample_id, T_box_tcp, last_failure)
            sample_results.append(res)
            continue

        q_grasp, ok = _try_project(
            scene, T_grasp, [q_pre, *seeds], f"grasp_{grasp.name}_{sample_id}", exact=True
        )
        if not ok:
            last_failure = "grasp projection failed"
            res = CandidateResult(
                grasp,
                False,
                q_pre,
                None,
                None,
                float("inf"),
                last_failure,
                sample_id=sample_id,
                T_box_tcp=T_box_tcp,
                centering_error=centering_error,
            )
            sample_results.append(res)
            continue

        q_grasp = _polish_tcp_pose(scene, q_grasp, T_grasp, f"grasp_{grasp.name}")
        grasp_pos_err, grasp_rot_err = _pose_error(scene, q_grasp, T_grasp)
        grasp_rot_err_deg = float(np.rad2deg(grasp_rot_err))
        if grasp_pos_err > GRASP_POS_TOL_STRICT or grasp_rot_err_deg > GRASP_ROT_TOL_STRICT_DEG:
            last_failure = (
                f"grasp pose error too large: pos={grasp_pos_err:.4f} m, "
                f"rot={grasp_rot_err_deg:.2f} deg"
            )
            res = CandidateResult(
                grasp,
                False,
                q_pre,
                q_grasp,
                None,
                float("inf"),
                last_failure,
                sample_id=sample_id,
                T_box_tcp=T_box_tcp,
                centering_error=centering_error,
                grasp_pos_err=grasp_pos_err,
                grasp_rot_err_deg=grasp_rot_err_deg,
            )
            sample_results.append(res)
            continue

        q_lift, ok = _try_project(
            scene, T_lift, [q_grasp, q_pre, *seeds], f"lift_{grasp.name}_{sample_id}", exact=True
        )
        if not ok:
            last_failure = "lift projection failed"
            res = CandidateResult(
                grasp,
                False,
                q_pre,
                q_grasp,
                None,
                float("inf"),
                last_failure,
                sample_id=sample_id,
                T_box_tcp=T_box_tcp,
                centering_error=centering_error,
            )
            sample_results.append(res)
            continue

        q_lift = _polish_tcp_pose(scene, q_lift, T_lift, f"lift_{grasp.name}")
        lift_pos_err, lift_rot_err = _pose_error(scene, q_lift, T_lift)
        lift_rot_err_deg = float(np.rad2deg(lift_rot_err))

        q_carry, ok = _try_project(
            scene,
            T_carry,
            [q_lift, q_grasp, q_pre, *seeds],
            f"carry_{grasp.name}_{sample_id}",
            exact=True,
        )
        if not ok:
            last_failure = "carry projection failed"
            res = CandidateResult(
                grasp,
                False,
                q_pre,
                q_grasp,
                q_lift,
                float("inf"),
                last_failure,
                sample_id=sample_id,
                T_box_tcp=T_box_tcp,
                centering_error=centering_error,
                grasp_pos_err=grasp_pos_err,
                grasp_rot_err_deg=grasp_rot_err_deg,
                lift_pos_err=lift_pos_err,
                lift_rot_err_deg=lift_rot_err_deg,
            )
            sample_results.append(res)
            continue

        q_carry = _polish_tcp_pose(scene, q_carry, T_carry, f"carry_{grasp.name}")
        carry_pos_err, carry_rot_err = _pose_error(scene, q_carry, T_carry)
        carry_rot_err_deg = float(np.rad2deg(carry_rot_err))

        q_place, ok = _try_project(
            scene,
            T_place,
            [q_carry, q_lift, q_grasp, q_pre, *seeds],
            f"place_{grasp.name}_{sample_id}",
            exact=True,
        )
        if not ok:
            last_failure = "place projection failed"
            res = CandidateResult(
                grasp,
                False,
                q_pre,
                q_grasp,
                q_lift,
                float("inf"),
                last_failure,
                sample_id=sample_id,
                T_box_tcp=T_box_tcp,
                q_carry=q_carry,
                centering_error=centering_error,
                grasp_pos_err=grasp_pos_err,
                grasp_rot_err_deg=grasp_rot_err_deg,
                lift_pos_err=lift_pos_err,
                lift_rot_err_deg=lift_rot_err_deg,
                carry_pos_err=carry_pos_err,
                carry_rot_err_deg=carry_rot_err_deg,
            )
            sample_results.append(res)
            continue

        q_place = _polish_tcp_pose(scene, q_place, T_place, f"place_{grasp.name}")
        place_pos_err, place_rot_err = _pose_error(scene, q_place, T_place)
        place_rot_err_deg = float(np.rad2deg(place_rot_err))
        if place_pos_err > PLACE_POS_TOL_STRICT or place_rot_err_deg > PLACE_ROT_TOL_STRICT_DEG:
            last_failure = (
                f"place pose error too large: pos={place_pos_err:.4f} m, "
                f"rot={place_rot_err_deg:.2f} deg"
            )
            res = CandidateResult(
                grasp,
                False,
                q_pre,
                q_grasp,
                q_lift,
                float("inf"),
                last_failure,
                sample_id=sample_id,
                T_box_tcp=T_box_tcp,
                q_carry=q_carry,
                q_place=q_place,
                centering_error=centering_error,
                grasp_pos_err=grasp_pos_err,
                grasp_rot_err_deg=grasp_rot_err_deg,
                lift_pos_err=lift_pos_err,
                lift_rot_err_deg=lift_rot_err_deg,
                carry_pos_err=carry_pos_err,
                carry_rot_err_deg=carry_rot_err_deg,
                place_pos_err=place_pos_err,
                place_rot_err_deg=place_rot_err_deg,
            )
            sample_results.append(res)
            continue

        q_start = seeds[1] if len(seeds) > 1 else seeds[0]
        (
            path,
            _labels,
            _finger_widths,
            _attached,
            _contact_reached,
            _box_pose_state,
            _box_poses,
            path_reason,
        ) = _build_phase_path(
            scene,
            q_start,
            q_pre,
            q_grasp,
            q_lift,
            q_carry,
            q_place,
            grasp,
            T_box_initial,
            T_box_lift,
            T_box_carry,
            T_box_place,
            T_box_tcp,
        )
        if path is None:
            last_failure = f"path invalid: {path_reason}"
            res = CandidateResult(
                grasp=grasp,
                success=False,
                q_pre=q_pre,
                q_grasp=q_grasp,
                q_lift=q_lift,
                score=float("inf"),
                reason=last_failure,
                sample_id=sample_id,
                T_box_tcp=T_box_tcp,
                q_carry=q_carry,
                q_place=q_place,
                centering_error=centering_error,
                grasp_pos_err=grasp_pos_err,
                grasp_rot_err_deg=grasp_rot_err_deg,
                lift_pos_err=lift_pos_err,
                lift_rot_err_deg=lift_rot_err_deg,
                carry_pos_err=carry_pos_err,
                carry_rot_err_deg=carry_rot_err_deg,
                place_pos_err=place_pos_err,
                place_rot_err_deg=place_rot_err_deg,
            )
            sample_results.append(res)
            continue

        audit_pass, _audit_counts, audit_reasons = _audit_full_demo_path(
            scene,
            path,
            _box_poses,
            _box_pose_state,
            T_box_tcp,
            grasp,
            _labels,
        )
        if not audit_pass:
            first_reason = audit_reasons[0] if audit_reasons else "unknown"
            last_failure = f"path audit failed: {first_reason}"
            res = CandidateResult(
                grasp=grasp,
                success=False,
                q_pre=q_pre,
                q_grasp=q_grasp,
                q_lift=q_lift,
                score=float("inf"),
                reason=last_failure,
                sample_id=sample_id,
                T_box_tcp=T_box_tcp,
                q_carry=q_carry,
                q_place=q_place,
                centering_error=centering_error,
                grasp_pos_err=grasp_pos_err,
                grasp_rot_err_deg=grasp_rot_err_deg,
                lift_pos_err=lift_pos_err,
                lift_rot_err_deg=lift_rot_err_deg,
                carry_pos_err=carry_pos_err,
                carry_rot_err_deg=carry_rot_err_deg,
                place_pos_err=place_pos_err,
                place_rot_err_deg=place_rot_err_deg,
            )
            sample_results.append(res)
            continue

        collision_violations = _pybullet_audit_path(
            checker,
            path,
            _finger_widths,
            _box_poses,
            _box_pose_state,
            _labels,
        )
        if collision_violations:
            i, label, reason = collision_violations[0]
            last_failure = f"collision audit failed: sample {i}, phase={label}: {reason}"
            res = CandidateResult(
                grasp=grasp,
                success=False,
                q_pre=q_pre,
                q_grasp=q_grasp,
                q_lift=q_lift,
                score=float("inf"),
                reason=last_failure,
                sample_id=sample_id,
                T_box_tcp=T_box_tcp,
                q_carry=q_carry,
                q_place=q_place,
                centering_error=centering_error,
                grasp_pos_err=grasp_pos_err,
                grasp_rot_err_deg=grasp_rot_err_deg,
                lift_pos_err=lift_pos_err,
                lift_rot_err_deg=lift_rot_err_deg,
                carry_pos_err=carry_pos_err,
                carry_rot_err_deg=carry_rot_err_deg,
                place_pos_err=place_pos_err,
                place_rot_err_deg=place_rot_err_deg,
            )
            sample_results.append(res)
            continue

        joint_len = _joint_length(path)
        max_util = _max_torque_util(scene, path)
        score = _score_sample(grasp, joint_len, max_util, centering_error)

        res = CandidateResult(
            grasp=grasp,
            success=True,
            q_pre=q_pre,
            q_grasp=q_grasp,
            q_lift=q_lift,
            score=float(score),
            reason="ok",
            max_torque_util=max_util,
            joint_length=joint_len,
            sample_id=sample_id,
            T_box_tcp=T_box_tcp,
            q_carry=q_carry,
            q_place=q_place,
            centering_error=centering_error,
            grasp_pos_err=grasp_pos_err,
            grasp_rot_err_deg=grasp_rot_err_deg,
            lift_pos_err=lift_pos_err,
            lift_rot_err_deg=lift_rot_err_deg,
            carry_pos_err=carry_pos_err,
            carry_rot_err_deg=carry_rot_err_deg,
            place_pos_err=place_pos_err,
            place_rot_err_deg=place_rot_err_deg,
        )
        sample_results.append(res)
        if best_result is None or res.score < best_result.score:
            best_result = res

    if best_result is not None:
        return best_result

    return CandidateResult(
        grasp,
        False,
        None,
        None,
        None,
        float("inf"),
        f"all samples failed: {last_failure}",
    )


def _build_phase_path(
    scene,
    q_start,
    q_pre,
    q_grasp,
    q_lift,
    q_carry,
    q_place,
    grasp,
    T_world_box_initial,
    T_world_box_lift,
    T_world_box_carry,
    T_world_box_target,
    T_box_tcp,
):
    q_move, T_move, reason = _checked_interp_segment(
        scene,
        q_start,
        q_pre,
        T_world_box_initial,
        T_world_box_initial,
        T_box_tcp,
        grasp,
        box_state="initial",
        label="move_to_pregrasp",
        n=90,
    )
    if q_move is None:
        return None, None, None, None, None, None, None, reason

    q_approach, T_approach, reason = _checked_interp_segment(
        scene,
        q_pre,
        q_grasp,
        T_world_box_initial,
        T_world_box_initial,
        T_box_tcp,
        grasp,
        box_state="initial",
        label="approach_grasp",
        n=70,
    )
    if q_approach is None:
        return None, None, None, None, None, None, None, reason
    q_approach = q_approach[1:]
    T_approach = T_approach[1:]

    q_close, T_close, reason = _checked_hold_segment(
        scene,
        q_grasp,
        T_world_box_initial,
        T_box_tcp,
        grasp,
        box_state="initial",
        label="close_gripper",
        n=45,
    )
    if q_close is None:
        return None, None, None, None, None, None, None, reason

    q_hold, T_hold, reason = _checked_hold_segment(
        scene,
        q_grasp,
        T_world_box_initial,
        T_box_tcp,
        grasp,
        box_state="attached",
        label="hold_contact",
        n=20,
    )
    if q_hold is None:
        return None, None, None, None, None, None, None, reason

    q_lift_path, T_lift_path, reason = _checked_interp_segment(
        scene,
        q_grasp,
        q_lift,
        T_world_box_initial,
        T_world_box_lift,
        T_box_tcp,
        grasp,
        box_state="attached",
        label="lift_box",
        n=80,
    )
    if q_lift_path is None:
        return None, None, None, None, None, None, None, reason
    q_lift_path = q_lift_path[1:]
    T_lift_path = T_lift_path[1:]

    q_carry_path, T_carry_path, reason = _checked_interp_segment(
        scene,
        q_lift,
        q_carry,
        T_world_box_lift,
        T_world_box_carry,
        T_box_tcp,
        grasp,
        box_state="attached",
        label="carry_box",
        n=100,
    )
    if q_carry_path is None:
        return None, None, None, None, None, None, None, reason
    q_carry_path = q_carry_path[1:]
    T_carry_path = T_carry_path[1:]

    q_lower_path, T_lower_path, reason = _checked_interp_segment(
        scene,
        q_carry,
        q_place,
        T_world_box_carry,
        T_world_box_target,
        T_box_tcp,
        grasp,
        box_state="attached",
        label="lower_box",
        n=80,
    )
    if q_lower_path is None:
        return None, None, None, None, None, None, None, reason
    q_lower_path = q_lower_path[1:]
    T_lower_path = T_lower_path[1:]

    q_release, T_release, reason = _checked_hold_segment(
        scene,
        q_place,
        T_world_box_target,
        T_box_tcp,
        grasp,
        box_state="target",
        label="release_gripper",
        n=45,
    )
    if q_release is None:
        return None, None, None, None, None, None, None, reason

    q_retreat, T_retreat, reason = _checked_interp_segment(
        scene,
        q_place,
        q_pre,
        T_world_box_target,
        T_world_box_target,
        T_box_tcp,
        grasp,
        box_state="target",
        label="retreat",
        n=70,
    )
    if q_retreat is None:
        return None, None, None, None, None, None, None, reason
    q_retreat = q_retreat[1:]
    T_retreat = T_retreat[1:]

    path = np.vstack(
        [
            q_move,
            q_approach,
            q_close,
            q_hold,
            q_lift_path,
            q_carry_path,
            q_lower_path,
            q_release,
            q_retreat,
        ]
    )

    labels = (
        ["move_to_pregrasp"] * len(q_move)
        + ["approach_grasp"] * len(q_approach)
        + ["close_gripper"] * len(q_close)
        + ["hold_contact"] * len(q_hold)
        + ["lift_box"] * len(q_lift_path)
        + ["carry_box"] * len(q_carry_path)
        + ["lower_box"] * len(q_lower_path)
        + ["release_gripper"] * len(q_release)
        + ["retreat"] * len(q_retreat)
    )
    widths = []
    open_w = _open_finger_width()
    hold_w = _contact_finger_width(grasp)
    widths += [open_w] * len(q_move)
    widths += [open_w] * len(q_approach)
    widths += list(np.linspace(open_w, hold_w, len(q_close)))
    widths += [hold_w] * len(q_hold)
    widths += [hold_w] * len(q_lift_path)
    widths += [hold_w] * len(q_carry_path)
    widths += [hold_w] * len(q_lower_path)
    widths += list(np.linspace(hold_w, open_w, len(q_release)))
    widths += [open_w] * len(q_retreat)
    attached = np.asarray(
        [False] * len(q_move)
        + [False] * len(q_approach)
        + [False] * len(q_close)
        + [True] * len(q_hold)
        + [True] * len(q_lift_path)
        + [True] * len(q_carry_path)
        + [True] * len(q_lower_path)
        + [False] * len(q_release)
        + [False] * len(q_retreat),
        dtype=bool,
    )
    contact_reached = np.asarray(
        [False] * len(q_move)
        + [False] * len(q_approach)
        + [False] * (len(q_close) - 1)
        + [True]
        + [True] * len(q_hold)
        + [True] * len(q_lift_path)
        + [True] * len(q_carry_path)
        + [True] * len(q_lower_path)
        + [False] * len(q_release)
        + [False] * len(q_retreat),
        dtype=bool,
    )
    box_pose_state = np.asarray(
        ["initial"] * len(q_move)
        + ["initial"] * len(q_approach)
        + ["initial"] * len(q_close)
        + ["attached"] * len(q_hold)
        + ["attached"] * len(q_lift_path)
        + ["attached"] * len(q_carry_path)
        + ["attached"] * len(q_lower_path)
        + ["target"] * len(q_release)
        + ["target"] * len(q_retreat)
    )
    box_poses = np.concatenate(
        [T_move, T_approach, T_close, T_hold, T_lift_path, T_carry_path, T_lower_path, T_release, T_retreat],
        axis=0,
    )
    if len(box_poses) != len(path):
        raise RuntimeError(
            f"box pose trajectory length {len(box_poses)} does not match path length {len(path)}"
        )
    return (
        path,
        np.asarray(labels),
        np.asarray(widths, dtype=float),
        attached,
        contact_reached,
        box_pose_state,
        box_poses,
        "ok",
    )


def main():
    np.set_printoptions(precision=4, suppress=True)
    scene = _make_scene()
    seeds = _seed_set(scene)
    q_start = seeds[1] if len(seeds) > 1 else seeds[0]

    print("=" * 70)
    print("Panda box grasp TSR demo")
    print("This is isolated from run_panda.py / payload transfer.")
    print(f"box size: {BOX_SIZE.tolist()} m")
    print(f"box center: {BOX_CENTER_WORLD.tolist()} m")
    print(f"target center: {BOX_TARGET_CENTER_WORLD.tolist()} m")
    print(f"table bounds: {TABLE_XY_BOUNDS.tolist()} at z={TABLE_HEIGHT:.3f}")
    print("Planning/validation mode:")
    print("- Key poses: FrameConstraint projection")
    print("- Segment path: generated phase trajectory")
    print("- Collision checks: robot-table proxy, box-table OBB, TCP-box grasp relation")
    print("- Invalid paths are not saved")
    print("Collision mode:")
    print("- Table is a solid box.")
    print("- Box/table collision uses oriented-box corner checks.")
    print("- Robot/table collision uses FK proxy spheres on hand/wrist/finger frames.")
    print("- Invalid plans are not saved.")
    print("grasp candidates:", flush=True)

    try:
        checker = _make_collision_checker(scene)
    except BoxDemoCollisionError as exc:
        print(f"Collision checker setup failed: {exc}")
        _move_stale_plan()
        return 1

    results = []
    sample_results = []
    for grasp in grasp_library():
        res = evaluate_candidate(scene, grasp, seeds, checker, sample_results)
        results.append(res)
        print(
            f"  {grasp.name:14s} success={res.success} "
            f"sample={res.sample_id} score={res.score:.4f} reason={res.reason} "
            f"joint_len={res.joint_length} max_util={res.max_torque_util}",
            flush=True,
        )

    feasible = [r for r in results if r.success]
    if not feasible:
        print("No feasible grasp candidate found.")
        _move_stale_plan()
        checker.close()
        return 1

    best = min(feasible, key=lambda r: r.score)
    selected_T_box_tcp = best.T_box_tcp if best.T_box_tcp is not None else best.grasp.T_box_tcp
    selected_T_grasp = T_world_box() @ selected_T_box_tcp
    selected_T_place = T_world_box_target() @ selected_T_box_tcp

    print("-" * 70)
    print("candidate sample ranking:")
    print(
        "rank | grasp          | sample | score   | joint_len | max_util | width | "
        "center | grasp_pos | grasp_rot | place_pos | place_rot | reason"
    )
    successful_samples = sorted([r for r in sample_results if r.success], key=lambda r: r.score)
    for rank, res in enumerate(successful_samples, start=1):
        print(
            f"{rank:4d} | {res.grasp.name:14s} | {res.sample_id!s:6s} | "
            f"{res.score:7.4f} | {res.joint_length!s:9s} | "
            f"{res.max_torque_util!s:8s} | {res.grasp.required_width:5.3f} | "
            f"{(res.centering_error if res.centering_error is not None else np.nan):6.4f} | "
            f"{(res.grasp_pos_err if res.grasp_pos_err is not None else np.nan):9.4f} | "
            f"{(res.grasp_rot_err_deg if res.grasp_rot_err_deg is not None else np.nan):9.2f} | "
            f"{(res.place_pos_err if res.place_pos_err is not None else np.nan):9.4f} | "
            f"{(res.place_rot_err_deg if res.place_rot_err_deg is not None else np.nan):9.2f} | "
            f"{res.reason}"
        )
    failed_grasps = [r for r in results if not r.success]
    if failed_grasps:
        print("failed grasp-level reasons:")
        for res in failed_grasps:
            print(f"  {res.grasp.name} rejected: {res.reason}")
    print("Grasp selection explanation:")
    print("- Each grasp mode defines a face/approach type.")
    print("- Each mode generated TSR samples on the box.")
    print("- Samples were projected to pregrasp, grasp, lift, carry, and place.")
    print("- Invalid samples were rejected by width, reachability, pose error, or projection failure.")
    print("- The selected sample has the lowest score among feasible samples.")
    print(
        f"Selected {best.grasp.name} sample {best.sample_id} because it had the lowest score "
        f"({best.score:.4f}) among feasible samples."
    )

    print("-" * 70)
    print(f"selected grasp: {best.grasp.name}")
    print(f"selected sample: {best.sample_id}")
    print(f"description   : {best.grasp.description}")
    print(f"score         : {best.score:.4f}")
    print(f"max torque util estimate: {best.max_torque_util}")
    if best.max_torque_util is not None and np.isfinite(best.max_torque_util) and best.max_torque_util > 1.0:
        print("WARNING: selected grasp has quasi-static torque utilisation > 1.0 under current derate.")
    selected_centering_error = best.centering_error if best.centering_error is not None else np.nan
    print(f"selected grasp T_box_tcp translation: {selected_T_box_tcp[:3, 3]}")
    print(f"selected grasp centering error: {selected_centering_error:.4f} m")
    if best.grasp.name == "top" and selected_centering_error > 0.002:
        print("WARNING: selected top grasp is visibly off-center; expect asymmetric finger contact.")
    print(f"box top z in box frame: {BOX_SIZE[2] / 2.0}")
    print(f"initial box center    : {BOX_CENTER_WORLD}")
    print(f"target box center     : {BOX_TARGET_CENTER_WORLD}")
    print(
        "planned phases        : move_to_pregrasp, approach_grasp, close_gripper, "
        "hold_contact, lift_box, carry_box, lower_box, release_gripper, retreat"
    )
    print(f"gripper max gap       : {_total_gap_from_finger(FINGER_MAX):.3f} m")
    print(f"selected object width : {best.grasp.object_width:.3f} m")
    print(f"contact gap           : {_contact_gap_for_grasp(best.grasp):.3f} m")
    print(f"hold finger width     : {_contact_finger_width(best.grasp):.3f} m per finger")
    print(f"finger total contact gap      : {_contact_gap_for_grasp(best.grasp):.4f} m")
    print(f"box width across fingers      : {best.grasp.object_width:.4f} m")
    print(f"contact margin each side      : {best.grasp.contact_margin:.4f} m")
    print("selected pose audit:")
    print(
        f"  grasp pos={best.grasp_pos_err if best.grasp_pos_err is not None else np.nan:.4f} m, "
        f"rot={best.grasp_rot_err_deg if best.grasp_rot_err_deg is not None else np.nan:.2f} deg"
    )
    print(
        f"  lift  pos={best.lift_pos_err if best.lift_pos_err is not None else np.nan:.4f} m, "
        f"rot={best.lift_rot_err_deg if best.lift_rot_err_deg is not None else np.nan:.2f} deg"
    )
    print(
        f"  carry pos={best.carry_pos_err if best.carry_pos_err is not None else np.nan:.4f} m, "
        f"rot={best.carry_rot_err_deg if best.carry_rot_err_deg is not None else np.nan:.2f} deg"
    )
    print(
        f"  place pos={best.place_pos_err if best.place_pos_err is not None else np.nan:.4f} m, "
        f"rot={best.place_rot_err_deg if best.place_rot_err_deg is not None else np.nan:.2f} deg"
    )
    _print_grasp_alignment(scene, best.q_grasp, selected_T_grasp, "selected grasp")
    _print_grasp_alignment(scene, best.q_place, selected_T_place, "selected place")

    T_world_box_initial = T_world_box()
    T_world_box_target_pose = T_world_box_target()
    T_world_box_lift = T_world_box_initial.copy()
    T_world_box_lift[:3, 3] += np.array([0.0, 0.0, CARRY_HEIGHT], dtype=float)
    T_world_box_carry = T_world_box_target_pose.copy()
    T_world_box_carry[:3, 3] += np.array([0.0, 0.0, CARRY_HEIGHT], dtype=float)

    (
        path,
        labels,
        finger_widths,
        attached,
        contact_reached,
        box_pose_state,
        box_poses,
        path_reason,
    ) = _build_phase_path(
        scene,
        q_start,
        best.q_pre,
        best.q_grasp,
        best.q_lift,
        best.q_carry,
        best.q_place,
        best.grasp,
        T_world_box_initial,
        T_world_box_lift,
        T_world_box_carry,
        T_world_box_target_pose,
        selected_T_box_tcp,
    )
    if path is None:
        print(f"Selected candidate path invalid: {path_reason}")
        _move_stale_plan()
        checker.close()
        return 1

    audit_pos = []
    audit_rot = []
    for q, T_box, state in zip(path, box_poses, box_pose_state):
        if state != "attached":
            continue
        T_expected_tcp = T_box @ selected_T_box_tcp
        T_actual_tcp = np.asarray(scene.robot.frame_pose(TCP_FRAME, q), dtype=float)
        audit_pos.append(float(np.linalg.norm(T_actual_tcp[:3, 3] - T_expected_tcp[:3, 3])))
        audit_rot.append(float(np.rad2deg(_rotation_error_angle(T_actual_tcp[:3, :3], T_expected_tcp[:3, :3]))))
    max_attach_pos_err = float(max(audit_pos)) if audit_pos else float("nan")
    max_attach_rot_err_deg = float(max(audit_rot)) if audit_rot else float("nan")
    print("attachment pose audit:")
    print(f"  max TCP position error: {max_attach_pos_err:.4f} m")
    print(f"  max TCP rotation error: {max_attach_rot_err_deg:.2f} deg")
    if (
        np.isfinite(max_attach_pos_err)
        and np.isfinite(max_attach_rot_err_deg)
        and (
            max_attach_pos_err > ATTACH_AUDIT_POS_WARN
            or max_attach_rot_err_deg > ATTACH_AUDIT_ROT_WARN_DEG
        )
    ):
        print(
            "WARNING: attached object pose and TCP pose differ noticeably; "
            "animation may show gripper/object mismatch."
        )

    audit_pass, audit_counts, _audit_reasons = _audit_full_demo_path(
        scene,
        path,
        box_poses,
        box_pose_state,
        selected_T_box_tcp,
        best.grasp,
        labels,
    )
    if not audit_pass:
        _move_stale_plan()
        print("No valid plan saved because full path audit failed.")
        checker.close()
        return 1

    collision_violations = _pybullet_audit_path(
        checker,
        path,
        finger_widths,
        box_poses,
        box_pose_state,
        labels,
    )
    _print_pybullet_audit(collision_violations)
    if collision_violations:
        _move_stale_plan()
        print("No valid plan saved because collision audit failed.")
        checker.close()
        return 1

    np.savez(
        OUT,
        path=path,
        labels=labels,
        finger_widths=finger_widths,
        object_attached=attached,
        contact_reached=contact_reached,
        box_pose_state=box_pose_state,
        box_poses=box_poses,
        plan_valid=np.asarray(True),
        plan_version=np.asarray("box_grasp_tsr_v2_exact_pick_place"),
        path_audit_pass=np.asarray(True),
        path_audit_robot_table_violations=np.asarray(audit_counts["robot_table"]),
        path_audit_box_table_violations=np.asarray(audit_counts["box_table"]),
        path_audit_grasp_relation_errors=np.asarray(audit_counts["grasp_relation"]),
        path_audit_other_violations=np.asarray(audit_counts["other"]),
        collision_audit_pass=np.asarray(True),
        pybullet_collision_audit_pass=np.asarray(True),
        collision_audit_violation_count=np.asarray(0),
        pybullet_collision_violation_count=np.asarray(0),
        collision_model_note=np.asarray(
            "Hybrid collision: FK proxy robot-table, box OBB table, solid table geometry"
        ),
        selected_grasp=np.asarray(best.grasp.name),
        selected_sample_id=np.asarray(best.sample_id if best.sample_id is not None else -1),
        candidate_names=np.asarray(
            [
                f"{res.grasp.name}[{res.sample_id if res.sample_id is not None else '-'}]"
                for res in sample_results
            ]
        ),
        candidate_success=np.asarray([res.success for res in sample_results], dtype=bool),
        candidate_scores=np.asarray([res.score for res in sample_results], dtype=float),
        candidate_reasons=np.asarray([res.reason for res in sample_results]),
        candidate_centering_errors=np.asarray(
            [
                res.centering_error if res.centering_error is not None else np.nan
                for res in sample_results
            ],
            dtype=float,
        ),
        candidate_grasp_pos_errors=np.asarray(
            [res.grasp_pos_err if res.grasp_pos_err is not None else np.nan for res in sample_results],
            dtype=float,
        ),
        candidate_grasp_rot_errors_deg=np.asarray(
            [
                res.grasp_rot_err_deg if res.grasp_rot_err_deg is not None else np.nan
                for res in sample_results
            ],
            dtype=float,
        ),
        candidate_lift_pos_errors=np.asarray(
            [res.lift_pos_err if res.lift_pos_err is not None else np.nan for res in sample_results],
            dtype=float,
        ),
        candidate_lift_rot_errors_deg=np.asarray(
            [
                res.lift_rot_err_deg if res.lift_rot_err_deg is not None else np.nan
                for res in sample_results
            ],
            dtype=float,
        ),
        candidate_carry_pos_errors=np.asarray(
            [res.carry_pos_err if res.carry_pos_err is not None else np.nan for res in sample_results],
            dtype=float,
        ),
        candidate_carry_rot_errors_deg=np.asarray(
            [
                res.carry_rot_err_deg if res.carry_rot_err_deg is not None else np.nan
                for res in sample_results
            ],
            dtype=float,
        ),
        candidate_place_pos_errors=np.asarray(
            [res.place_pos_err if res.place_pos_err is not None else np.nan for res in sample_results],
            dtype=float,
        ),
        candidate_place_rot_errors_deg=np.asarray(
            [
                res.place_rot_err_deg if res.place_rot_err_deg is not None else np.nan
                for res in sample_results
            ],
            dtype=float,
        ),
        box_size=BOX_SIZE,
        box_center_world=BOX_CENTER_WORLD,
        box_target_center_world=BOX_TARGET_CENTER_WORLD,
        table_xy_bounds=TABLE_XY_BOUNDS,
        table_height=np.asarray(TABLE_HEIGHT),
        table_thickness=np.asarray(TABLE_THICKNESS),
        T_world_box=T_world_box_initial,
        T_world_box_initial=T_world_box_initial,
        T_world_box_target=T_world_box_target_pose,
        T_box_tcp=selected_T_box_tcp,
        gripper_open_gap=np.asarray(_total_gap_from_finger(FINGER_MAX), dtype=float),
        gripper_contact_gap=np.asarray(_contact_gap_for_grasp(best.grasp), dtype=float),
        gripper_hold_finger_width=np.asarray(_contact_finger_width(best.grasp), dtype=float),
        gripper_max_gap=np.asarray(_total_gap_from_finger(FINGER_MAX), dtype=float),
        selected_required_width=np.asarray(best.grasp.required_width, dtype=float),
        selected_object_width=np.asarray(best.grasp.object_width, dtype=float),
        selected_contact_gap=np.asarray(_contact_gap_for_grasp(best.grasp), dtype=float),
        selected_centering_error=np.asarray(selected_centering_error, dtype=float),
        selected_grasp_axis=np.asarray(best.grasp.grasp_axis),
        selected_grasp_pos_error=np.asarray(
            best.grasp_pos_err if best.grasp_pos_err is not None else np.nan,
            dtype=float,
        ),
        selected_grasp_rot_error_deg=np.asarray(
            best.grasp_rot_err_deg if best.grasp_rot_err_deg is not None else np.nan,
            dtype=float,
        ),
        selected_place_pos_error=np.asarray(
            best.place_pos_err if best.place_pos_err is not None else np.nan,
            dtype=float,
        ),
        selected_place_rot_error_deg=np.asarray(
            best.place_rot_err_deg if best.place_rot_err_deg is not None else np.nan,
            dtype=float,
        ),
        selected_lift_pos_error=np.asarray(
            best.lift_pos_err if best.lift_pos_err is not None else np.nan,
            dtype=float,
        ),
        selected_lift_rot_error_deg=np.asarray(
            best.lift_rot_err_deg if best.lift_rot_err_deg is not None else np.nan,
            dtype=float,
        ),
        selected_carry_pos_error=np.asarray(
            best.carry_pos_err if best.carry_pos_err is not None else np.nan,
            dtype=float,
        ),
        selected_carry_rot_error_deg=np.asarray(
            best.carry_rot_err_deg if best.carry_rot_err_deg is not None else np.nan,
            dtype=float,
        ),
        attach_audit_max_pos_error=np.asarray(max_attach_pos_err, dtype=float),
        attach_audit_max_rot_error_deg=np.asarray(max_attach_rot_err_deg, dtype=float),
        selected_grasp_object_width=np.asarray(best.grasp.object_width, dtype=float),
        selected_grasp_contact_margin=np.asarray(best.grasp.contact_margin, dtype=float),
        q_pre=best.q_pre,
        q_grasp=best.q_grasp,
        q_lift=best.q_lift,
        q_carry=best.q_carry,
        q_place=best.q_place,
    )
    checker.close()
    print(f"saved demo plan -> {OUT}")
    print("animate with:")
    print("  python -B animate_box_grasp_tsr.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
