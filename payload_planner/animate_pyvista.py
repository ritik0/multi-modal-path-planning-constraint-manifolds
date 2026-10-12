"""PyVista renderer for the saved Panda payload trajectory.

This file is intentionally renderer-only: it consumes ``panda_plan.npz`` and a
URDF with visual geometry, then drives the mentor ``robot_tools`` visualizer
with the already planned joint trajectory.  The Pinocchio dynamics/planner URDF
used by ``RobotModel`` is not touched.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyvista as pv
import payload_model
from scene import Scene
from run_panda import CFG, FRAME

pv.global_theme.allow_empty_mesh = True


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PLAN_PATH = HERE / "panda_plan.npz"
TOPPFAU_RENDER_URDF = HERE / "TOPPFAU_GOFA5" / "robot-assets" / "model" / "urdf" / "fer_franka_hand.urdf"
LOCAL_RENDER_URDF = HERE / "fer_franka_hand_render.urdf"
EE_LINK_NAME = "fer_hand_tcp"
TRAIL_Z_OFFSET = 0.004
MARKER_RADIUS = 0.022
_PAYLOAD_SCENE = None
_ACTIVE_ANIMATION_PAYLOAD_CASE = None
_GRASP_TRANSFORM_CACHE = {}

# --------------------------------------------------------------------------- #
# Payload visual anchoring
# --------------------------------------------------------------------------- #
# The payload mesh is built in the payload_model frame:
#   - origin   = the table-contact (assembly BOTTOM), z = 0
#   - +z points DOWN toward the table; the assembly extends in -z toward the
#     gripper (so the extrusion TOP, the grasp point, is at z = -EXTRUSION_LEN)
#   - x = wing axis (bricks stick out +/- x), y = depth.
#
# Historically the visual was placed with the *planning* model's
# Scene.frame_pose("payload", q).  That model is a DIFFERENT URDF than the one
# robot_tools renders, so the two forward-kinematics chains disagree and the
# load drifts away from the visible gripper (the gap) and can pick up the
# planner's payload-frame orientation (the wrong rotation).
#
# Default behaviour now anchors the payload RIGIDLY to the *visible* gripper TCP
# via the render robot's own FK, using the grasp transform below.  This removes
# any planning/render mismatch by construction.  Set PAYLOAD_ANCHOR_TO_TCP=0 to
# fall back to the old planning-frame pose.
#
# T_grasp expresses the payload frame *in the TCP frame*.  Defaults place the
# extrusion top at the TCP (grasp point) with the assembly aligned to the
# gripper approach axis, matching the physical setup photo.  Tune live with:
#   PAYLOAD_GRASP_XYZ      "x y z"   (m)   translation of payload origin in TCP frame
#   PAYLOAD_GRASP_RPY_DEG  "r p y"   (deg) rotation of payload frame in TCP frame
PAYLOAD_ANCHOR_TO_TCP = os.environ.get("PAYLOAD_ANCHOR_TO_TCP", "0") not in {
    "0", "false", "False", "no", "NO"
}


def _tube(points: np.ndarray, radius: float = 0.004):
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    if len(points) < 2:
        return None
    poly = pv.PolyData(points)
    n = len(points)
    poly.lines = np.hstack([[n], np.arange(n, dtype=np.int64)])
    tube = poly.tube(radius=float(radius), n_sides=12)
    if getattr(tube, "n_points", 0) == 0:
        return None
    return tube


def _box_from_xy(bounds: np.ndarray, height: float, thickness: float = 0.02):
    import pyvista as pv

    (xmin, ymin), (xmax, ymax) = np.asarray(bounds, dtype=float).reshape(2, 2)
    return pv.Box(
        bounds=(
            float(xmin),
            float(xmax),
            float(ymin),
            float(ymax),
            float(height) - thickness,
            float(height),
        )
    )


def _box_from_center_half_extents(center: np.ndarray, half_extents: np.ndarray):
    import pyvista as pv

    center = np.asarray(center, dtype=float).reshape(3)
    half = np.asarray(half_extents, dtype=float).reshape(3)
    lo = center - half
    hi = center + half
    return pv.Box(
        bounds=(
            float(lo[0]),
            float(hi[0]),
            float(lo[1]),
            float(hi[1]),
            float(lo[2]),
            float(hi[2]),
        )
    )


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in {"0", "false", "False", "no", "NO"}


def _points_poly(points: np.ndarray):
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    if len(points) == 0:
        return None
    return pv.PolyData(points)


def _edge_poly(edges: np.ndarray):
    edges = np.asarray(edges, dtype=float).reshape((-1, 2, 3))
    if len(edges) == 0:
        return None
    points = edges.reshape((-1, 3))
    poly = pv.PolyData(points)
    lines = []
    for i in range(len(edges)):
        lines.extend([2, 2 * i, 2 * i + 1])
    poly.lines = np.asarray(lines, dtype=np.int64)
    return poly


def _mode_color(mode: str) -> str:
    if mode in {"on_A", "slide_A"}:
        return "#10b981"
    if mode in {"on_B", "slide_B"}:
        return "#2563eb"
    if mode in {"free", "free_flight"}:
        return "#f59e0b"
    return "#64748b"


def _switch_indices_from_modes(modes: np.ndarray) -> tuple[int | None, int | None]:
    labels = [str(m) for m in np.asarray(modes).reshape(-1)]
    lift = None
    set_down = None
    for i in range(1, len(labels)):
        prev = labels[i - 1]
        cur = labels[i]
        if lift is None and prev in {"on_A", "slide_A"} and cur in {"free", "free_flight"}:
            lift = i
        if set_down is None and prev in {"free", "free_flight"} and cur in {"on_B", "slide_B"}:
            set_down = i
    return lift, set_down


def _mesh_filenames(urdf_path: Path) -> list[str]:
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:
        print(f"Could not parse URDF for mesh diagnostics: {urdf_path} ({exc})")
        return []
    return [
        str(mesh.attrib.get("filename", "")).strip()
        for mesh in root.findall(".//mesh")
        if str(mesh.attrib.get("filename", "")).strip()
    ]


def _candidate_mesh_paths(urdf_path: Path, mesh_name: str) -> list[Path]:
    mesh_path = Path(mesh_name)
    if mesh_path.is_absolute():
        return [mesh_path]
    roots = [
        urdf_path.parent,
        urdf_path.parent / "franka_description",
        urdf_path.parent.parent,
        Path.cwd(),
        HERE,
        HERE / "franka_description",
        HERE.parent,
    ]
    env_root = os.environ.get("ROBOT_ASSETS_ROOT") or os.environ.get("TOPPFAU_GOFA5")
    if env_root:
        env_path = Path(env_root)
        roots.insert(0, env_path)
        roots.insert(1, env_path / "robot-assets" / "model")
    candidates = []
    for root in roots:
        candidate = root / mesh_path
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _report_mesh_paths(urdf_path: Path) -> bool:
    missing = []
    for mesh_name in _mesh_filenames(urdf_path):
        tried = _candidate_mesh_paths(urdf_path, mesh_name)
        if not any(p.exists() for p in tried):
            missing.append((mesh_name, tried))
    if not missing:
        return True
    print("Mesh path diagnostics:")
    for mesh_name, tried in missing:
        print(f"  missing mesh reference: {mesh_name}")
        for path in tried:
            print(f"    tried: {path}")
    return False


def _resolve_urdf(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_urdf = os.environ.get("PAYLOAD_RENDER_URDF")
    if env_urdf:
        candidates.append(Path(env_urdf))
    topfrau = os.environ.get("TOPPFAU_GOFA5")
    if topfrau:
        candidates.append(
            Path(topfrau) / "robot-assets" / "model" / "urdf" / "fer_franka_hand.urdf"
        )
    assets_root = os.environ.get("ROBOT_ASSETS_ROOT")
    if assets_root:
        candidates.append(Path(assets_root) / "urdf" / "fer_franka_hand.urdf")
    candidates.extend([TOPPFAU_RENDER_URDF, LOCAL_RENDER_URDF])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    print("No render URDF found. Paths checked:")
    for candidate in candidates:
        print(f"  {candidate}")
    raise FileNotFoundError("render URDF not found")


def _load_robot_tools():
    try:
        from robot_tools.robot_visualization import Robot
    except Exception as exc:
        print("Could not import robot_tools.robot_visualization.Robot.")
        print(f"Import error: {exc}")
        print("Run this in the mentor environment that has robot_tools + pyvista on PYTHONPATH.")
        raise
    return Robot


def _print_run_instructions() -> None:
    print("Run in the environment that provides pyvista and pinocchio, for example:")
    if sys.platform.startswith("win"):
        print(r"  cd C:\dev\mmp\primitive-manifold-planner")
        print(r"  python -B payload_planner\animate_pyvista.py")
    else:
        print("  cd /mnt/c/dev/mmp/primitive-manifold-planner")
        print("  python -B payload_planner/animate_pyvista.py")
    print("Optional asset overrides:")
    print("  PAYLOAD_RENDER_URDF=<path to fer_franka_hand.urdf>")
    print("  ROBOT_ASSETS_ROOT=<path containing meshes/...>")


def _setup_scene_style(plotter) -> None:
    import pyvista as pv

    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    try:
        plotter.set_background("#eef3f8", top="#ffffff")
    except TypeError:
        plotter.set_background("#f6f8fb")
    try:
        plotter.remove_all_lights()
    except Exception:
        pass
    for position, intensity in [
        ((1.1, -1.1, 1.4), 0.42),
        ((-1.1, 0.9, 1.2), 0.36),
        ((0.0, 1.0, 1.7), 0.32),
    ]:
        light = pv.Light(position=position, focal_point=(0.48, 0.0, 0.38),
                         color="white", intensity=float(intensity))
        plotter.add_light(light)


def _apply_uniform_robot_material(robot) -> None:
    """Override embedded mesh colors for a clean mentor-style grey Franka."""
    grey = (0.82, 0.82, 0.80)
    for visual in getattr(robot, "visuals", []):
        actor = getattr(visual, "actor", None)
        if actor is None:
            continue
        try:
            try:
                actor.SetTexture(None)
            except Exception:
                pass
            try:
                actor.GetMapper().ScalarVisibilityOff()
            except Exception:
                pass
            prop = actor.GetProperty()
            prop.SetColor(*grey)
            prop.SetDiffuseColor(*grey)
            prop.SetAmbientColor(*grey)
            prop.SetAmbient(0.32)
            prop.SetDiffuse(0.72)
            prop.SetSpecular(0.08)
            prop.SetSpecularPower(12.0)
            if hasattr(prop, "SetMetallic"):
                prop.SetMetallic(0.0)
            if hasattr(prop, "SetRoughness"):
                prop.SetRoughness(0.78)
            if hasattr(prop, "SetInterpolationToGouraud"):
                prop.SetInterpolationToGouraud()
        except Exception:
            pass


def _check_renderer_environment() -> bool:
    ok = True
    if importlib.util.find_spec("pyvista") is None:
        print("Missing Python package: pyvista")
        print("Install in the active environment, for example:")
        print("  conda install -c conda-forge pyvista vtk trimesh")
        print("or:")
        print("  python -m pip install pyvista vtk trimesh")
        ok = False
    if importlib.util.find_spec("trimesh") is None:
        print("Missing Python package: trimesh")
        print("Install it so the high-resolution .dae visual meshes load.")
        ok = False
    if importlib.util.find_spec("robot_tools") is None:
        print("Missing local package: robot_tools")
        print("Run from the repository root so ./robot_tools is on PYTHONPATH.")
        ok = False
    if importlib.util.find_spec("pinocchio") is None:
        print("Missing Python package: pinocchio")
        print("Use the same environment that successfully runs the payload planner.")
        ok = False
    return ok


def _marker_point(robot, q: np.ndarray, table_height: float, radius: float = MARKER_RADIUS) -> np.ndarray:
    T = np.asarray(robot.fk(q, ee_link_name=EE_LINK_NAME), dtype=float)
    return np.asarray([T[0, 3], T[1, 3], float(table_height) + float(radius)], dtype=float)


def _add_marker(plotter, name: str, point: np.ndarray, color: str, radius: float = MARKER_RADIUS):
    import pyvista as pv

    p = np.asarray(point, dtype=float).reshape(3).copy()
    actor = plotter.add_mesh(
        pv.Sphere(radius=radius, center=p),
        name=name,
        color=color,
        smooth_shading=True,
    )
    return actor


def _add_marker_labels(plotter, points: list[np.ndarray], labels: list[str]) -> None:
    if not points:
        return
    pts = np.asarray(points, dtype=float).reshape((-1, 3))
    try:
        plotter.add_point_labels(
            pts,
            labels,
            font_size=14,
            text_color="black",
            point_size=0,
            shape_opacity=0.65,
            always_visible=True,
        )
    except TypeError:
        plotter.add_point_labels(pts, labels, font_size=14, text_color="black", point_size=0)


def _payload_box(center, size, rgb):
    import pyvista as pv

    center = np.asarray(center, dtype=float).reshape(3)
    size = np.asarray(size, dtype=float).reshape(3)
    half = 0.5 * size
    mesh = pv.Box(
        bounds=(
            float(center[0] - half[0]), float(center[0] + half[0]),
            float(center[1] - half[1]), float(center[1] + half[1]),
            float(center[2] - half[2]), float(center[2] + half[2]),
        )
    )
    colors = np.repeat(np.asarray(rgb, dtype=np.uint8).reshape(1, 3), mesh.n_cells, axis=0)
    mesh.cell_data["payload_rgb"] = colors
    return mesh


def _payload_case_from_plan(plan) -> str:
    if "payload_case" in plan.files:
        return payload_model.normalise_case(str(np.asarray(plan["payload_case"]).item()))
    return payload_model.normalise_case(
        os.environ.get("PAYLOAD_CASE", payload_model.ACTIVE_PAYLOAD_CASE)
    )


def _payload_mesh(case=None) -> "pv.PolyData":
    import pyvista as pv

    case = payload_model.normalise_case(case)
    colors = {
        "extrusion": (90, 90, 90),
        "brick_L":  (150, 55, 55),
        "brick_L2": (150, 55, 55),
        "brick_R":  (150, 55, 55),
        "brick_R2": (150, 55, 55),
    }
    parts = []
    for box in payload_model.component_boxes(case=case):
        name = box["name"]
        rgb = colors.get(name, (130, 130, 130))
        parts.append(_payload_box(box["center"], box["size"], rgb))
    return pv.MultiBlock(parts).combine()


def _payload_scene():
    global _PAYLOAD_SCENE
    if _PAYLOAD_SCENE is None:
        _PAYLOAD_SCENE = Scene(CFG)
    return _PAYLOAD_SCENE


def _env_floats(name: str, default: tuple[float, float, float]) -> np.ndarray:
    raw = os.environ.get(name)
    if not raw:
        return np.asarray(default, dtype=float)
    try:
        vals = [float(tok) for tok in raw.replace(",", " ").split()]
    except ValueError:
        print(f"Could not parse {name}={raw!r}; using default {default}")
        return np.asarray(default, dtype=float)
    if len(vals) != 3:
        print(f"{name} needs 3 numbers, got {len(vals)}; using default {default}")
        return np.asarray(default, dtype=float)
    return np.asarray(vals, dtype=float)


def _rpy_deg_to_matrix(rpy_deg: np.ndarray) -> np.ndarray:
    r, p, y = np.deg2rad(np.asarray(rpy_deg, dtype=float).reshape(3))
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _grasp_transform(case=None) -> np.ndarray:
    """Fixed pose of the payload frame expressed in the gripper TCP frame.

    Default: payload origin (table contact) sits +EXTRUSION_LEN along the TCP
    approach axis, so the extrusion TOP coincides with the TCP (the grasp point)
    and the assembly hangs below it.  Yaw is 0 here: the "bricks face front"
    orientation now lives in payload_model.py's geometry (wings on +/- y) and in
    the URDF link, so the payload frame stays aligned with the hand and the EE
    orientation is preserved.  Do NOT add 90 deg here as well or it double-rotates.
    """
    case = payload_model.normalise_case(case or _ACTIVE_ANIMATION_PAYLOAD_CASE)
    if case not in _GRASP_TRANSFORM_CACHE:
        default_xyz = (0.0, 0.0, payload_model.extrusion_len(case))
        xyz = _env_floats("PAYLOAD_GRASP_XYZ", default_xyz)
        rpy = _env_floats("PAYLOAD_GRASP_RPY_DEG", (0.0, 0.0, 0.0))
        T = np.eye(4, dtype=float)
        T[:3, :3] = _rpy_deg_to_matrix(rpy)
        T[:3, 3] = xyz
        _GRASP_TRANSFORM_CACHE[case] = T
    return _GRASP_TRANSFORM_CACHE[case]


def _payload_pose(robot, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if PAYLOAD_ANCHOR_TO_TCP:
        # Anchor the load rigidly to the VISIBLE gripper TCP (render robot FK),
        # so it tracks the gripper exactly with no planning/render gap.
        T_tcp = np.asarray(robot.fk(q, ee_link_name=EE_LINK_NAME), dtype=float)
        return T_tcp @ _grasp_transform(_ACTIVE_ANIMATION_PAYLOAD_CASE)
    # Legacy fallback: pose from the planning model's payload link.
    scene = _payload_scene()
    return scene.robot.frame_pose(FRAME, q)


def _safe_robot_update(robot, q: np.ndarray, frame_id: int) -> None:
    try:
        robot.update(q, id=0, color=(0.75, 0.75, 0.75, 1.0), opacity=1.0)
    except TypeError:
        try:
            robot.update(q, id=frame_id, opacity=1.0)
        except TypeError:
            robot.update(q)


def _window_closed(plotter) -> bool:
    try:
        return bool(plotter.iren.interactor.GetDone())
    except Exception:
        return False


def animate(plan_path: Path, urdf_path: Path, fps: float, loop: bool) -> int:
    import pyvista as pv
    global _ACTIVE_ANIMATION_PAYLOAD_CASE

    if not plan_path.exists():
        print(f"Missing plan file: {plan_path}")
        return 1
    urdf_path = urdf_path.resolve()
    print(f"render URDF: {urdf_path}")
    _report_mesh_paths(urdf_path)

    Robot = _load_robot_tools()

    plan = np.load(plan_path, allow_pickle=False)
    _ACTIVE_ANIMATION_PAYLOAD_CASE = _payload_case_from_plan(plan)
    print(f"payload animation case: {_ACTIVE_ANIMATION_PAYLOAD_CASE}")
    if PAYLOAD_ANCHOR_TO_TCP:
        g = _grasp_transform(_ACTIVE_ANIMATION_PAYLOAD_CASE)
        print(
            "payload visual pose source: render TCP "
            f"({EE_LINK_NAME}) + grasp transform "
            f"xyz={np.round(g[:3, 3], 4).tolist()}"
        )
    else:
        print("payload visual pose source: planning Scene.frame_pose(payload)")
    for box in payload_model.component_boxes(case=_ACTIVE_ANIMATION_PAYLOAD_CASE):
        print(
            f"  payload visual box: {box['name']} "
            f"center={np.round(box['center'], 4)} size={np.round(box['size'], 4)}"
        )
    path = np.asarray(plan["path"], dtype=float).reshape((-1, 7))
    dense = np.asarray(plan["dense"], dtype=float).reshape((-1, 7))
    ee_xyz = np.asarray(plan["ee_xyz"], dtype=float).reshape((-1, 3))
    table_a = np.asarray(plan["tableA"], dtype=float)
    table_b = np.asarray(plan["tableB"], dtype=float)
    h_a = float(np.asarray(plan["hA"]).item())
    h_b = float(np.asarray(plan["hB"]).item())
    table_collision_margin = (
        float(np.asarray(plan["table_collision_margin"]).item())
        if "table_collision_margin" in plan.files
        else 0.0
    )
    obstacle_center = (
        np.asarray(plan["obstacle_center"], dtype=float).reshape(3)
        if "obstacle_center" in plan.files
        else None
    )
    obstacle_half_extents = (
        np.asarray(plan["obstacle_half_extents"], dtype=float).reshape(3)
        if "obstacle_half_extents" in plan.files
        else None
    )
    gap_wall_enabled = (
        bool(np.asarray(plan["gap_wall_enabled"]).item())
        if "gap_wall_enabled" in plan.files
        else obstacle_center is not None
    )
    mode_labels = (
        np.asarray(plan["mode_labels"]).astype(str).reshape(-1)
        if "mode_labels" in plan.files
        else np.asarray([], dtype=str)
    )
    if len(dense) != len(ee_xyz):
        raise ValueError(f"dense and ee_xyz length mismatch: {len(dense)} vs {len(ee_xyz)}")

    record_path = os.environ.get("PAYLOAD_RECORD", "").strip()
    recording = bool(record_path)
    plotter = pv.Plotter(window_size=(1280, 800), off_screen=recording)
    _setup_scene_style(plotter)
    robot = Robot(str(urdf_path), plotter)
    _apply_uniform_robot_material(robot)
    _safe_robot_update(robot, dense[0], 0)
    payload_actor = plotter.add_mesh(
        _payload_mesh(_ACTIVE_ANIMATION_PAYLOAD_CASE),
        scalars="payload_rgb",
        rgb=True,
        smooth_shading=True,
        pickable=False,
    )
    payload_actor.user_matrix = _payload_pose(robot, dense[0])

    if table_collision_margin > 0.0:
        for table, height in [(table_a, h_a), (table_b, h_b)]:
            collision_top = max(0.0, float(height) - table_collision_margin)
            if collision_top > 0.0:
                plotter.add_mesh(
                    _box_from_xy(table, collision_top, thickness=collision_top),
                    color="#6b7280",
                    opacity=0.12,
                    show_edges=True,
                    edge_color="#4b5563",
                    pickable=False,
                )
    plotter.add_mesh(
        _box_from_xy(table_a, h_a),
        color="mediumseagreen",
        opacity=1.0,
        show_edges=False,
        pickable=False,
    )
    plotter.add_mesh(
        _box_from_xy(table_b, h_b),
        color="steelblue",
        opacity=1.0,
        show_edges=False,
        pickable=False,
    )
    if gap_wall_enabled and obstacle_center is not None and obstacle_half_extents is not None:
        plotter.add_mesh(
            _box_from_center_half_extents(obstacle_center, obstacle_half_extents),
            color="#c62828",
            opacity=0.45,
            show_edges=True,
            edge_color="#7f1d1d",
            pickable=False,
        )
    plotter.add_mesh(
        pv.Plane(center=(0.50, 0.0, -0.012), direction=(0, 0, 1),
                 i_size=1.25, j_size=1.20),
        color="#d7dde5",
        opacity=0.28,
        smooth_shading=False,
        pickable=False,
    )

    marker_points: list[np.ndarray] = []
    marker_labels: list[str] = []

    start_pos = _marker_point(robot, path[0], h_a)
    _add_marker(plotter, "start_marker", start_pos, "cyan")
    marker_points.append(start_pos)
    marker_labels.append("start")

    goal_pos = _marker_point(robot, path[-1], h_b)
    _add_marker(plotter, "goal_marker", goal_pos, "red")
    marker_points.append(goal_pos)
    marker_labels.append("goal")

    lift_idx = set_idx = None
    if len(mode_labels) == len(path):
        lift_idx, set_idx = _switch_indices_from_modes(mode_labels)
    if lift_idx is not None:
        lift_pos = _marker_point(robot, path[lift_idx], h_a)
        _add_marker(plotter, "lift_off_marker", lift_pos, "yellow")
        marker_points.append(lift_pos)
        marker_labels.append("lift-off")
    elif "lift_off" in plan.files and np.asarray(plan["lift_off"]).size == 7:
        lift_pos = _marker_point(robot, np.asarray(plan["lift_off"], dtype=float), h_a)
        _add_marker(plotter, "lift_off_marker", lift_pos, "yellow")
        marker_points.append(lift_pos)
        marker_labels.append("lift-off")
    if set_idx is not None:
        set_pos = _marker_point(robot, path[set_idx], h_b)
        _add_marker(plotter, "set_down_marker", set_pos, "magenta")
        marker_points.append(set_pos)
        marker_labels.append("set-down")
    elif "set_down" in plan.files and np.asarray(plan["set_down"]).size == 7:
        set_pos = _marker_point(robot, np.asarray(plan["set_down"], dtype=float), h_b)
        _add_marker(plotter, "set_down_marker", set_pos, "magenta")
        marker_points.append(set_pos)
        marker_labels.append("set-down")
    _add_marker_labels(plotter, marker_points, marker_labels)

    if _env_flag("PAYLOAD_SHOW_SAMPLES", "1"):
        ambient = np.asarray(plan["ambient_payload_points"], dtype=float).reshape((-1, 3)) if "ambient_payload_points" in plan.files else np.zeros((0, 3))
        poly = _points_poly(ambient)
        if poly is not None:
            plotter.add_mesh(poly, color="#9ca3af", opacity=0.18, render_points_as_spheres=True, point_size=5, pickable=False)
    if _env_flag("PAYLOAD_SHOW_TREE", "1"):
        accepted = np.asarray(plan["accepted_payload_points"], dtype=float).reshape((-1, 3)) if "accepted_payload_points" in plan.files else np.zeros((0, 3))
        accepted_modes = np.asarray(plan["accepted_modes"]).astype(str).reshape(-1) if "accepted_modes" in plan.files else np.asarray([], dtype=str)
        if len(accepted_modes) == len(accepted) and len(accepted):
            for mode in sorted(set(accepted_modes)):
                pts = accepted[accepted_modes == mode]
                poly = _points_poly(pts)
                if poly is not None:
                    plotter.add_mesh(poly, color=_mode_color(mode), opacity=0.62, render_points_as_spheres=True, point_size=7, pickable=False)
        else:
            poly = _points_poly(accepted)
            if poly is not None:
                plotter.add_mesh(poly, color="#f59e0b", opacity=0.62, render_points_as_spheres=True, point_size=7, pickable=False)
        edges = np.asarray(plan["tree_edges"], dtype=float).reshape((-1, 2, 3)) if "tree_edges" in plan.files else np.zeros((0, 2, 3))
        poly = _edge_poly(edges)
        if poly is not None:
            plotter.add_mesh(poly, color="#475569", opacity=0.22, line_width=1, pickable=False)
    if _env_flag("PAYLOAD_SHOW_REJECTED", "0"):
        rejected = np.asarray(plan["rejected_payload_points"], dtype=float).reshape((-1, 3)) if "rejected_payload_points" in plan.files else np.zeros((0, 3))
        poly = _points_poly(rejected)
        if poly is not None:
            plotter.add_mesh(poly, color="#ef4444", opacity=0.35, render_points_as_spheres=True, point_size=5, pickable=False)

    trail_actor = None

    try:
        ee_frame_actor = robot.plot_ee_frame(dense[0], ee_link_name=EE_LINK_NAME, plotter=plotter)
    except Exception:
        ee_frame_actor = None

    plotter.add_axes()
    plotter.camera_position = [(1.10, -1.05, 0.82), (0.48, 0.0, 0.36), (0.0, 0.0, 1.0)]

    if recording:
        # Off-screen render straight to a movie/GIF file. Reuses the exact same
        # robot/payload/FK update path as the interactive loop, but writes frames
        # instead of opening a window (works on headless / OSMesa VTK builds).
        if record_path.lower().endswith(".gif"):
            plotter.open_gif(record_path)
        else:
            plotter.open_movie(record_path, framerate=int(max(float(fps), 1.0)))
        plotter.show(auto_close=False)  # initializes the offscreen render window
        print(f"Recording {len(dense)} frames -> {record_path}")
        trail_points = []
        trail_actor = None
        for i, (q, ee) in enumerate(zip(dense, ee_xyz)):
            _safe_robot_update(robot, q, i)
            if ee_frame_actor is not None:
                try:
                    ee_frame_actor.user_matrix = robot.fk(q, ee_link_name=EE_LINK_NAME)
                except Exception:
                    pass
            try:
                payload_actor.user_matrix = _payload_pose(robot, q)
            except Exception:
                pass
            ee_pt = np.asarray(ee, dtype=float).reshape(3).copy()
            ee_pt[2] += TRAIL_Z_OFFSET
            trail_points.append(ee_pt)
            if len(trail_points) >= 2:
                tube = _tube(np.asarray(trail_points, dtype=float), radius=0.003)
                if tube is not None:
                    if trail_actor is None:
                        trail_actor = plotter.add_mesh(tube, color="#ff6a00")
                    else:
                        trail_actor.mapper.SetInputData(tube)
                        trail_actor.visibility = True
            plotter.write_frame()
        try:
            plotter.close()
        except Exception:
            pass
        print(f"Saved animation -> {record_path}")
        return 0

    plotter.show(auto_close=False, interactive_update=True)

    delay = 1.0 / max(float(fps), 1.0)
    print("Animating. Close the PyVista window or press Ctrl+C in the terminal to exit.")
    try:
        while True:
            trail_points = []
            if trail_actor is not None:
                trail_actor.visibility = False
            for i, (q, ee) in enumerate(zip(dense, ee_xyz)):
                if _window_closed(plotter):
                    return 0
                _safe_robot_update(robot, q, i)
                if ee_frame_actor is not None:
                    try:
                        ee_frame_actor.user_matrix = robot.fk(q, ee_link_name=EE_LINK_NAME)
                    except Exception:
                        pass
                try:
                    payload_actor.user_matrix = _payload_pose(robot, q)
                except Exception:
                    pass
                ee_pt = np.asarray(ee, dtype=float).reshape(3).copy()
                ee_pt[2] += TRAIL_Z_OFFSET
                trail_points.append(ee_pt)
                if len(trail_points) >= 2:
                    tube = _tube(np.asarray(trail_points, dtype=float), radius=0.003)
                    if tube is None:
                        plotter.update()
                        time.sleep(delay)
                        continue
                    if trail_actor is None:
                        trail_actor = plotter.add_mesh(tube, color="#ff6a00")
                    else:
                        trail_actor.mapper.SetInputData(tube)
                        trail_actor.visibility = True
                plotter.update()
                time.sleep(delay)
            if not loop:
                break
        plotter.show()
    except KeyboardInterrupt:
        print("Animation stopped.")
    finally:
        try:
            plotter.close()
        except Exception:
            pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(PLAN_PATH), help="Path to panda_plan.npz")
    parser.add_argument("--urdf", default=None, help="Renderer URDF override")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--once", action="store_true", help="Play once instead of looping")
    args = parser.parse_args()

    urdf = _resolve_urdf(args.urdf)
    _print_run_instructions()
    if not _check_renderer_environment():
        return 1
    return animate(Path(args.plan), urdf, fps=args.fps, loop=not args.once)


if __name__ == "__main__":
    raise SystemExit(main())