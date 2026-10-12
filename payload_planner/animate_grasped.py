"""PyVista animator for panda_plan_grasped.npz.

The grasp-schedule plan stores the arm path and payload pose separately.  This
renderer drives the arm from ``path`` and uses the saved object poses only for
phase markers; the render URDF has no payload link, so the gripper remains
visible.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PLAN_PATH = HERE / "panda_plan_grasped.npz"
TOPPFAU_RENDER_URDF = HERE / "TOPPFAU_GOFA5" / "robot-assets" / "model" / "urdf" / "fer_franka_hand.urdf"
LOCAL_RENDER_URDF = HERE / "fer_franka_hand_render.urdf"


def _renderer_helpers():
    from animate_pyvista import (
        _add_marker,
        _apply_uniform_robot_material,
        _box_from_center_half_extents,
        _box_from_xy,
        _load_robot_tools,
        _report_mesh_paths,
        _safe_robot_update,
        _setup_scene_style,
        _tube,
        _window_closed,
    )

    return {
        "add_marker": _add_marker,
        "apply_uniform_robot_material": _apply_uniform_robot_material,
        "box_from_center_half_extents": _box_from_center_half_extents,
        "box_from_xy": _box_from_xy,
        "load_robot_tools": _load_robot_tools,
        "report_mesh_paths": _report_mesh_paths,
        "safe_robot_update": _safe_robot_update,
        "setup_scene_style": _setup_scene_style,
        "tube": _tube,
        "window_closed": _window_closed,
    }


def _print_run_instructions() -> None:
    print("Run in the environment that provides pyvista and pinocchio, for example:")
    print(r"  cd C:\dev\mmp\primitive-manifold-planner")
    print(r"  python -B payload_planner\animate_grasped.py --plan payload_planner\panda_plan_grasped.npz --once")
    print("Optional asset overrides:")
    print("  PAYLOAD_RENDER_URDF=<path to fer_franka_hand.urdf>")
    print("  ROBOT_ASSETS_ROOT=<path containing meshes/...>")


def _check_renderer_environment() -> bool:
    ok = True
    for package, message in [
        ("pyvista", "Missing Python package: pyvista"),
        ("trimesh", "Missing Python package: trimesh"),
        ("robot_tools", "Missing local package: robot_tools"),
        ("pinocchio", "Missing Python package: pinocchio"),
    ]:
        if importlib.util.find_spec(package) is None:
            print(message)
            ok = False
    if not ok:
        print("Use the same environment that successfully runs the payload planner and PyVista renderer.")
    return ok


def _resolve_urdf(explicit: str | None) -> Path:
    import os

    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_urdf = os.environ.get("PAYLOAD_RENDER_URDF")
    if env_urdf:
        candidates.append(Path(env_urdf))
    topfrau = os.environ.get("TOPPFAU_GOFA5")
    if topfrau:
        candidates.append(Path(topfrau) / "robot-assets" / "model" / "urdf" / "fer_franka_hand.urdf")
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


def _scalar_int(value, default: int = -1) -> int:
    arr = np.asarray(value)
    if arr.size == 0:
        return int(default)
    return int(arr.reshape(-1)[0])


def _load_grasped_plan(plan_path: Path) -> dict[str, np.ndarray]:
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing grasped plan file: {plan_path}")
    data = np.load(plan_path, allow_pickle=True)
    required = [
        "path",
        "object_positions",
        "object_rotations",
        "grasp_yaw",
        "labels",
        "lift_off_index",
        "set_down_index",
        "regrasp_indices",
        "chosen_grasp_yaw",
        "tableA",
        "tableB",
        "hA",
        "hB",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{plan_path.name} is missing {', '.join(missing)} for a grasped plan animation.")

    path = np.asarray(data["path"], dtype=float).reshape((-1, 7))
    obj_pos = np.asarray(data["object_positions"], dtype=float).reshape((-1, 3))
    obj_rot = np.asarray(data["object_rotations"], dtype=float).reshape((-1, 3, 3))
    yaws = np.asarray(data["grasp_yaw"], dtype=float).reshape((-1,))
    labels = np.asarray(data["labels"]).astype(str).reshape((-1,))
    n = len(path)
    if not (len(obj_pos) == len(obj_rot) == len(yaws) == len(labels) == n):
        raise ValueError(
            "grasped plan arrays have inconsistent lengths: "
            f"path={n}, object_positions={len(obj_pos)}, object_rotations={len(obj_rot)}, "
            f"grasp_yaw={len(yaws)}, labels={len(labels)}"
        )
    obstacle_center = None
    obstacle_half_extents = None
    table_collision_margin = 0.0
    if "table_collision_margin" in data.files:
        raw_margin = np.asarray(data["table_collision_margin"], dtype=float).reshape(-1)
        if raw_margin.size:
            table_collision_margin = float(raw_margin[0])
    if "obstacle_center" in data.files and "obstacle_half_extents" in data.files:
        raw_center = np.asarray(data["obstacle_center"], dtype=float).reshape(-1)
        raw_half = np.asarray(data["obstacle_half_extents"], dtype=float).reshape(-1)
        if raw_center.size == 3 and raw_half.size == 3:
            obstacle_center = raw_center.reshape(3)
            obstacle_half_extents = raw_half.reshape(3)
    if obstacle_center is None or obstacle_half_extents is None or table_collision_margin <= 0.0:
        sibling = plan_path.with_name("panda_plan.npz")
        if sibling.exists():
            sibling_data = np.load(sibling, allow_pickle=True)
            if "obstacle_center" in sibling_data.files and "obstacle_half_extents" in sibling_data.files:
                obstacle_center = np.asarray(sibling_data["obstacle_center"], dtype=float).reshape(3)
                obstacle_half_extents = np.asarray(sibling_data["obstacle_half_extents"], dtype=float).reshape(3)
            if table_collision_margin <= 0.0 and "table_collision_margin" in sibling_data.files:
                raw_margin = np.asarray(sibling_data["table_collision_margin"], dtype=float).reshape(-1)
                if raw_margin.size:
                    table_collision_margin = float(raw_margin[0])

    result = {
        "path": path,
        "object_positions": obj_pos,
        "object_rotations": obj_rot,
        "grasp_yaw": yaws,
        "labels": labels,
        "lift_off_index": np.asarray(_scalar_int(data["lift_off_index"]), dtype=int),
        "set_down_index": np.asarray(_scalar_int(data["set_down_index"]), dtype=int),
        "regrasp_indices": np.asarray(data["regrasp_indices"], dtype=int).reshape((-1,)),
        "regrasp_from_yaw": np.asarray(data["regrasp_from_yaw"], dtype=float).reshape((-1,)) if "regrasp_from_yaw" in data.files else np.zeros(0),
        "regrasp_to_yaw": np.asarray(data["regrasp_to_yaw"], dtype=float).reshape((-1,)) if "regrasp_to_yaw" in data.files else np.zeros(0),
        "chosen_grasp_yaw": np.asarray(data["chosen_grasp_yaw"], dtype=float).reshape((-1,)),
        "tableA": np.asarray(data["tableA"], dtype=float),
        "tableB": np.asarray(data["tableB"], dtype=float),
        "hA": np.asarray(data["hA"], dtype=float),
        "hB": np.asarray(data["hB"], dtype=float),
        "table_collision_margin": np.asarray(table_collision_margin, dtype=float),
    }
    if obstacle_center is not None and obstacle_half_extents is not None:
        result["obstacle_center"] = obstacle_center
        result["obstacle_half_extents"] = obstacle_half_extents
    return result


def _add_plan_markers(plotter, plan: dict[str, np.ndarray]) -> None:
    helpers = _renderer_helpers()
    add_marker = helpers["add_marker"]
    obj_pos = np.asarray(plan["object_positions"], dtype=float)
    h_a = float(np.asarray(plan["hA"]).item())
    h_b = float(np.asarray(plan["hB"]).item())
    lift_idx = int(plan["lift_off_index"])
    set_idx = int(plan["set_down_index"])
    if len(obj_pos):
        start = obj_pos[0].copy()
        if np.isfinite(h_a):
            start[2] = h_a
        add_marker(plotter, "start_marker", start, "cyan")
        goal = obj_pos[-1].copy()
        if np.isfinite(h_b):
            goal[2] = h_b
        add_marker(plotter, "goal_marker", goal, "red")
    if 0 <= lift_idx < len(obj_pos):
        p = obj_pos[lift_idx].copy()
        if np.isfinite(h_a):
            p[2] = h_a
        add_marker(plotter, "lift_off_marker", p, "yellow")
    if 0 <= set_idx < len(obj_pos):
        p = obj_pos[set_idx].copy()
        if np.isfinite(h_b):
            p[2] = h_b
        add_marker(plotter, "set_down_marker", p, "magenta")


def _uniform_joint_frames(
    path: np.ndarray,
    object_positions: np.ndarray,
    fps: float,
    seconds: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = np.asarray(path, dtype=float).reshape((-1, 7))
    object_positions = np.asarray(object_positions, dtype=float).reshape((-1, 3))
    frame_count = max(2, int(round(max(1.0, float(fps)) * float(seconds))))
    if len(path) == 1:
        return (
            np.repeat(path, frame_count, axis=0),
            np.zeros(frame_count, dtype=int),
            np.repeat(object_positions[:1], frame_count, axis=0),
        )
    d = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumdist = np.concatenate([[0.0], np.cumsum(d)])
    total = float(cumdist[-1])
    if total <= 1.0e-12:
        return (
            np.repeat(path[:1], frame_count, axis=0),
            np.zeros(frame_count, dtype=int),
            np.repeat(object_positions[:1], frame_count, axis=0),
        )
    targets = np.linspace(0.0, total, frame_count)
    frames = np.column_stack([np.interp(targets, cumdist, path[:, j]) for j in range(path.shape[1])])
    object_frames = np.column_stack(
        [np.interp(targets, cumdist, object_positions[:, j]) for j in range(object_positions.shape[1])]
    )
    right = np.searchsorted(cumdist, targets, side="left")
    right = np.clip(right, 0, len(path) - 1)
    left = np.clip(right - 1, 0, len(path) - 1)
    choose_left = np.abs(targets - cumdist[left]) <= np.abs(cumdist[right] - targets)
    source_indices = np.where(choose_left, left, right).astype(int)
    return frames, source_indices, object_frames


def animate(plan_path: Path, urdf_path: Path, fps: float, loop: bool) -> int:
    import pyvista as pv

    helpers = _renderer_helpers()
    apply_uniform_robot_material = helpers["apply_uniform_robot_material"]
    box_from_center_half_extents = helpers["box_from_center_half_extents"]
    box_from_xy = helpers["box_from_xy"]
    load_robot_tools = helpers["load_robot_tools"]
    report_mesh_paths = helpers["report_mesh_paths"]
    safe_robot_update = helpers["safe_robot_update"]
    setup_scene_style = helpers["setup_scene_style"]
    tube = helpers["tube"]
    window_closed = helpers["window_closed"]

    plan = _load_grasped_plan(plan_path)
    path = plan["path"]
    object_positions = plan["object_positions"]
    yaws = plan["grasp_yaw"]
    labels = plan["labels"]
    frames, source_indices, object_frames = _uniform_joint_frames(path, object_positions, fps)

    urdf_path = urdf_path.resolve()
    print(f"render URDF: {urdf_path}")
    print(f"grasped plan: {plan_path}")
    print(f"frames: {len(path)} source, {len(frames)} uniform-speed animated")
    report_mesh_paths(urdf_path)
    Robot = load_robot_tools()

    plotter = pv.Plotter(window_size=(1280, 800))
    setup_scene_style(plotter)
    robot = Robot(str(urdf_path), plotter)
    apply_uniform_robot_material(robot)
    safe_robot_update(robot, path[0], 0)

    table_a = np.asarray(plan["tableA"], dtype=float)
    table_b = np.asarray(plan["tableB"], dtype=float)
    h_a = float(np.asarray(plan["hA"]).item())
    h_b = float(np.asarray(plan["hB"]).item())
    table_collision_margin = float(np.asarray(plan["table_collision_margin"]).item())
    if table_collision_margin > 0.0:
        for table, height in [(table_a, h_a), (table_b, h_b)]:
            if not np.asarray(table).size:
                continue
            collision_top = max(0.0, float(height) - table_collision_margin)
            if collision_top > 0.0:
                plotter.add_mesh(
                    box_from_xy(table, collision_top, thickness=collision_top),
                    color="#6b7280",
                    opacity=0.12,
                    show_edges=True,
                    edge_color="#4b5563",
                    pickable=False,
                )
    if table_a.size:
        plotter.add_mesh(box_from_xy(table_a, h_a), color="mediumseagreen", opacity=1.0, show_edges=False, pickable=False)
    if table_b.size:
        plotter.add_mesh(box_from_xy(table_b, h_b), color="steelblue", opacity=1.0, show_edges=False, pickable=False)
    if "obstacle_center" in plan and "obstacle_half_extents" in plan:
        plotter.add_mesh(
            box_from_center_half_extents(plan["obstacle_center"], plan["obstacle_half_extents"]),
            color="#c62828",
            opacity=0.45,
            show_edges=True,
            edge_color="#7f1d1d",
            pickable=False,
        )
    plotter.add_mesh(
        pv.Plane(center=(0.50, 0.0, -0.012), direction=(0, 0, 1), i_size=1.25, j_size=1.20),
        color="#d7dde5",
        opacity=0.28,
        smooth_shading=False,
        pickable=False,
    )
    _add_plan_markers(plotter, plan)
    full_trace = tube(object_positions, radius=0.0025)
    if full_trace is not None:
        plotter.add_mesh(full_trace, color="#64748b", opacity=0.28, pickable=False)
    trail_actor = None
    hud = plotter.add_text("", position="upper_left", font_size=12, color="#111827", name="hud")

    plotter.add_axes()
    plotter.camera_position = [(1.10, -1.05, 0.82), (0.48, 0.0, 0.36), (0.0, 0.0, 1.0)]
    plotter.show(auto_close=False, interactive_update=True)

    delay = 1.0 / max(float(fps), 1.0)
    print("Animating grasped plan. Close the PyVista window or press Ctrl+C in the terminal to exit.")
    try:
        while True:
            trail_points: list[np.ndarray] = []
            if trail_actor is not None:
                trail_actor.visibility = False
            for frame_no, q in enumerate(frames):
                if window_closed(plotter):
                    return 0
                i = int(source_indices[frame_no])
                safe_robot_update(robot, q, int(frame_no))
                trail_points.append(np.asarray(object_frames[frame_no], dtype=float).reshape(3))
                if len(trail_points) >= 2:
                    trail = tube(np.asarray(trail_points, dtype=float), radius=0.004)
                    if trail is not None:
                        if trail_actor is None:
                            trail_actor = plotter.add_mesh(trail, color="#ff6a00", pickable=False)
                        else:
                            trail_actor.mapper.SetInputData(trail)
                            trail_actor.visibility = True
                text = f"{str(labels[i])} | grasp {np.rad2deg(yaws[i]):.0f}deg"
                try:
                    hud.SetText(0, text)
                except Exception:
                    try:
                        hud.input = text
                    except Exception:
                        pass
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
    parser.add_argument("--plan", default=str(PLAN_PATH), help="Path to panda_plan_grasped.npz")
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
