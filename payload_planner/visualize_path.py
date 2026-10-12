"""Visualize a saved Franka payload path as a Pinocchio skeleton animation.

The view avoids URDF meshes completely: every frame is drawn from joint origins
computed by forward kinematics, plus the controlled payload frame.  That keeps
the script independent of package:// mesh resolution while still showing the
motion clearly.
"""

from __future__ import annotations

import os

import numpy as np
import pinocchio

from run_panda import CFG, SEED, START, GOAL, SURFACE_FOOTPRINTS
from robot_model import RobotModel


HERE = os.path.dirname(os.path.abspath(__file__))
PATH_IN = os.path.join(HERE, "panda_path.npy")
GIF_OUT = os.path.join(HERE, "panda_plan.gif")
PNG_OUT = os.path.join(HERE, "panda_plan.png")


def _surface_by_name(name: str) -> dict:
    for surface in CFG["surfaces"]:
        if surface["name"] == name:
            return surface
    raise KeyError(name)


def _footprint(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return finite xy corners for drawing a support surface."""
    if name in SURFACE_FOOTPRINTS:
        lo, hi = SURFACE_FOOTPRINTS[name]
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    surface = _surface_by_name(name)
    if "xy_bounds" in surface:
        arr = np.asarray(surface["xy_bounds"], dtype=float).reshape(2, 2)
        return arr[0], arr[1]
    center = np.asarray([START.x, START.y] if name == START.surface else [GOAL.x, GOAL.y], dtype=float)
    half = np.asarray([0.30, 0.30], dtype=float)
    return center - half, center + half


def _joint_segments(robot: RobotModel, q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return parent-child joint-origin line segments for one configuration.

    Args:
        robot: RobotModel wrapping a Pinocchio model.
        q: Joint vector with shape ``(nq,)``.

    Returns:
        List of ``(parent_xyz, child_xyz)`` segment endpoints, each shape ``(3,)``.
    """
    model = robot.pin_model
    data = robot.pin_data
    pinocchio.forwardKinematics(model, data, np.asarray(q, dtype=float))

    def origin(joint_id: int) -> np.ndarray:
        if joint_id == 0:
            return np.asarray(robot.T0.translation, dtype=float).reshape(3)
        return np.asarray((robot.T0 * data.oMi[joint_id]).translation, dtype=float).reshape(3)

    segments = []
    for joint_id in range(1, model.njoints):
        parent = int(model.parents[joint_id])
        segments.append((origin(parent), origin(joint_id)))
    return segments


def _payload_positions(robot: RobotModel, path: np.ndarray) -> np.ndarray:
    """Compute controlled-frame positions for a path with shape ``(N, nq)``."""
    return np.asarray(
        [robot.frame_pose(CFG["controlled_frame"], q)[:3, 3] for q in np.asarray(path, dtype=float)],
        dtype=float,
    )


def _leave_land_indices(payload_xyz: np.ndarray, tol: float = 2.0e-3) -> tuple[int, int]:
    """Infer lift-off and set-down indices from payload height relative to surfaces."""
    h_start = float(_surface_by_name(START.surface)["height"])
    h_goal = float(_surface_by_name(GOAL.surface)["height"])
    z = payload_xyz[:, 2]
    leave = 0
    for i in range(1, len(z)):
        if abs(float(z[i]) - h_start) > tol:
            leave = i
            break
    land = len(z) - 1
    for i in range(max(leave + 1, 1), len(z)):
        if abs(float(z[i]) - h_goal) <= tol:
            land = i
            break
    return int(leave), int(land)


def _draw_surface_3d(ax, name: str) -> None:
    surface = _surface_by_name(name)
    z = float(surface["height"])
    lo, hi = _footprint(name)
    xs = np.asarray([[lo[0], hi[0]], [lo[0], hi[0]]], dtype=float)
    ys = np.asarray([[lo[1], lo[1]], [hi[1], hi[1]]], dtype=float)
    zs = np.full_like(xs, z)
    ax.plot_surface(xs, ys, zs, alpha=0.25, color="#78a8ff", edgecolor="#3f5f99", linewidth=0.5)
    ax.text(float(lo[0]), float(lo[1]), z + 0.02, name)


def _draw_surface_2d(ax, name: str) -> None:
    import matplotlib.patches as patches

    lo, hi = _footprint(name)
    width, height = hi - lo
    rect = ax.add_patch(
        patches.Rectangle(
            lo,
            float(width),
            float(height),
            facecolor="#dbe8ff",
            edgecolor="#3f5f99",
            alpha=0.65,
        )
    )
    rect.set_label(f"table {name}")
    ax.text(float(lo[0]), float(lo[1]), name)


def save_static(robot: RobotModel, path: np.ndarray, payload_xyz: np.ndarray) -> tuple[int, int]:
    """Save top-down payload path and payload-z profile plots."""
    import matplotlib.pyplot as plt

    leave, land = _leave_land_indices(payload_xyz)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_xy, ax_z = axes
    for name in (START.surface, GOAL.surface):
        _draw_surface_2d(ax_xy, name)
    ax_xy.plot(payload_xyz[:, 0], payload_xyz[:, 1], "k-", lw=1.5, label="payload path")
    ax_xy.plot(payload_xyz[0, 0], payload_xyz[0, 1], "go", label="start")
    ax_xy.plot(payload_xyz[-1, 0], payload_xyz[-1, 1], "ro", label="goal")
    ax_xy.plot(payload_xyz[leave, 0], payload_xyz[leave, 1], "yo", label="leave A")
    ax_xy.plot(payload_xyz[land, 0], payload_xyz[land, 1], "mo", label="land B")
    ax_xy.set_title("Top-down payload path")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.axis("equal")
    ax_xy.legend(loc="best")

    steps = np.arange(len(payload_xyz))
    ax_z.plot(steps, payload_xyz[:, 2], "k-", lw=1.5)
    for name in (START.surface, GOAL.surface):
        h = float(_surface_by_name(name)["height"])
        ax_z.axhline(h, ls="--", lw=1.0, label=f"{name} z={h:.2f}")
    ax_z.axvline(leave, color="y", lw=1.0)
    ax_z.axvline(land, color="m", lw=1.0)
    ax_z.set_title("Payload height")
    ax_z.set_xlabel("path step")
    ax_z.set_ylabel("z [m]")
    ax_z.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=160)
    print(f"saved static plot -> {PNG_OUT}")
    return leave, land


def save_animation(robot: RobotModel, path: np.ndarray, payload_xyz: np.ndarray) -> None:
    """Save a 3-D skeleton animation and try to show it interactively."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frame_ids = np.linspace(0, len(path) - 1, min(len(path), 180), dtype=int)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for name in (START.surface, GOAL.surface):
        _draw_surface_3d(ax, name)

    all_xyz = payload_xyz
    margin = 0.15
    ax.set_xlim(float(np.min(all_xyz[:, 0]) - margin), float(np.max(all_xyz[:, 0]) + margin))
    ax.set_ylim(float(np.min(all_xyz[:, 1]) - margin), float(np.max(all_xyz[:, 1]) + margin))
    z_hi = max(float(np.max(all_xyz[:, 2]) + 0.35), 0.9)
    ax.set_zlim(0.0, z_hi)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Panda payload plan skeleton")

    skeleton_lines = []
    for _a, _b in _joint_segments(robot, path[0]):
        (line,) = ax.plot([], [], [], color="#1f77b4", lw=2.0)
        skeleton_lines.append(line)
    (trace_line,) = ax.plot([], [], [], "k-", lw=1.5)
    (payload_dot,) = ax.plot([], [], [], "o", color="#ffbf00", ms=6)

    def update(frame_number: int):
        idx = int(frame_ids[frame_number])
        for line, (a, b) in zip(skeleton_lines, _joint_segments(robot, path[idx])):
            pts = np.vstack([a, b])
            line.set_data(pts[:, 0], pts[:, 1])
            line.set_3d_properties(pts[:, 2])
        trace = payload_xyz[: idx + 1]
        trace_line.set_data(trace[:, 0], trace[:, 1])
        trace_line.set_3d_properties(trace[:, 2])
        p = payload_xyz[idx]
        payload_dot.set_data([p[0]], [p[1]])
        payload_dot.set_3d_properties([p[2]])
        return [*skeleton_lines, trace_line, payload_dot]

    anim = FuncAnimation(fig, update, frames=len(frame_ids), interval=50, blit=False)
    anim.save(GIF_OUT, writer=PillowWriter(fps=20))
    print(f"saved animation -> {GIF_OUT}")
    try:
        plt.show()
    except Exception as exc:
        print(f"plt.show() skipped: {exc}")


def main() -> int:
    if not os.path.exists(PATH_IN):
        print(f"missing path file: {PATH_IN}")
        print("run run_panda.py first")
        return 1
    path = np.load(PATH_IN)
    path = np.asarray(path, dtype=float).reshape((-1, len(SEED)))
    robot = RobotModel(CFG["urdf"])
    payload_xyz = _payload_positions(robot, path)
    leave, land = save_static(robot, path, payload_xyz)
    print(f"leave index: {leave}, land index: {land}")
    save_animation(robot, path, payload_xyz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
