#!/usr/bin/env python3
"""
gripper_open_close.py  --  simulation-only gripper test (step 1)

Renders the Franka and animates the gripper opening and closing. Nothing here
touches the planner, the payload URDF, or any working code:

  * it makes a *copy* of fer_franka_hand_render.urdf with the two finger joints
    switched from `fixed` to `prismatic` (written next to this script), and
  * drives the two finger DOFs over time while the arm is held still.

Because robot_visualization.Robot.update(q) poses every link by Pinocchio FK,
prismatic finger joints move the finger meshes automatically once their values
are included in q -- no extra actors needed.

Run:
    python gripper_open_close.py
Needs: pyvista, pinocchio, trimesh, and robot_visualization.py on the path,
plus the render meshes resolvable (same as your normal renderer).
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyvista as pv

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_tools.robot_visualization import Robot

BASE_RENDER_URDF = HERE / "fer_franka_hand_render.urdf"
GRIP_RENDER_URDF = HERE / "fer_franka_hand_render_prismatic.urdf"   # generated

FINGER_JOINTS = ["fer_finger_joint1", "fer_finger_joint2"]
FINGER_MAX = 0.04            # per-finger travel (m); total gap = 2 * value (0..0.08)
ARM_JOINTS = [f"fer_joint{k}" for k in range(1, 8)]
READY_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # nice viewing pose

STEP_DT = 0.02              # s between frames
PERIOD_STEPS = 120          # steps for one full open+close cycle


def make_prismatic_render_urdf(src: Path, dst: Path, joints=FINGER_JOINTS) -> Path:
    """Copy the render URDF, switching the named finger joints fixed -> prismatic.

    The joints already carry <axis> and <limit>, so the gripper just becomes
    movable. The original file is left untouched.
    """
    tree = ET.parse(src)
    root = tree.getroot()
    switched = []
    for joint in root.findall("joint"):
        if joint.attrib.get("name") in joints:
            joint.set("type", "prismatic")
            if joint.find("axis") is None:
                ET.SubElement(joint, "axis", {"xyz": "0 1 0"})
            if joint.find("limit") is None:
                ET.SubElement(joint, "limit",
                              {"effort": "100", "lower": "0.0",
                               "upper": str(FINGER_MAX), "velocity": "0.2"})
            switched.append(joint.attrib.get("name"))
    if len(switched) != len(joints):
        missing = set(joints) - set(switched)
        raise RuntimeError(f"finger joints not found in URDF: {sorted(missing)}")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    return dst


def joint_q_index(robot: Robot, joint_name: str) -> int:
    """Index of a 1-DOF joint's value inside the Pinocchio q vector."""
    jid = robot.model.getJointId(joint_name)
    return int(robot.model.joints[jid].idx_q)


def build_full_q(robot: Robot, arm_q, finger_w) -> np.ndarray:
    """Assemble a full nq vector: arm angles + both finger widths (robust to order)."""
    q = np.zeros(robot.model.nq, dtype=float)
    for name, val in zip(ARM_JOINTS, arm_q):
        q[joint_q_index(robot, name)] = float(val)
    for name in FINGER_JOINTS:
        q[joint_q_index(robot, name)] = float(finger_w)
    return q


def main() -> None:
    if not BASE_RENDER_URDF.exists():
        raise SystemExit(f"render URDF not found: {BASE_RENDER_URDF}")
    make_prismatic_render_urdf(BASE_RENDER_URDF, GRIP_RENDER_URDF)

    plotter = pv.Plotter()
    plotter.set_background("white")
    robot = Robot(str(GRIP_RENDER_URDF), plotter)

    # a simple ground/table so the robot isn't floating
    plotter.add_mesh(
        pv.Plane(center=(0.4, 0.0, 0.0), direction=(0, 0, 1), i_size=1.4, j_size=1.4),
        color=(0.82, 0.82, 0.85), opacity=1.0,
    )

    # pose everything once (closed gripper) before showing
    q = build_full_q(robot, READY_Q, 0.0)
    robot.update(q)

    # aim the camera at the hand
    hand = np.asarray(robot.fk(READY_Q if robot.model.nq == 7 else q, ee_link_name="fer_hand"))
    focus = hand[:3, 3]
    plotter.camera.focal_point = tuple(focus)
    plotter.camera.position = tuple(focus + np.array([0.6, -0.6, 0.35]))
    plotter.camera.up = (0.0, 0.0, 1.0)

    plotter.add_text("gripper test -- close window to stop", position="upper_left",
                     font_size=10, color="black", name="title")

    plotter.show(interactive_update=True, auto_close=False)

    k = 0
    try:
        while True:
            frac = (k % PERIOD_STEPS) / PERIOD_STEPS
            tri = 1.0 - abs(2.0 * frac - 1.0)        # 0 -> 1 -> 0 (closed->open->closed)
            w = FINGER_MAX * tri
            q = build_full_q(robot, READY_Q, w)
            robot.update(q)
            plotter.add_text(f"gap = {2.0 * w * 100.0:5.1f} cm   (finger = {w * 1000:4.1f} mm)",
                             position="lower_left", font_size=12, color="black", name="gap")
            plotter.update()
            time.sleep(STEP_DT)
            k += 1
    except Exception:
        # window closed (or interrupted) -> stop quietly
        pass


if __name__ == "__main__":
    main()