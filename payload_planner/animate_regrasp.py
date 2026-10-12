from __future__ import annotations

import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyvista as pv


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_tools.robot_visualization import Robot


PLAN = Path(os.environ.get("BOX_REGRASP_ANIMATE_PLAN", HERE / "box_regrasp_plan.npz"))
BASE_RENDER_URDF = HERE / "fer_franka_hand_render.urdf"
GRIP_RENDER_URDF = HERE / "fer_franka_hand_render_prismatic.urdf"

FINGER_JOINTS = ["fer_finger_joint1", "fer_finger_joint2"]
ARM_JOINTS = [f"fer_joint{k}" for k in range(1, 8)]
FPS = float(os.environ.get("BOX_GRASP_FPS", "60"))
SPEED = float(os.environ.get("BOX_GRASP_SPEED", "1.0"))
FRAME_STRIDE = max(1, int(os.environ.get("BOX_GRASP_FRAME_STRIDE", "1")))
FRAME_DELAY = 1.0 / max(FPS * SPEED, 1.0)


def make_prismatic_render_urdf(src: Path, dst: Path) -> Path:
    tree = ET.parse(src)
    root = tree.getroot()
    switched = []
    for joint in root.findall("joint"):
        if joint.attrib.get("name") in FINGER_JOINTS:
            joint.set("type", "prismatic")
            if joint.find("axis") is None:
                if joint.attrib.get("name") == "fer_finger_joint1":
                    ET.SubElement(joint, "axis", {"xyz": "0 1 0"})
                elif joint.attrib.get("name") == "fer_finger_joint2":
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


def joint_q_index(robot: Robot, joint_name: str) -> int:
    jid = robot.model.getJointId(joint_name)
    return int(robot.model.joints[jid].idx_q)


def build_full_q(robot: Robot, arm_q, finger_w) -> np.ndarray:
    q = np.zeros(robot.model.nq, dtype=float)
    arm_q = np.asarray(arm_q, dtype=float).reshape(-1)
    for name, val in zip(ARM_JOINTS, arm_q):
        q[joint_q_index(robot, name)] = float(val)
    for name in FINGER_JOINTS:
        q[joint_q_index(robot, name)] = float(finger_w)
    return q


def box_mesh(size):
    size = np.asarray(size, dtype=float).reshape(3)
    half = 0.5 * size
    return pv.Box(bounds=(-half[0], half[0], -half[1], half[1], -half[2], half[2]))


def table_mesh(bounds, height):
    (xlo, ylo), (xhi, yhi) = np.asarray(bounds, dtype=float).reshape(2, 2)
    return pv.Box(bounds=(xlo, xhi, ylo, yhi, height - 0.025, height))


def main():
    if not PLAN.exists():
        print(f"missing plan: {PLAN}")
        print("Run first:")
        print("  python -B regrasp_plan.py")
        return 1
    if not BASE_RENDER_URDF.exists():
        print(f"missing render URDF: {BASE_RENDER_URDF}")
        return 1

    data = np.load(PLAN, allow_pickle=True)
    plan_version = str(np.asarray(data["plan_version"]).item()) if "plan_version" in data.files else "<old>"
    print(f"plan version: {plan_version}")
    if "plan_valid" not in data.files or not bool(np.asarray(data["plan_valid"]).item()):
        raise SystemExit("Plan is missing plan_valid=True. Re-run regrasp_plan.py successfully.")
    if "path_audit_pass" not in data.files or not bool(np.asarray(data["path_audit_pass"]).item()):
        raise SystemExit("Plan is missing path_audit_pass=True. Re-run regrasp_plan.py successfully.")
    if "collision_audit_pass" not in data.files:
        raise SystemExit("Plan is missing collision_audit_pass=True. Re-run regrasp_plan.py successfully.")
    collision_audit_pass = bool(np.asarray(data["collision_audit_pass"]).item())
    if not collision_audit_pass:
        raise SystemExit("Plan collision audit failed. Re-run regrasp_plan.py successfully.")
    if "box_poses" not in data.files:
        raise SystemExit("Regrasp animation requires saved box_poses. Re-run regrasp_plan.py.")

    path = np.asarray(data["path"], dtype=float)
    labels = np.asarray(data["labels"]).astype(str)
    finger_widths = np.asarray(data["finger_widths"], dtype=float)
    attached = np.asarray(data["object_attached"], dtype=bool)
    box_poses = np.asarray(data["box_poses"], dtype=float)
    box_pose_state = (
        np.asarray(data["box_pose_state"]).astype(str)
        if "box_pose_state" in data.files
        else np.where(attached, "attached", "resting").astype(str)
    )
    contact_reached = (
        np.asarray(data["contact_reached"], dtype=bool)
        if "contact_reached" in data.files
        else attached.copy()
    )
    if len(box_poses) != len(path):
        raise SystemExit(f"box_poses length {len(box_poses)} does not match path length {len(path)}")
    if len(finger_widths) != len(path):
        raise SystemExit(f"finger_widths length {len(finger_widths)} does not match path length {len(path)}")

    box_size = np.asarray(data["box_size"], dtype=float)
    T_world_box_initial = np.asarray(data["T_world_box_initial"], dtype=float)
    T_world_box_target = (
        np.asarray(data["T_world_box_target"], dtype=float)
        if "T_world_box_target" in data.files
        else box_poses[-1]
    )
    table_bounds = np.asarray(data["table_xy_bounds"], dtype=float)
    table_height = float(np.asarray(data["table_height"]).item())
    table_surfaces_bounds = (
        np.asarray(data["table_surfaces_bounds"], dtype=float)
        if "table_surfaces_bounds" in data.files
        else None
    )
    table_surfaces_heights = (
        np.asarray(data["table_surfaces_heights"], dtype=float)
        if "table_surfaces_heights" in data.files
        else None
    )
    selected = str(np.asarray(data["selected_grasp"]).item())

    print(f"selected grasp route: {selected}")
    print("box animation source: saved regrasp box_poses")
    print("gripper animation source: saved finger_widths")
    final_err = np.linalg.norm(box_poses[-1, :3, 3] - T_world_box_target[:3, 3])
    print(f"final box-target position error: {final_err:.6f} m")
    if "gripper_hold_finger_width" in data.files:
        print(
            f"initial hold finger width: "
            f"{float(np.asarray(data['gripper_hold_finger_width']).item()):.4f} m"
        )

    make_prismatic_render_urdf(BASE_RENDER_URDF, GRIP_RENDER_URDF)

    plotter = pv.Plotter(window_size=(1280, 800))
    plotter.set_background("white")

    robot = Robot(str(GRIP_RENDER_URDF), plotter)
    q0 = build_full_q(robot, path[0], finger_widths[0])
    robot.update(q0)

    if table_surfaces_bounds is not None:
        heights = (
            table_surfaces_heights
            if table_surfaces_heights is not None and len(table_surfaces_heights) == len(table_surfaces_bounds)
            else np.full((len(table_surfaces_bounds),), table_height, dtype=float)
        )
        for bounds, height in zip(table_surfaces_bounds, heights):
            plotter.add_mesh(
                table_mesh(bounds, float(height)),
                color="mediumseagreen",
                opacity=1.0,
                show_edges=False,
            )
    else:
        plotter.add_mesh(
            table_mesh(table_bounds, table_height),
            color="mediumseagreen",
            opacity=1.0,
            show_edges=False,
        )

    box_actor = plotter.add_mesh(
        box_mesh(box_size),
        color="#b45309",
        opacity=1.0,
        show_edges=True,
    )
    box_actor.user_matrix = T_world_box_initial

    target_actor = plotter.add_mesh(
        box_mesh(box_size),
        color="dodgerblue",
        opacity=0.25,
        show_edges=True,
    )
    target_actor.user_matrix = T_world_box_target

    plotter.add_text(
        f"box regrasp demo | route: {selected}",
        position="upper_left",
        font_size=10,
        color="black",
        name="title",
    )

    plotter.add_axes()
    plotter.camera_position = [(0.95, -0.75, 0.55), (0.38, 0.36, 0.12), (0, 0, 1)]
    plotter.show(interactive_update=True, auto_close=False)

    print("Animating regrasp plan. Close window to stop.")

    try:
        for i in range(0, len(path), FRAME_STRIDE):
            q_arm = path[i]
            q_full = build_full_q(robot, q_arm, finger_widths[i])
            robot.update(q_full)

            box_actor.user_matrix = box_poses[i]

            if i % 10 == 0:
                contact = bool(contact_reached[i]) if len(contact_reached) == len(path) else bool(attached[i])
                held = bool(attached[i])
                state = str(box_pose_state[i]) if len(box_pose_state) == len(path) else ("attached" if held else "resting")
                plotter.add_text(
                    (
                        f"phase: {labels[i]} | "
                        f"gap={2.0 * finger_widths[i] * 100.0:.1f} cm | "
                        f"contact={contact} | attached={held} | box={state}"
                    ),
                    position="lower_left",
                    font_size=11,
                    color="black",
                    name="phase",
                )
            plotter.update()
            time.sleep(FRAME_DELAY)

        plotter.show()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            plotter.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
