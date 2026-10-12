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


PLAN = HERE / "box_grasp_tsr_plan.npz"
BASE_RENDER_URDF = HERE / "fer_franka_hand_render.urdf"
GRIP_RENDER_URDF = HERE / "fer_franka_hand_render_prismatic.urdf"

FINGER_JOINTS = ["fer_finger_joint1", "fer_finger_joint2"]
ARM_JOINTS = [f"fer_joint{k}" for k in range(1, 8)]
TCP_FRAME = "fer_hand_tcp"
FPS = float(os.environ.get("BOX_GRASP_FPS", "60"))
SPEED = float(os.environ.get("BOX_GRASP_SPEED", "1.0"))
FRAME_STRIDE = max(1, int(os.environ.get("BOX_GRASP_FRAME_STRIDE", "1")))
FRAME_DELAY = 1.0 / max(FPS * SPEED, 1.0)
ALLOW_LEGACY = os.environ.get("BOX_GRASP_ALLOW_LEGACY_ANIMATION", "0") == "1"
SHOW_CONTACT_DEBUG = os.environ.get("BOX_GRASP_SHOW_CONTACT_DEBUG", "1") not in {
    "0",
    "false",
    "False",
    "no",
    "NO",
}


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


def table_mesh(bounds, height, thickness):
    (xlo, ylo), (xhi, yhi) = np.asarray(bounds, dtype=float).reshape(2, 2)
    return pv.Box(bounds=(xlo, xhi, ylo, yhi, height - float(thickness), height))


def main():
    if not PLAN.exists():
        print(f"missing plan: {PLAN}")
        print("Run first:")
        print("  python -B run_box_grasp_tsr.py")
        return 1
    if not BASE_RENDER_URDF.exists():
        print(f"missing render URDF: {BASE_RENDER_URDF}")
        return 1

    data = np.load(PLAN, allow_pickle=True)
    plan_version = str(np.asarray(data["plan_version"]).item()) if "plan_version" in data.files else "<old>"
    print(f"plan version: {plan_version}")
    if "plan_valid" not in data.files or not bool(np.asarray(data["plan_valid"]).item()):
        raise SystemExit("Plan is missing plan_valid=True. Re-run run_box_grasp_tsr.py successfully.")
    if "path_audit_pass" not in data.files or not bool(np.asarray(data["path_audit_pass"]).item()):
        raise SystemExit("Saved plan does not have path_audit_pass=True. Re-run run_box_grasp_tsr.py.")
    if "pybullet_collision_audit_pass" in data.files:
        collision_audit_pass = bool(np.asarray(data["pybullet_collision_audit_pass"]).item())
    elif "collision_audit_pass" in data.files:
        collision_audit_pass = bool(np.asarray(data["collision_audit_pass"]).item())
    else:
        raise SystemExit("Plan has no collision audit pass flag. Re-run run_box_grasp_tsr.py.")
    if not collision_audit_pass:
        raise SystemExit("Saved plan failed collision audit. Refusing animation.")
    print(f"path audit pass: {bool(np.asarray(data['path_audit_pass']).item())}")
    print(f"collision audit pass: {collision_audit_pass}")
    if "path_audit_robot_table_violations" in data.files:
        print(
            "robot/table violations: "
            f"{int(np.asarray(data['path_audit_robot_table_violations']).item())}"
        )
    if "path_audit_box_table_violations" in data.files:
        print(
            "box/table violations: "
            f"{int(np.asarray(data['path_audit_box_table_violations']).item())}"
        )
    if "path_audit_grasp_relation_errors" in data.files:
        print(
            "grasp relation errors: "
            f"{int(np.asarray(data['path_audit_grasp_relation_errors']).item())}"
        )
    print("animation note: this renderer trusts the saved path; validity must pass in run_box_grasp_tsr.py")

    path = np.asarray(data["path"], dtype=float)
    labels = np.asarray(data["labels"]).astype(str)
    finger_widths = np.asarray(data["finger_widths"], dtype=float)
    attached = np.asarray(data["object_attached"], dtype=bool)
    contact_reached = (
        np.asarray(data["contact_reached"], dtype=bool)
        if "contact_reached" in data.files
        else np.zeros(len(path), dtype=bool)
    )
    box_size = np.asarray(data["box_size"], dtype=float)
    T_world_box_initial = (
        np.asarray(data["T_world_box_initial"], dtype=float)
        if "T_world_box_initial" in data.files
        else np.asarray(data["T_world_box"], dtype=float)
    )
    T_world_box_target = (
        np.asarray(data["T_world_box_target"], dtype=float)
        if "T_world_box_target" in data.files
        else T_world_box_initial.copy()
    )
    box_pose_state = (
        np.asarray(data["box_pose_state"]).astype(str)
        if "box_pose_state" in data.files
        else np.where(attached, "attached", "initial").astype(str)
    )
    if "box_poses" not in data.files:
        if not ALLOW_LEGACY:
            raise SystemExit("This plan has no saved box_poses. Re-run run_box_grasp_tsr.py.")
        box_poses = None
    else:
        box_poses = np.asarray(data["box_poses"], dtype=float)
        if len(box_poses) != len(path):
            if not ALLOW_LEGACY:
                raise SystemExit(
                    f"box_poses length {len(box_poses)} does not match path length {len(path)}"
                )
            box_poses = None

    T_box_tcp = np.asarray(data["T_box_tcp"], dtype=float)
    table_bounds = np.asarray(data["table_xy_bounds"], dtype=float)
    table_height = float(np.asarray(data["table_height"]).item())
    table_thickness = (
        float(np.asarray(data["table_thickness"]).item())
        if "table_thickness" in data.files
        else float(os.environ.get("BOX_DEMO_TABLE_THICKNESS", "0.025"))
    )
    selected = str(np.asarray(data["selected_grasp"]).item())
    selected_sample = (
        int(np.asarray(data["selected_sample_id"]).item())
        if "selected_sample_id" in data.files
        else -1
    )
    selected_grasp_axis = (
        str(np.asarray(data["selected_grasp_axis"]).item())
        if "selected_grasp_axis" in data.files
        else "y"
    )
    selected_centering_error = (
        float(np.asarray(data["selected_centering_error"]).item())
        if "selected_centering_error" in data.files
        else float("nan")
    )

    print(f"selected grasp: {selected}")
    print(f"selected grasp sample: {selected_sample}")
    print(f"selected grasp axis: {selected_grasp_axis}")
    print(f"selected centering error: {selected_centering_error:.4f} m")
    print(f"T_box_tcp translation: {T_box_tcp[:3, 3]}")
    print(f"target box translation: {T_world_box_target[:3, 3]}")
    if box_poses is not None:
        print("box animation source: saved planned box_poses")
        final_err = np.linalg.norm(box_poses[-1, :3, 3] - T_world_box_target[:3, 3])
        print(f"final box-target position error: {final_err:.6f} m")
    else:
        print("box animation source: legacy render-TCP attachment")
    if "gripper_contact_gap" in data.files:
        print(f"contact gap: {float(np.asarray(data['gripper_contact_gap']).item()):.4f} m")
    if "gripper_hold_finger_width" in data.files:
        print(
            f"hold finger width: "
            f"{float(np.asarray(data['gripper_hold_finger_width']).item()):.4f} m"
        )
    print("box attaches only after contact/hold phase")

    make_prismatic_render_urdf(BASE_RENDER_URDF, GRIP_RENDER_URDF)

    plotter = pv.Plotter(window_size=(1280, 800))
    plotter.set_background("white")

    robot = Robot(str(GRIP_RENDER_URDF), plotter)
    q0 = build_full_q(robot, path[0], finger_widths[0])
    robot.update(q0)

    plotter.add_mesh(
        table_mesh(table_bounds, table_height, table_thickness),
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

    if SHOW_CONTACT_DEBUG and selected == "top":
        cx = float(T_box_tcp[0, 3])
        cy = float(T_box_tcp[1, 3])
        contact_points_box = [
            np.array([cx, cy - box_size[1] / 2.0, box_size[2] / 2.0, 1.0]),
            np.array([cx, cy + box_size[1] / 2.0, box_size[2] / 2.0, 1.0]),
        ]
        contact_colors = ["red", "blue"]
        for point_box, color in zip(contact_points_box, contact_colors):
            point_world = (T_world_box_initial @ point_box)[:3]
            plotter.add_mesh(
                pv.Sphere(radius=0.006, center=point_world),
                color=color,
                opacity=1.0,
                show_edges=False,
            )

    target_actor = plotter.add_mesh(
        box_mesh(box_size),
        color="dodgerblue",
        opacity=0.25,
        show_edges=True,
    )
    target_actor.user_matrix = T_world_box_target

    plotter.add_text(
        f"box grasp TSR demo | selected grasp: {selected} | sample: {selected_sample}",
        position="upper_left",
        font_size=10,
        color="black",
        name="title",
    )

    plotter.add_axes()
    plotter.camera_position = [(0.95, -0.75, 0.55), (0.35, 0.35, 0.12), (0, 0, 1)]
    plotter.show(interactive_update=True, auto_close=False)

    print("Animating. Close window to stop.")

    try:
        for i in range(0, len(path), FRAME_STRIDE):
            q_arm = path[i]
            q_full = build_full_q(robot, q_arm, finger_widths[i])
            robot.update(q_full)

            state = str(box_pose_state[i]) if len(box_pose_state) == len(path) else "attached" if attached[i] else "initial"
            if box_poses is not None:
                box_actor.user_matrix = box_poses[i]
            else:
                if state == "attached":
                    T_tcp = np.asarray(robot.fk(q_full, ee_link_name=TCP_FRAME), dtype=float)
                    # We planned T_world_tcp = T_world_box @ T_box_tcp.
                    # Therefore T_world_box = T_world_tcp @ inv(T_box_tcp).
                    box_actor.user_matrix = T_tcp @ np.linalg.inv(T_box_tcp)
                elif state == "target":
                    box_actor.user_matrix = T_world_box_target
                else:
                    box_actor.user_matrix = T_world_box_initial

            if i % 10 == 0:
                contact = bool(contact_reached[i]) if len(contact_reached) == len(path) else False
                held = state == "attached"
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
