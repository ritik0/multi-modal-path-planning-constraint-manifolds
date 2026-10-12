"""
box_payload.py  --  mass model for the simulation-only regrasp demo.

Gives the manipulated box real inertial properties and generates, per grasp, a
planning URDF where the box is rigidly attached to the gripper THROUGH that
grasp's transform.  Holding the box in different grasps therefore yields
different gravity-holding torques g(q) -- which is exactly what lets the regrasp
choice be torque-aware, and ties this demo back to the torque-limited theme of
the main thesis.

Self-contained: depends only on numpy + box_grasp_library (geometry).  It does
NOT import or modify any of the robot-project planning code, and it writes its
URDFs into a separate _generated_box_urdfs/ folder.
"""

from __future__ import annotations

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from box_grasp_library import BOX_SIZE

# Box mass in kg. Light by default (this is a sim demo, not the heavy payload);
# override with BOX_MASS=... to see the torque effect of the regrasp choice.
BOX_MASS = float(os.environ.get("BOX_MASS", "0.75"))


def box_inertia(mass, size) -> np.ndarray:
    """Inertia of a uniform box about its own centre, in the box frame."""
    lx, ly, lz = np.asarray(size, dtype=float)
    m = float(mass)
    return np.diag([
        m * (ly * ly + lz * lz) / 12.0,
        m * (lx * lx + lz * lz) / 12.0,
        m * (lx * lx + ly * ly) / 12.0,
    ])


def box_inertial(mass=None):
    """(mass, COM, inertia) for the box. Uniform box -> COM at the box-frame origin."""
    mass = BOX_MASS if mass is None else float(mass)
    com = np.zeros(3, dtype=float)
    return mass, com, box_inertia(mass, BOX_SIZE)


def _matrix_to_rpy(R) -> np.ndarray:
    """URDF fixed-axis rpy (R = Rz(yaw) Ry(pitch) Rx(roll)) from a rotation matrix."""
    R = np.asarray(R, dtype=float)
    sy = max(-1.0, min(1.0, -float(R[2, 0])))
    pitch = np.arcsin(sy)
    if abs(R[2, 0]) < 0.999999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: fix yaw = 0
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def _fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in np.asarray(v).reshape(-1))


def _se3_inv(T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def write_box_urdf_for_grasp(base_urdf_path, out_urdf_path, T_box_tcp, mass=None) -> str:
    """Generate a planning URDF: box (with mass) attached to fer_hand_tcp via a grasp.

    The payload_joint origin is set to T_tcp_box = inv(T_box_tcp) (the box pose in
    the TCP frame), and the payload link is rebuilt as the box.  The base URDF's
    arm + hand are kept untouched.
    """
    base = Path(base_urdf_path)
    out = Path(out_urdf_path)
    tree = ET.parse(base)
    root = tree.getroot()

    # 1) place the box relative to the TCP via the grasp transform
    joint = root.find(".//joint[@name='payload_joint']")
    if joint is None:
        raise ValueError(f"payload_joint missing from {base}")
    T_tcp_box = _se3_inv(T_box_tcp)
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin")
    origin.set("xyz", _fmt(T_tcp_box[:3, 3]))
    origin.set("rpy", _fmt(_matrix_to_rpy(T_tcp_box[:3, :3])))

    # 2) rebuild the payload link as the box (mass + geometry)
    link = root.find(".//link[@name='payload']")
    if link is None:
        raise ValueError(f"payload link missing from {base}")
    for child in list(link):
        if child.tag in {"inertial", "visual", "collision"}:
            link.remove(child)

    m, com, I = box_inertial(mass)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"rpy": "0 0 0", "xyz": _fmt(com)})
    ET.SubElement(inertial, "mass", {"value": f"{m:.9g}"})
    ET.SubElement(inertial, "inertia", {
        "ixx": f"{I[0, 0]:.9g}", "ixy": "0", "ixz": "0",
        "iyy": f"{I[1, 1]:.9g}", "iyz": "0", "izz": f"{I[2, 2]:.9g}",
    })

    visual = ET.SubElement(link, "visual", {"name": "box_visual"})
    ET.SubElement(visual, "origin", {"rpy": "0 0 0", "xyz": "0 0 0"})
    vg = ET.SubElement(visual, "geometry")
    ET.SubElement(vg, "box", {"size": _fmt(BOX_SIZE)})
    mat = ET.SubElement(visual, "material", {"name": "box_material"})
    ET.SubElement(mat, "color", {"rgba": "0.70 0.33 0.05 1"})

    collision = ET.SubElement(link, "collision", {"name": "box_collision"})
    ET.SubElement(collision, "origin", {"rpy": "0 0 0", "xyz": "0 0 0"})
    cg = ET.SubElement(collision, "geometry")
    ET.SubElement(cg, "box", {"size": _fmt(BOX_SIZE)})

    out.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return str(out)


def ensure_box_urdf_for_grasp(base_urdf_path, output_dir, grasp_name, T_box_tcp, mass=None) -> str:
    """Write (once) and return the path to the box-URDF for a named grasp."""
    output_dir = Path(output_dir) / "_generated_box_urdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"box_grasp_{grasp_name}.urdf"
    return write_box_urdf_for_grasp(base_urdf_path, out, T_box_tcp, mass=mass)


if __name__ == "__main__":
    m, com, I = box_inertial()
    np.set_printoptions(suppress=True)
    print(f"box size (m)   : {np.asarray(BOX_SIZE)}")
    print(f"box mass (kg)  : {m:.4f}")
    print(f"box COM (m)    : {com}")
    print("box inertia (kg m^2):")
    print(I)