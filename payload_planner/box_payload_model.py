"""
box_payload_model.py  --  single-box payload, same interface as payload_model.py.

A simulation-only drop-in payload for the regrasp study: instead of the
brick-stack cases (2p1/3p1/4p1), the payload is ONE box of configurable size and
mass. It mirrors payload_model.py exactly where the transport pipeline touches it
-- compute(), inertial_properties(), component_boxes(), payload_points(), and
write_payload_urdf_variant()/ensure_payload_urdf_for_case() -- so transport.plan,
build_modes, collision and certification all behave identically to the working
payload runs. Only the geometry (a box) and the per-grasp attachment differ.

Frame convention matches payload_model.py:
  origin = table contact at local z = 0; body extends toward the gripper in -z.
Nothing here imports or edits the robot-project files.
"""

from __future__ import annotations

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

# Box geometry shared with the grasp library (single source of truth).
from box_grasp_library import BOX_SIZE  # [x, y, z] = [0.12, 0.06, 0.04] m

BOX_MASS = float(os.environ.get("BOX_PAYLOAD_MASS", "2.0"))   # kg (your number)


def box_inertia(mass, size):
    lx, ly, lz = np.asarray(size, dtype=float)
    mass = float(mass)
    return np.diag([
        mass * (ly * ly + lz * lz) / 12.0,
        mass * (lx * lx + lz * lz) / 12.0,
        mass * (lx * lx + ly * ly) / 12.0,
    ])


def _grasp_height():
    """Distance from the table-contact frame (z=0) to the TCP along approach.

    For a box held from the top the TCP sits at the box's upper face, i.e. the
    full box height above the table-contact origin. Kept as the single source of
    truth for the payload_joint origin, exactly like EXTRUSION_LEN in
    payload_model.py.
    """
    return float(BOX_SIZE[2])


def compute(case=None):
    """(mass, COM, inertia) of the box about its COM, in the payload frame.

    The box is centred in x and y; its COM sits half a height below the upper
    face. The payload frame origin is the table contact (z=0), so the centre is
    at z = -BOX_SIZE[2]/2 (body extends toward the gripper in -z).
    """
    mass = BOX_MASS
    com = np.array([0.0, 0.0, -0.5 * float(BOX_SIZE[2])], dtype=float)
    inertia = box_inertia(mass, BOX_SIZE)
    return float(mass), com, inertia


def inertial_properties(case=None):
    mass, com, inertia = compute(case=case)
    return float(mass), np.asarray(com, float), np.asarray(inertia, float)


def component_boxes(case=None):
    com = np.array([0.0, 0.0, -0.5 * float(BOX_SIZE[2])], dtype=float)
    return [{"name": "box", "mass": float(BOX_MASS),
             "size": np.asarray(BOX_SIZE, float), "center": com}]


def payload_points(case=None):
    half = 0.5 * np.asarray(BOX_SIZE, float)
    center = np.array([0.0, 0.0, -0.5 * float(BOX_SIZE[2])], dtype=float)
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append(center + np.asarray([sx, sy, sz]) * half)
    pts.append(center)
    return np.asarray(pts, dtype=float)


def _fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in np.asarray(values).reshape(-1))


def _matrix_to_rpy(R):
    R = np.asarray(R, float)
    sy = max(-1.0, min(1.0, -float(R[2, 0])))
    pitch = np.arcsin(sy)
    if abs(R[2, 0]) < 0.999999:
        roll = np.arctan2(R[2, 1], R[2, 2]); yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1]); yaw = 0.0
    return np.array([roll, pitch, yaw], float)


def _se3_inv(T):
    T = np.asarray(T, float); R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4); Ti[:3, :3] = R.T; Ti[:3, 3] = -R.T @ p
    return Ti


def write_payload_urdf_variant(base_urdf_path, out_urdf_path, case=None, T_box_tcp=None):
    """Generate the box planning URDF.

    If T_box_tcp is given (a grasp transform, TCP pose in the box frame), the
    payload_joint is set to inv(T_box_tcp) so the box is attached THROUGH that
    grasp and g(q) reflects holding it that way. If omitted, the default top grasp
    (payload frame straight under the TCP at the box height) is used, matching the
    payload_model.py convention.
    """
    base = Path(base_urdf_path); out = Path(out_urdf_path)
    tree = ET.parse(base); root = tree.getroot()

    link = root.find(".//link[@name='payload']")
    if link is None:
        raise ValueError(f"payload link missing from {base}")
    for child in list(link):
        if child.tag in {"inertial", "visual", "collision"}:
            link.remove(child)

    mass, com, inertia = inertial_properties(case)
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", {"rpy": "0 0 0", "xyz": _fmt(com)})
    ET.SubElement(inertial, "mass", {"value": f"{mass:.9g}"})
    ET.SubElement(inertial, "inertia", {
        "ixx": f"{inertia[0,0]:.9g}", "ixy": "0", "ixz": "0",
        "iyy": f"{inertia[1,1]:.9g}", "iyz": "0", "izz": f"{inertia[2,2]:.9g}"})
    link.insert(0, inertial)

    box = component_boxes(case=case)[0]
    visual = ET.SubElement(link, "visual", {"name": "payload_box_visual"})
    ET.SubElement(visual, "origin", {"rpy": "0 0 0", "xyz": _fmt(box["center"])})
    vg = ET.SubElement(visual, "geometry"); ET.SubElement(vg, "box", {"size": _fmt(box["size"])})
    mat = ET.SubElement(visual, "material", {"name": "payload_box_material"})
    ET.SubElement(mat, "color", {"rgba": "0.70 0.33 0.05 1"})
    collision = ET.SubElement(link, "collision", {"name": "payload_box_collision"})
    ET.SubElement(collision, "origin", {"rpy": "0 0 0", "xyz": _fmt(box["center"])})
    cg = ET.SubElement(collision, "geometry"); ET.SubElement(cg, "box", {"size": _fmt(box["size"])})

    payload_joint = root.find(".//joint[@name='payload_joint']")
    if payload_joint is not None:
        jo = payload_joint.find("origin")
        if jo is None:
            jo = ET.SubElement(payload_joint, "origin")
        if T_box_tcp is None:
            jo.set("rpy", "0 0 0"); jo.set("xyz", f"0 0 {_grasp_height():.9g}")
        else:
            T_tcp_box = _se3_inv(T_box_tcp)
            jo.set("xyz", _fmt(T_tcp_box[:3, 3]))
            jo.set("rpy", _fmt(_matrix_to_rpy(T_tcp_box[:3, :3])))

    out.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return str(out)


def ensure_payload_urdf_for_case(base_urdf_path, output_dir, case=None, tag="box", T_box_tcp=None):
    output_dir = Path(output_dir) / "_generated_box_urdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"fer_franka_hand_payload_{tag}.urdf"
    write_payload_urdf_variant(base_urdf_path, out, case=case, T_box_tcp=T_box_tcp)
    return str(out)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    m, com, I = compute()
    print(f"box size (m)  = {np.asarray(BOX_SIZE)}")
    print(f"box mass (kg) = {m:.4f}")
    print(f"COM (payload frame) = {com}")
    print("inertia about COM (kg m^2):"); print(I)