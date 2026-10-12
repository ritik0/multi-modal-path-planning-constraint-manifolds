"""Case-aware compound payload model for the Franka transport task.

This module is the single source of truth for payload geometry, mass
properties, collision samples, support contact data, and generated planning
URDF variants.

Frame convention:
  - origin = table contact at local z = 0
  - the body extends in local negative z toward the gripper
  - x = depth/front-back
  - y = wing/side direction where bricks stick out
  - z = vertical/gripped length
"""

from __future__ import annotations

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


EXTRUSION_SIZE = (0.03, 0.05, 0.065)  # x, y, z in metres (z = measured gripper-to-ground)
EXTRUSION_LEN = EXTRUSION_SIZE[2]
BRICK = (0.10, 0.05, 0.025)  # x, y, z in metres
Y_OFF = EXTRUSION_SIZE[1] / 2.0 + BRICK[1] / 2.0


def _layer_z(layer: int) -> float:
    return -(int(layer) + 0.5) * BRICK[2]


PAYLOAD_CASES = {
    "2p1": [
        ("extrusion", 0.100, EXTRUSION_SIZE, (0.0, 0.0, -EXTRUSION_LEN / 2.0)),
        ("brick_L", 0.994, BRICK, (0.0, -Y_OFF, _layer_z(0))),
        ("brick_R", 1.007, BRICK, (0.0, +Y_OFF, _layer_z(0))),
    ],
    "3p1": [
        ("extrusion", 0.100, EXTRUSION_SIZE, (0.0, 0.0, -EXTRUSION_LEN / 2.0)),
        ("brick_L", 0.994, BRICK, (0.0, -Y_OFF, _layer_z(0))),
        ("brick_L2", 1.000, BRICK, (0.0, -Y_OFF, _layer_z(1))),
        ("brick_R", 1.007, BRICK, (0.0, +Y_OFF, _layer_z(0))),
    ],
    "4p1": [
        ("extrusion", 0.100, EXTRUSION_SIZE, (0.0, 0.0, -EXTRUSION_LEN / 2.0)),
        ("brick_L", 0.994, BRICK, (0.0, -Y_OFF, _layer_z(0))),
        ("brick_L2", 1.000, BRICK, (0.0, -Y_OFF, _layer_z(1))),
        ("brick_R", 1.007, BRICK, (0.0, +Y_OFF, _layer_z(0))),
        ("brick_R2", 1.000, BRICK, (0.0, +Y_OFF, _layer_z(1))),
    ],
}

DEFAULT_PAYLOAD_CASE = "3p1"


def normalise_case(case=None) -> str:
    if case is None:
        case = os.environ.get("PAYLOAD_CASE", DEFAULT_PAYLOAD_CASE)
    case = str(case).strip().lower().replace(".", "p")
    aliases = {
        "2kg": "2p1",
        "2": "2p1",
        "3kg": "3p1",
        "3": "3p1",
        "4kg": "4p1",
        "4": "4p1",
    }
    case = aliases.get(case, case)
    if case not in PAYLOAD_CASES:
        raise ValueError(
            f"Unknown PAYLOAD_CASE={case!r}; use one of {sorted(PAYLOAD_CASES)}"
        )
    return case


def components_for_case(case=None):
    return list(PAYLOAD_CASES[normalise_case(case)])


def box_inertia(mass, size):
    lx, ly, lz = np.asarray(size, dtype=float)
    mass = float(mass)
    return np.diag(
        [
            mass * (ly * ly + lz * lz) / 12.0,
            mass * (lx * lx + lz * lz) / 12.0,
            mass * (lx * lx + ly * ly) / 12.0,
        ]
    )


def compute(components=None, case=None):
    if components is None:
        components = components_for_case(case)
    mass = float(sum(component[1] for component in components))
    com = sum(
        float(component_mass) * np.asarray(center, dtype=float)
        for _, component_mass, _, center in components
    ) / mass
    inertia = np.zeros((3, 3), dtype=float)
    for _, component_mass, size, center in components:
        offset = np.asarray(center, dtype=float) - com
        inertia += box_inertia(component_mass, size) + float(component_mass) * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )
    return mass, com, inertia


def component_boxes(case=None, components=None):
    if components is None:
        components = components_for_case(case)
    return [
        {
            "name": name,
            "mass": float(mass),
            "size": np.asarray(size, dtype=float),
            "center": np.asarray(center, dtype=float),
        }
        for name, mass, size, center in components
    ]


def payload_points(components=None, case=None):
    if components is None:
        components = components_for_case(case)
    points = []
    for _, _, size, center in components:
        half = 0.5 * np.asarray(size, dtype=float)
        center = np.asarray(center, dtype=float)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    points.append(center + np.asarray([sx, sy, sz]) * half)
        points.append(center)
    return np.asarray(points, dtype=float)


def inertial_properties(case=None):
    mass, com, inertia = compute(case=case)
    return float(mass), np.asarray(com, dtype=float), np.asarray(inertia, dtype=float)


def extrusion_len(case=None):
    normalise_case(case)
    return float(EXTRUSION_LEN)


def _fmt(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in np.asarray(values).reshape(-1))


def write_payload_urdf_variant(base_urdf_path, out_urdf_path, case=None):
    case = normalise_case(case)
    base_urdf_path = Path(base_urdf_path)
    out_urdf_path = Path(out_urdf_path)
    tree = ET.parse(base_urdf_path)
    root = tree.getroot()
    payload_link = root.find(".//link[@name='payload']")
    if payload_link is None:
        raise ValueError(f"payload link missing from {base_urdf_path}")

    for child in list(payload_link):
        if child.tag in {"inertial", "visual", "collision"}:
            payload_link.remove(child)

    mass, com, inertia = inertial_properties(case)
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", {"rpy": "0 0 0", "xyz": _fmt(com)})
    ET.SubElement(inertial, "mass", {"value": f"{mass:.9g}"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{inertia[0, 0]:.9g}",
            "ixy": f"{inertia[0, 1]:.9g}",
            "ixz": f"{inertia[0, 2]:.9g}",
            "iyy": f"{inertia[1, 1]:.9g}",
            "iyz": f"{inertia[1, 2]:.9g}",
            "izz": f"{inertia[2, 2]:.9g}",
        },
    )
    payload_link.insert(0, inertial)

    for box in component_boxes(case=case):
        name = str(box["name"])
        color = "0.35 0.35 0.35 1" if name == "extrusion" else "0.45 0.08 0.08 1"
        visual = ET.SubElement(payload_link, "visual", {"name": f"payload_{name}_visual"})
        ET.SubElement(visual, "origin", {"rpy": "0 0 0", "xyz": _fmt(box["center"])})
        visual_geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(visual_geometry, "box", {"size": _fmt(box["size"])})
        material = ET.SubElement(visual, "material", {"name": f"payload_{name}_material"})
        ET.SubElement(material, "color", {"rgba": color})

        collision = ET.SubElement(
            payload_link, "collision", {"name": f"payload_{name}_collision"}
        )
        ET.SubElement(collision, "origin", {"rpy": "0 0 0", "xyz": _fmt(box["center"])})
        collision_geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(collision_geometry, "box", {"size": _fmt(box["size"])})

    # Keep the grasp height in sync with the extrusion length (single source of
    # truth): the payload (table-contact) frame sits EXTRUSION_LEN below the TCP
    # along the approach axis, i.e. the gripper holds the extrusion at its TOP.
    # Without this the joint origin can go stale when EXTRUSION_LEN changes,
    # which desyncs the planner (URDF joint) from the renderer (grasp transform).
    payload_joint = root.find(".//joint[@name='payload_joint']")
    if payload_joint is not None:
        joint_origin = payload_joint.find("origin")
        if joint_origin is None:
            joint_origin = ET.SubElement(payload_joint, "origin")
        joint_origin.set("rpy", "0 0 0")
        joint_origin.set("xyz", f"0 0 {EXTRUSION_LEN:.9g}")

    out_urdf_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(out_urdf_path, encoding="utf-8", xml_declaration=True)
    return str(out_urdf_path)


def ensure_payload_urdf_for_case(base_urdf_path, output_dir, case=None):
    case = normalise_case(case)
    output_dir = Path(output_dir) / "_generated_payload_urdfs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"fer_franka_hand_payload_{case}.urdf"
    write_payload_urdf_variant(base_urdf_path, out, case=case)
    return str(out)


ACTIVE_PAYLOAD_CASE = normalise_case()
COMPONENTS = components_for_case(ACTIVE_PAYLOAD_CASE)
TOTAL_MASS, COM, INERTIA = compute(COMPONENTS)
PAYLOAD_POINTS = payload_points(COMPONENTS)
SUPPORT_CONTACT_LOCAL = np.array([0.0, 0.0, 0.0], dtype=float)
SUPPORT_NORMAL_WORLD = np.array([0.0, 0.0, 1.0], dtype=float)
_payload_points = payload_points


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    print(f"active payload case = {ACTIVE_PAYLOAD_CASE}")
    print(f"total mass         = {TOTAL_MASS:.4f} kg")
    print(f"COM (payload frame)= {COM}")
    print("inertia about COM (kg m^2):")
    print(INERTIA)
    print("component boxes:")
    for payload_box in component_boxes(case=ACTIVE_PAYLOAD_CASE):
        print(
            f"  {payload_box['name']}: mass={payload_box['mass']:.3f} "
            f"center={payload_box['center']} size={payload_box['size']}"
        )
    print(f"support contact local = {SUPPORT_CONTACT_LOCAL}")
    print("\nURDF <inertial>:")
    print(f'  <origin rpy="0 0 0" xyz="{_fmt(COM)}"/>')
    print(f'  <mass value="{TOTAL_MASS:.9g}"/>')
    print(
        "  <inertia "
        f'ixx="{INERTIA[0, 0]:.9g}" ixy="{INERTIA[0, 1]:.9g}" '
        f'ixz="{INERTIA[0, 2]:.9g}" iyy="{INERTIA[1, 1]:.9g}" '
        f'iyz="{INERTIA[1, 2]:.9g}" izz="{INERTIA[2, 2]:.9g}"/>'
    )