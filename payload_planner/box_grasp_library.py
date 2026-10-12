from __future__ import annotations

import os
import math
from dataclasses import dataclass

import numpy as np


BOX_SIZE = np.array([0.12, 0.06, 0.04], dtype=float)  # x, y, z [m]
BOX_CENTER_WORLD = np.array([0.34, 0.34, BOX_SIZE[2] / 2.0], dtype=float)
BOX_TARGET_CENTER_WORLD = np.array([0.46, 0.42, BOX_SIZE[2] / 2.0], dtype=float)
TABLE_XY_BOUNDS = np.array([[0.10, 0.15], [0.58, 0.62]], dtype=float)
TABLE_HEIGHT = 0.0

PREGRASP_DISTANCE = 0.08
TOP_GRASP_CLEARANCE = 0.0
TOP_PREGRASP_DISTANCE = 0.08
GRASP_Z_TWEAK = float(os.environ.get("BOX_GRASP_Z_TWEAK", "0.0"))
LIFT_HEIGHT = 0.12
CARRY_HEIGHT = 0.16
FINGER_MAX = 0.04


@dataclass(frozen=True)
class GraspCandidate:
    name: str
    T_box_tcp: np.ndarray
    pregrasp_offset_world: np.ndarray
    required_width: float
    pos_tol: np.ndarray
    rot_tol: np.ndarray
    grasp_axis: str = "y"
    object_width: float = 0.06
    contact_margin: float = 0.002
    sample_offsets: tuple[tuple[float, float, float], ...] = ()
    sample_yaws_deg: tuple[float, ...] = (0.0,)
    description: str = ""


def rpy_to_matrix(rpy):
    r, p, y = [float(v) for v in rpy]
    cr, sr = math.cos(float(r)), math.sin(float(r))
    cp, sp = math.cos(float(p)), math.sin(float(p))
    cy, sy = math.cos(float(y)), math.sin(float(y))
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def make_T(R=None, p=None):
    T = np.eye(4, dtype=float)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    if p is not None:
        T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


def T_world_box():
    return make_T(np.eye(3), BOX_CENTER_WORLD)


def T_world_box_target():
    return make_T(np.eye(3), BOX_TARGET_CENTER_WORLD)


def _tcp_down_rotation():
    # Same convention as the existing payload examples: gripper points downward.
    return np.diag([1.0, -1.0, -1.0])


def _tcp_side_x_plus_rotation():
    # Approach from +x toward the box, while closing across the box y-width.
    R = np.eye(3, dtype=float)
    R[:, 0] = [0.0, 0.0, 1.0]
    R[:, 1] = [0.0, 1.0, 0.0]
    R[:, 2] = [-1.0, 0.0, 0.0]
    return R


def _tcp_side_x_minus_rotation():
    # Approach from -x toward the box, while closing across the box y-width.
    R = np.eye(3, dtype=float)
    R[:, 0] = [0.0, 0.0, -1.0]
    R[:, 1] = [0.0, 1.0, 0.0]
    R[:, 2] = [1.0, 0.0, 0.0]
    return R


def sampled_T_box_tcp(grasp: GraspCandidate):
    offsets = grasp.sample_offsets or ((0.0, 0.0, 0.0),)
    yaws = grasp.sample_yaws_deg or (0.0,)
    out = []
    for offset in offsets:
        T_offset = make_T(np.eye(3), np.asarray(offset, dtype=float))
        for yaw_deg in yaws:
            T_yaw = make_T(rpy_to_matrix([0.0, 0.0, np.deg2rad(float(yaw_deg))]))
            out.append(grasp.T_box_tcp @ T_offset @ T_yaw)
    return out


def grasp_library():
    sx, sy, sz = BOX_SIZE
    R_down = _tcp_down_rotation()
    contact_margin = 0.002

    # For the first demo we make the top grasp robust and side grasps approximate.
    # If side grasps fail projection, the evaluator simply rejects them.
    top = make_T(
        R_down,
        np.array(
            [0.0, 0.0, sz / 2.0 + TOP_GRASP_CLEARANCE + GRASP_Z_TWEAK],
            dtype=float,
        ),
    )

    # Side candidates: approach from +/- y face. Orientation is approximate and
    # intentionally simple for this first demo.
    R_side_plus = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        dtype=float,
    )
    R_side_minus = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )

    side_plus = make_T(
        R_side_plus,
        np.array([0.0, sy / 2.0 + 0.030, 0.0], dtype=float),
    )
    side_minus = make_T(
        R_side_minus,
        np.array([0.0, -sy / 2.0 - 0.030, 0.0], dtype=float),
    )
    side_x_plus = make_T(
        _tcp_side_x_plus_rotation(),
        np.array([sx / 2.0 + 0.030, 0.0, 0.0], dtype=float),
    )
    side_x_minus = make_T(
        _tcp_side_x_minus_rotation(),
        np.array([-sx / 2.0 - 0.030, 0.0, 0.0], dtype=float),
    )
    top_offsets = (
        (0.000, 0.000, 0.000),
    )
    side_x_offsets = (
        (0.000, 0.000, 0.000),
        (0.000, +0.008, 0.000),
        (0.000, -0.008, 0.000),
        (0.000, 0.000, +0.005),
    )

    return [
        GraspCandidate(
            name="top",
            T_box_tcp=top,
            pregrasp_offset_world=np.array(
                [0.0, 0.0, TOP_PREGRASP_DISTANCE],
                dtype=float,
            ),
            required_width=sy + 2.0 * contact_margin,
            pos_tol=np.array([0.015, 0.015, 0.005], dtype=float),
            rot_tol=np.deg2rad([8.0, 8.0, 20.0]),
            grasp_axis="y",
            object_width=sy,
            contact_margin=contact_margin,
            sample_offsets=top_offsets,
            sample_yaws_deg=(0.0,),
            description="top grasp from above",
        ),
        GraspCandidate(
            name="side_x_plus",
            T_box_tcp=side_x_plus,
            pregrasp_offset_world=np.array([PREGRASP_DISTANCE, 0.0, 0.0], dtype=float),
            required_width=sy + 2.0 * contact_margin,
            pos_tol=np.array([0.010, 0.015, 0.010], dtype=float),
            rot_tol=np.deg2rad([12.0, 12.0, 20.0]),
            grasp_axis="y",
            object_width=sy,
            contact_margin=contact_margin,
            sample_offsets=side_x_offsets,
            sample_yaws_deg=(-10.0, 0.0, 10.0),
            description="side grasp from +x face",
        ),
        GraspCandidate(
            name="side_x_minus",
            T_box_tcp=side_x_minus,
            pregrasp_offset_world=np.array([-PREGRASP_DISTANCE, 0.0, 0.0], dtype=float),
            required_width=sy + 2.0 * contact_margin,
            pos_tol=np.array([0.010, 0.015, 0.010], dtype=float),
            rot_tol=np.deg2rad([12.0, 12.0, 20.0]),
            grasp_axis="y",
            object_width=sy,
            contact_margin=contact_margin,
            sample_offsets=side_x_offsets,
            sample_yaws_deg=(-10.0, 0.0, 10.0),
            description="side grasp from -x face",
        ),
        GraspCandidate(
            name="side_y_plus",
            T_box_tcp=side_plus,
            pregrasp_offset_world=np.array([0.0, PREGRASP_DISTANCE, 0.0], dtype=float),
            required_width=sx + 2.0 * contact_margin,
            pos_tol=np.array([0.015, 0.010, 0.010], dtype=float),
            rot_tol=np.deg2rad([12.0, 12.0, 20.0]),
            grasp_axis="x",
            object_width=sx,
            contact_margin=contact_margin,
            description="side grasp from +y face (too wide for this gripper)",
        ),
        GraspCandidate(
            name="side_y_minus",
            T_box_tcp=side_minus,
            pregrasp_offset_world=np.array([0.0, -PREGRASP_DISTANCE, 0.0], dtype=float),
            required_width=sx + 2.0 * contact_margin,
            pos_tol=np.array([0.015, 0.010, 0.010], dtype=float),
            rot_tol=np.deg2rad([12.0, 12.0, 20.0]),
            grasp_axis="x",
            object_width=sx,
            contact_margin=contact_margin,
            description="side grasp from -y face (too wide for this gripper)",
        ),
    ]


def box_corners_world(T_box=None):
    T_box = T_world_box() if T_box is None else np.asarray(T_box, dtype=float)
    half = 0.5 * BOX_SIZE
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = np.array([sx * half[0], sy * half[1], sz * half[2], 1.0])
                pts.append((T_box @ local)[:3])
    return np.asarray(pts, dtype=float)
