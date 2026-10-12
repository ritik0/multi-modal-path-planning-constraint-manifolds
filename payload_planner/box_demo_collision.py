from __future__ import annotations

import os
import numpy as np

from constraints import CollisionConstraint


class BoxDemoCollisionError(RuntimeError):
    pass


ROBOT_TABLE_PROXY_FRAMES = [
    "fer_link1",
    "fer_link2",
    "fer_link3",
    "fer_link4",
    "fer_link5",
    "fer_link6",
    "fer_link7",
    "fer_link8",
    "fer_hand",
    "fer_hand_tcp",
    "fer_leftfinger",
    "fer_rightfinger",
]
ROBOT_BASE_PROXY_FRAMES = [
    "fer_link4",
    "fer_link5",
    "fer_link6",
    "fer_link7",
    "fer_link8",
    "fer_hand",
    "fer_hand_tcp",
    "fer_leftfinger",
    "fer_rightfinger",
]
BOX_GRIPPER_ALLOWED_CONTACT_FRAMES = {
    "fer_link8",
    "fer_hand",
    "fer_hand_tcp",
    "fer_leftfinger",
    "fer_rightfinger",
}
BOX_GRIPPER_ALLOWED_PHASE_PREFIXES = (
    "close_",
    "carry_",
    "arrive_",
    "release_",
    "grasp_",
    "hold_",
    "transit_",
    "approach_",
)
DEFAULT_FRAME_RADIUS = float(os.environ.get("BOX_DEMO_FRAME_RADIUS", "0.035"))
FINGER_FRAME_RADIUS = float(os.environ.get("BOX_DEMO_FINGER_FRAME_RADIUS", "0.020"))
HAND_FRAME_RADIUS = float(os.environ.get("BOX_DEMO_HAND_FRAME_RADIUS", "0.045"))
BASE_RADIUS = float(os.environ.get("BOX_DEMO_BASE_RADIUS", "0.16"))
BASE_HEIGHT = float(os.environ.get("BOX_DEMO_BASE_HEIGHT", "0.18"))
BASE_MARGIN = float(os.environ.get("BOX_DEMO_BASE_MARGIN", "0.015"))


def _frame_radius(frame):
    frame = str(frame).lower()
    if "finger" in frame:
        return FINGER_FRAME_RADIUS
    if "hand" in frame:
        return HAND_FRAME_RADIUS
    if frame in {"fer_link1", "fer_link2", "fer_link3"}:
        return float(os.environ.get("BOX_DEMO_LOWER_LINK_RADIUS", "0.030"))
    return DEFAULT_FRAME_RADIUS


def _table_frame_radius(frame):
    frame = str(frame).lower()
    if frame == "fer_hand_tcp":
        return float(os.environ.get("BOX_DEMO_TCP_TABLE_RADIUS", "0.003"))
    if "finger" in frame:
        return float(os.environ.get("BOX_DEMO_FINGER_TABLE_RADIUS", "0.006"))
    if frame == "fer_hand":
        return float(os.environ.get("BOX_DEMO_HAND_TABLE_RADIUS", "0.020"))
    return _frame_radius(frame)


def _box_corners_world(T, size):
    T = np.asarray(T, dtype=float).reshape(4, 4)
    half = 0.5 * np.asarray(size, dtype=float).reshape(3)
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=float)
                pts.append(T[:3, 3] + T[:3, :3] @ local)
    return np.asarray(pts, dtype=float)


def _points_in_box_frame(T_box, points):
    T_box = np.asarray(T_box, dtype=float).reshape(4, 4)
    points = np.asarray(points, dtype=float).reshape((-1, 3))
    return (points - T_box[:3, 3]) @ T_box[:3, :3]


def _box_gripper_contact_allowed(frame, state, phase):
    if str(frame) not in BOX_GRIPPER_ALLOWED_CONTACT_FRAMES:
        return False
    if str(state) == "attached":
        return True
    phase = str(phase)
    return any(phase.startswith(prefix) for prefix in BOX_GRIPPER_ALLOWED_PHASE_PREFIXES)


class BoxDemoCollisionChecker:
    def __init__(
        self,
        box_size,
        table_xy_bounds,
        table_height,
        scene=None,
        scene_robot=None,
        table_thickness=0.025,
        robot_table_margin=0.002,
        box_table_tol=0.0015,
        verbose=False,
    ):
        self.scene = scene
        self.scene_robot = scene.robot if scene is not None else scene_robot
        self.box_size = np.asarray(box_size, dtype=float).reshape(3)
        table_xy_bounds = np.asarray(table_xy_bounds, dtype=float)
        if table_xy_bounds.shape == (2, 2):
            self.table_xy_bounds = table_xy_bounds.reshape(1, 2, 2)
        else:
            self.table_xy_bounds = table_xy_bounds.reshape(-1, 2, 2)
        self.table_height = float(table_height)
        self.table_thickness = float(table_thickness)
        self.robot_table_margin = float(robot_table_margin)
        self.box_table_tol = float(box_table_tol)
        self.verbose = bool(verbose)
        self.robot_collision = None
        if scene is not None:
            self.robot_collision = CollisionConstraint(
                scene.robot,
                scene.collision_frames,
                scene.obstacles,
                frame_points=scene.collision_frame_points,
                self_collision_frames=scene.self_collision_frames,
                self_collision_min_distance=scene.self_collision_min_distance,
            )

    def close(self):
        pass

    def box_table_valid_manual(self, T_box, state, phase):
        corners = _box_corners_world(T_box, self.box_size)
        z_min = float(np.min(corners[:, 2]))

        if z_min < self.table_height - self.box_table_tol:
            return False, (
                f"box penetrates table: z_min={z_min:.5f}, "
                f"table={self.table_height:.5f}, state={state}, phase={phase}"
            )

        if str(state) in {"initial", "target", "resting", "release_gripper", "retreat"}:
            if abs(z_min - self.table_height) > 0.006:
                return False, (
                    f"resting box not on table: z_min={z_min:.5f}, "
                    f"table={self.table_height:.5f}, state={state}, phase={phase}"
                )

        return True, "ok"

    def _point_near_table_xy(self, p, margin=0.0):
        for bounds in self.table_xy_bounds:
            (xlo, ylo), (xhi, yhi) = bounds
            if (
                xlo - margin <= p[0] <= xhi + margin
                and ylo - margin <= p[1] <= yhi + margin
            ):
                return True
        return False

    def robot_table_valid_fk(self, q_arm, phase):
        if self.scene_robot is None:
            raise BoxDemoCollisionError("scene_robot is required for FK robot/table collision checking.")

        points = {}
        for frame in ROBOT_TABLE_PROXY_FRAMES:
            try:
                points[frame] = np.asarray(self.scene_robot.frame_pose(frame, q_arm)[:3, 3], dtype=float)
            except Exception:
                continue
        return self.robot_table_proxy_points_valid(points, phase)

    def robot_table_proxy_points_valid(self, frame_points, phase):
        for frame, p in frame_points.items():
            r = _table_frame_radius(frame)
            if self._point_near_table_xy(p, margin=0.08):
                if p[2] - r < self.table_height:
                    return False, (
                        f"robot-table FK proxy collision: frame={frame}, "
                        f"p={np.round(p, 4)}, radius={r:.3f}, "
                        f"bottom={p[2] - r:.4f}, table={self.table_height:.4f}, "
                        f"phase={phase}"
                    )

        return True, "ok"

    def robot_environment_valid_collisionpy(self, q_arm, phase):
        if self.robot_collision is None:
            raise BoxDemoCollisionError("scene is required for collision.py robot collision checking.")
        hit = self.robot_collision.hit_info(q_arm)
        if hit is None:
            return True, "ok"
        if hit.get("kind") == "self":
            return False, (
                f"robot self-collision: frame_a={hit.get('frame_a')}, "
                f"frame_b={hit.get('frame_b')}, distance={hit.get('distance'):.4f}, "
                f"phase={phase}"
            )
        return False, (
            f"robot-environment collision: frame={hit.get('frame')}, "
            f"object={hit.get('object_name')}, point={np.round(hit.get('point'), 4)}, "
            f"phase={phase}"
        )

    def robot_base_valid_fk(self, q_arm, phase):
        if self.scene_robot is None:
            raise BoxDemoCollisionError("scene_robot is required for FK robot/base collision checking.")

        for frame in ROBOT_BASE_PROXY_FRAMES:
            try:
                p = np.asarray(self.scene_robot.frame_pose(frame, q_arm)[:3, 3], dtype=float)
            except Exception:
                continue

            r = _frame_radius(frame)
            radial = float(np.linalg.norm(p[:2]))
            if radial - r < BASE_RADIUS + BASE_MARGIN and p[2] - r < BASE_HEIGHT + BASE_MARGIN:
                return False, (
                    f"robot-base FK proxy collision: frame={frame}, "
                    f"p={np.round(p, 4)}, radius={r:.3f}, radial={radial:.4f}, "
                    f"base_radius={BASE_RADIUS:.4f}, phase={phase}"
                )

        return True, "ok"

    def box_base_valid_manual(self, T_box, phase):
        corners = _box_corners_world(T_box, self.box_size)
        for p in corners:
            radial = float(np.linalg.norm(p[:2]))
            if radial < BASE_RADIUS + BASE_MARGIN and p[2] < BASE_HEIGHT + BASE_MARGIN:
                return False, (
                    f"box-base collision: corner={np.round(p, 4)}, radial={radial:.4f}, "
                    f"base_radius={BASE_RADIUS:.4f}, phase={phase}"
                )
        return True, "ok"

    def box_robot_valid_fk(self, q_arm, T_box, state, phase):
        if self.scene_robot is None:
            raise BoxDemoCollisionError("scene_robot is required for FK box/robot collision checking.")

        half = 0.5 * self.box_size
        for frame in ROBOT_TABLE_PROXY_FRAMES:
            if _box_gripper_contact_allowed(frame, state, phase):
                continue
            try:
                p = np.asarray(self.scene_robot.frame_pose(frame, q_arm)[:3, 3], dtype=float)
            except Exception:
                continue
            r = _frame_radius(frame)
            local = _points_in_box_frame(T_box, p.reshape(1, 3))[0]
            if bool(np.all(np.abs(local) <= half + r)):
                return False, (
                    f"box-robot FK proxy collision: frame={frame}, "
                    f"box_local={np.round(local, 4)}, radius={r:.3f}, phase={phase}"
                )
        return True, "ok"

    def sample_valid(self, q_arm, finger_width, T_box, state, phase):
        ok, reason = self.robot_environment_valid_collisionpy(q_arm, phase)
        if not ok:
            return False, reason

        ok, reason = self.robot_table_valid_fk(q_arm, phase)
        if not ok:
            return False, reason

        ok, reason = self.robot_base_valid_fk(q_arm, phase)
        if not ok:
            return False, reason

        ok, reason = self.box_table_valid_manual(T_box, state, phase)
        if not ok:
            return False, reason

        ok, reason = self.box_base_valid_manual(T_box, phase)
        if not ok:
            return False, reason

        ok, reason = self.box_robot_valid_fk(q_arm, T_box, state, phase)
        if not ok:
            return False, reason

        return True, "ok"
