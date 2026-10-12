"""
scene.py
========
A scene is just data: which URDF, which controlled frame, the joint torque
limits, the support surfaces (tables/shelves), and a list of primitive
obstacles.  Start/goal poses do NOT live here -- those are queries handed to the
planner, which keeps the method general.

v2 change: a scene may have SEVERAL surfaces (e.g. two tables) instead of one
`table_height`.  Old single-table configs still work unchanged:
`scene.table_height` resolves to the first surface, so verify.py / visualize.py
need no edits.
"""

import numpy as np

from robot_model import RobotModel
from collision import create_collision_objects

# Franka FER actuator torque limits (Nm), from the URDF effort tags.
FRANKA_TAU_MAX = np.array([87., 87., 87., 87., 12., 12., 12.])


class Surface:
    """A horizontal support plane at z = height, optionally a finite xy patch."""

    def __init__(self, name, height, xy_bounds=None):
        self.name = str(name)
        self.height = float(height)
        # xy_bounds = [[x_lo, y_lo], [x_hi, y_hi]] or None (infinite plane)
        self.xy_bounds = None if xy_bounds is None else np.asarray(xy_bounds, float)

    def contains_xy(self, x, y):
        if self.xy_bounds is None:
            return True
        (xlo, ylo), (xhi, yhi) = self.xy_bounds
        return (xlo <= x <= xhi) and (ylo <= y <= yhi)

    def __repr__(self):
        return f"Surface({self.name!r}, h={self.height:.3f})"


class Scene:
    def __init__(self, config):
        self.config = config
        self.robot = RobotModel(config["urdf"])
        self.controlled_frame = config.get("controlled_frame", "payload")
        self.tau_max = np.asarray(config.get("tau_max", FRANKA_TAU_MAX), float)
        self.derate = float(config.get("torque_derate", 0.85))
        self.collision_frames = config.get("collision_frames",
                                            ["payload", "fer_hand"])
        self.collision_frame_points = config.get("collision_frame_points", {})
        self.self_collision_frames = config.get("self_collision_frames", [])
        self.self_collision_min_distance = float(config.get("self_collision_min_distance", 0.0))

        # surfaces: explicit list wins; else synthesise one from table_height.
        surfaces = config.get("surfaces")
        if surfaces:
            self.surfaces = [Surface(s["name"], s["height"], s.get("xy_bounds"))
                             for s in surfaces]
        else:
            self.surfaces = [Surface("table", float(config.get("table_height", 0.0)))]
        self._by_name = {s.name: s for s in self.surfaces}
        self.table_collision_margin = float(config.get("table_collision_margin", 0.0))
        obstacle_specs = list(config.get("obstacles", []))
        if self.table_collision_margin > 0.0:
            obstacle_specs.extend(self._table_collision_obstacles())
        self.obstacles = create_collision_objects(obstacle_specs)

    def _table_collision_obstacles(self):
        obstacles = []
        thickness = max(self.table_collision_margin, 1e-3)
        for surface in self.surfaces:
            if surface.xy_bounds is None:
                continue
            h = float(surface.height)
            # Solid slab occupying [h - thickness, h] in z, so any payload
            # point dipping below the surface penetrates it.
            z_center = h - 0.5 * thickness
            (xlo, ylo), (xhi, yhi) = np.asarray(surface.xy_bounds, dtype=float).reshape(2, 2)
            T = np.eye(4, dtype=float)
            T[:3, 3] = [0.5 * (xlo + xhi), 0.5 * (ylo + yhi), z_center]
            obstacles.append(
                {
                    "type": "box",
                    "name": f"table_{surface.name}_solid",
                    "T": T,
                    "xsize": float(xhi - xlo),
                    "ysize": float(yhi - ylo),
                    "zsize": float(thickness),
                }
            )
        return obstacles

    @property
    def table_height(self):
        """Back-compat: first surface height (used by old single-table scripts)."""
        return self.surfaces[0].height

    def surface(self, name):
        return self._by_name[name]

    @classmethod
    def from_json(cls, path):
        import json
        with open(path) as f:
            return cls(json.load(f))
