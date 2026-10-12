"""
collision.py
============
Analytic primitive obstacles (ellipsoid / cylinder / box).  Each one answers a
single question: "is this 3-D point inside me?"  We use that to turn collision
into just another inequality constraint (see constraints.CollisionConstraint).

These classes are the mentor's code, kept verbatim.  No PyBullet involved.
"""

import numpy as np


class EllipsoidCollision:
    def __init__(self, obstacle):
        self.obstacle = obstacle
        self.T = np.array(obstacle["T"])
        self.radii = np.array([obstacle["xradius"], obstacle["yradius"], obstacle["zradius"]])
        self.rotation_inv = np.linalg.inv(self.T[:3, :3])

    def is_in_collision(self, point):
        local_point = self.rotation_inv @ (point - self.T[:3, 3])
        value = np.sum((local_point / self.radii) ** 2)
        return value <= 1.0


class CylinderCollision:
    def __init__(self, obstacle):
        self.obstacle = obstacle
        self.T = np.array(obstacle["T"])
        self.radius = obstacle["radius"]
        self.height = obstacle["height"]
        self.rotation_inv = np.linalg.inv(self.T[:3, :3])

    def is_in_collision(self, point):
        local_point = self.rotation_inv @ (point - self.T[:3, 3])
        radial_dist_sq = local_point[0] ** 2 + local_point[1] ** 2
        within_radius = radial_dist_sq <= self.radius ** 2
        within_height = abs(local_point[2]) <= self.height / 2.0
        return within_radius and within_height


class BoxCollision:
    def __init__(self, obstacle):
        self.obstacle = obstacle
        self.T = np.array(obstacle["T"])
        self.size = np.array([obstacle["xsize"], obstacle["ysize"], obstacle["zsize"]])
        self.rotation_inv = np.linalg.inv(self.T[:3, :3])

    def is_in_collision(self, point):
        local_point = self.rotation_inv @ (point - self.T[:3, 3])
        within_x = abs(local_point[0]) <= self.size[0] / 2.0
        within_y = abs(local_point[1]) <= self.size[1] / 2.0
        within_z = abs(local_point[2]) <= self.size[2] / 2.0
        return within_x and within_y and within_z


def create_collision_objects(obstacles_list):
    obstacles = []
    for obstacle in obstacles_list:
        t = obstacle["type"]
        if t == "ellipsoid":
            obstacles.append(EllipsoidCollision(obstacle))
        elif t == "cylinder":
            obstacles.append(CylinderCollision(obstacle))
        elif t == "box":
            obstacles.append(BoxCollision(obstacle))
        else:
            raise ValueError(f"Unknown obstacle type: {t}")
    return obstacles