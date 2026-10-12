"""Animate the separated-table payload plan in Meshcat.

Loads ``panda_plan.npz`` from this folder, draws the two separated tables, marks
start/goal/lift-off/set-down payload positions, traces the payload path, and
loops over the dense joint path.

If the URDF's visual meshes can be resolved, the solid Franka model is shown.
If they cannot (e.g. the URDF names meshes that aren't on disk), the arm is
rendered as a forward-kinematics SKELETON -- joint spheres joined by a line --
so the motion is still visible without any mesh files.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pinocchio
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from pinocchio.visualize import MeshcatVisualizer


HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_PATH = os.path.join(HERE, "panda_plan.npz")
MARKER_Z_OFFSET = 0.012


def _as_str(value) -> str:
    arr = np.asarray(value)
    return str(arr.item() if arr.shape == () else value)


def _frame_position(model, data, frame_name: str, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    pinocchio.forwardKinematics(model, data, q)
    pinocchio.framesForwardKinematics(model, data, q)
    return np.asarray(data.oMf[model.getFrameId(frame_name)].translation, dtype=float).reshape(3)


def _chain_points(model, data, q: np.ndarray, frame_name: str) -> np.ndarray:
    """World positions of every joint origin (in chain order) plus the controlled
    frame -- a polyline that approximates the arm for a serial manipulator."""
    q = np.asarray(q, dtype=float)
    pinocchio.forwardKinematics(model, data, q)
    pinocchio.framesForwardKinematics(model, data, q)
    pts = [np.asarray(data.oMi[i].translation, dtype=float).reshape(3)
           for i in range(model.njoints)]              # 0 = universe (origin)
    pts.append(np.asarray(data.oMf[model.getFrameId(frame_name)].translation,
                          dtype=float).reshape(3))      # controlled frame (payload)
    return np.asarray(pts, dtype=float)


def _make_viewer(model, urdf: str):
    """Return (viz, has_model). Tries several candidate mesh roots; on failure,
    returns a plain Meshcat viewer and has_model=False (skeleton mode)."""
    here = os.path.dirname(os.path.abspath(urdf))
    package_dirs = [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
    try:
        visual = pinocchio.buildGeomFromUrdf(
            model, urdf, pinocchio.GeometryType.VISUAL, package_dirs=package_dirs)
        collision = pinocchio.buildGeomFromUrdf(
            model, urdf, pinocchio.GeometryType.COLLISION, package_dirs=package_dirs)
        viz = MeshcatVisualizer(model, collision, visual)
        viz.initViewer(open=False)
        viz.loadViewerModel()
        print("Loaded URDF meshes. Open this in your browser:", viz.viewer.url())
        return viz, True
    except Exception as exc:
        viewer = meshcat.Visualizer()
        print("Could not load URDF meshes:", exc)
        print("Falling back to a forward-kinematics SKELETON of the arm "
              "(joint spheres + line). The motion is fully shown; only the solid "
              "link meshes are missing.")
        print("Open this in your browser:", viewer.url())

        class _SkeletonViz:
            def __init__(self, model, viewer):
                self.model = model
                self.viewer = viewer

            def display(self, _q):
                return None

        return _SkeletonViz(model, viewer), False


def _draw_table(viewer, name: str, bounds: np.ndarray, height: float, color: int) -> None:
    bounds = np.asarray(bounds, dtype=float).reshape(2, 2)
    (xmin, ymin), (xmax, ymax) = bounds
    sx, sy = float(xmax - xmin), float(ymax - ymin)
    cx, cy = float(0.5 * (xmin + xmax)), float(0.5 * (ymin + ymax))
    viewer[f"tables/{name}"].set_object(
        g.Box([sx, sy, 0.02]),
        g.MeshLambertMaterial(color=color, opacity=0.6, transparent=True),
    )
    viewer[f"tables/{name}"].set_transform(tf.translation_matrix([cx, cy, float(height) - 0.01]))


def _draw_sphere(viewer, name: str, position: np.ndarray, color: int, radius: float = 0.02) -> None:
    p = np.asarray(position, dtype=float).reshape(3).copy()
    p[2] += MARKER_Z_OFFSET
    viewer[f"markers/{name}"].set_object(
        g.Sphere(float(radius)),
        g.MeshLambertMaterial(color=color, opacity=1.0),
    )
    viewer[f"markers/{name}"].set_transform(tf.translation_matrix(p))


def _draw_path_line(viewer, name: str, xyz: np.ndarray,
                    color: int = 0x999999, linewidth: int = 1,
                    opacity: float = 0.35) -> None:
    pts = np.asarray(xyz, dtype=float).reshape((-1, 3)).T
    viewer[name].set_object(
        g.Line(
            g.PointsGeometry(pts),
            g.LineBasicMaterial(
                color=color,
                linewidth=linewidth,
                opacity=opacity,
                transparent=opacity < 1.0,
            ),
        )
    )


def _draw_skeleton(viewer, chain: np.ndarray) -> None:
    """(Re)draw the arm polyline. Called every frame in skeleton mode."""
    viewer["skeleton/bones"].set_object(
        g.Line(g.PointsGeometry(chain.T), g.LineBasicMaterial(color=0x222222, linewidth=4))
    )


def main() -> int:
    if not os.path.exists(PLAN_PATH):
        print(f"Missing plan file: {PLAN_PATH}")
        print(f"Run first: python -B {os.path.join(HERE, 'run_panda.py')}")
        return 1

    plan = np.load(PLAN_PATH, allow_pickle=False)
    dense = np.asarray(plan["dense"], dtype=float)
    lift_off = np.asarray(plan["lift_off"], dtype=float)
    set_down = np.asarray(plan["set_down"], dtype=float)
    table_a = np.asarray(plan["tableA"], dtype=float)
    table_b = np.asarray(plan["tableB"], dtype=float)
    h_a = float(np.asarray(plan["hA"]).item())
    h_b = float(np.asarray(plan["hB"]).item())
    frame = _as_str(plan["frame"])
    urdf = _as_str(plan["urdf"])

    model = pinocchio.buildModelFromUrdf(urdf)
    data = model.createData()
    viz, has_model = _make_viewer(model, urdf)
    viewer = viz.viewer

    _draw_table(viewer, "A", table_a, h_a, 0x55AA55)
    _draw_table(viewer, "B", table_b, h_b, 0x5588CC)

    if "ee_xyz" in plan.files:
        payload_xyz = np.asarray(plan["ee_xyz"], dtype=float)
    else:
        payload_xyz = np.asarray([_frame_position(model, data, frame, q) for q in dense], dtype=float)
    _draw_sphere(viewer, "start", payload_xyz[0], 0x00E5FF)
    _draw_sphere(viewer, "goal", payload_xyz[-1], 0xCC0000)
    _draw_sphere(viewer, "lift_off", _frame_position(model, data, frame, lift_off), 0xDDCC00)
    _draw_sphere(viewer, "set_down", _frame_position(model, data, frame, set_down), 0xCC33CC)
    _draw_sphere(viewer, "current_payload", payload_xyz[0], 0xFF9900, radius=0.018)
    _draw_path_line(viewer, "payload_path/reference", payload_xyz,
                    color=0x999999, linewidth=1, opacity=0.35)

    # In skeleton mode, create joint-marker spheres once; transforms update per frame.
    if not has_model:
        chain0 = _chain_points(model, data, dense[0], frame)
        for i in range(len(chain0)):
            viewer[f"skeleton/joint_{i}"].set_object(
                g.Sphere(0.026), g.MeshLambertMaterial(color=0x777777, opacity=1.0))
        _draw_skeleton(viewer, chain0)

    print("Animating forever. Press Ctrl+C to stop.")
    print("Run commands:")
    print(f"  python -B {os.path.join(HERE, 'run_panda.py')}")
    print(f"  python -B {os.path.join(HERE, 'animate_meshcat.py')}")
    while True:
        live_trail = []
        viewer["ee_trail"].delete()
        for q, p in zip(dense, payload_xyz):
            if has_model:
                viz.display(np.asarray(q, dtype=float))
            else:
                chain = _chain_points(model, data, q, frame)
                _draw_skeleton(viewer, chain)
                for i in range(len(chain)):
                    viewer[f"skeleton/joint_{i}"].set_transform(tf.translation_matrix(chain[i]))
            marker_p = np.asarray(p, dtype=float).reshape(3).copy()
            marker_p[2] += MARKER_Z_OFFSET
            viewer["markers/current_payload"].set_transform(tf.translation_matrix(marker_p))
            live_trail.append(np.asarray(p, dtype=float).reshape(3).copy())
            if len(live_trail) >= 2:
                _draw_path_line(viewer, "ee_trail", np.asarray(live_trail, dtype=float),
                                color=0xFF7700, linewidth=4, opacity=1.0)
            time.sleep(0.03)


if __name__ == "__main__":
    raise SystemExit(main())
