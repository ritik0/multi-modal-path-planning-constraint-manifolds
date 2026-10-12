from __future__ import annotations

import numpy as np

from ompl_planner import plan_in_mode
from regrasp_plan import TRANSFER_DELTA, TRANSFER_PLANNER, TRANSFER_SOLVE_TIME, TRANSFER_SPACE, _seed_list
from regrasp_transport import (
    REGRASP_TRANSPORT_YAW_DEG,
    TABLE_A,
    _box_on_surface,
    _feasible_grasps,
    _held_scene,
    _project_pose,
    _surface_mode,
)


def main() -> int:
    grasp = _feasible_grasps()[0]
    scene = _held_scene(grasp)
    surface = scene.surface("A")
    mode = _surface_mode(scene, surface, grasp.name)

    q0, ok0 = _project_pose(
        scene,
        mode,
        _box_on_surface(surface, TABLE_A["xy_bounds"][1][0] - 0.10, 0.42, REGRASP_TRANSPORT_YAW_DEG),
        _seed_list(scene),
        f"test_slide_A_{grasp.name}_from",
    )
    if not ok0:
        raise RuntimeError("failed to project slide_A start payload pose")

    q1, ok1 = _project_pose(
        scene,
        mode,
        _box_on_surface(surface, TABLE_A["xy_bounds"][0][0] + 0.10, 0.42, REGRASP_TRANSPORT_YAW_DEG),
        _seed_list(scene, [q0]),
        f"test_slide_A_{grasp.name}_to",
    )
    if not ok1:
        raise RuntimeError("failed to project slide_A goal payload pose")

    seg = plan_in_mode(
        scene.robot,
        mode,
        np.asarray(q0, dtype=float),
        np.asarray(q1, dtype=float),
        planner_name=TRANSFER_PLANNER,
        solve_time=TRANSFER_SOLVE_TIME,
        delta=TRANSFER_DELTA,
        space=TRANSFER_SPACE,
    )
    if not seg.success or seg.path is None:
        raise RuntimeError(f"slide_A plan failed: {seg.message}")

    path = np.asarray(seg.path, dtype=float)
    print(f"SLIDE SEGMENT OK path shape={path.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
