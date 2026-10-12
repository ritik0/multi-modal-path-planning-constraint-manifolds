"""Plan a Franka payload transfer across two separated finite tables.

The core planner lives in scene.py, constraints.py, transport.py, and
ompl_planner.py.  This script only instantiates a separated-table scene, runs
the transport planner, prints diagnostics, and saves the resulting
path for PyVista playback.
"""

from __future__ import annotations

import os

import numpy as np

from scene import Scene
from constraints import FrameConstraint, Mode
from payload_model import (
    ACTIVE_PAYLOAD_CASE,
    COMPONENTS,
    TOTAL_MASS,
    COM,
    INERTIA,
    normalise_case,
    payload_points,
    SUPPORT_CONTACT_LOCAL as MODEL_SUPPORT_CONTACT_LOCAL,
    SUPPORT_NORMAL_WORLD as MODEL_SUPPORT_NORMAL_WORLD,
    ensure_payload_urdf_for_case,
)
import transport

best_utilisation = transport.best_utilisation

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# Editable scene constants
# --------------------------------------------------------------------------- #
PAYLOAD_CASE = normalise_case(os.environ.get("PAYLOAD_CASE", ACTIVE_PAYLOAD_CASE))
BASE_URDF = os.path.join(HERE, "fer_franka_hand_payload.urdf")
URDF = ensure_payload_urdf_for_case(BASE_URDF, HERE, case=PAYLOAD_CASE)
FRAME = "payload"
COLLISION_FRAMES = [
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
    "payload",
]
SUPPORT_MARGIN = 0.1
TABLE_COLLISION_MARGIN = float(os.environ.get("PAYLOAD_TABLE_COLLISION_MARGIN", "0.00"))
PAYLOAD_COLLISION_HALF_EXTENTS = (0.06, 0.06, 0.09)
OUTER_HALF_ONLY = True 
PAYLOAD_PLANNER_NAME = os.environ.get("PAYLOAD_PLANNER_NAME", "RRTConnect")
PAYLOAD_SOLVE_TIME = float(os.environ.get("PAYLOAD_SOLVE_TIME", "10"))
PAYLOAD_DELTA = float(os.environ.get("PAYLOAD_DELTA", "0.1"))
PAYLOAD_BEST_OF_N = int(os.environ.get("PAYLOAD_BEST_OF_N", "20"))
PAYLOAD_CONSTRAINED_SPACE = os.environ.get(
    "PAYLOAD_CONSTRAINED_SPACE", "projected"
).strip().lower()
if PAYLOAD_CONSTRAINED_SPACE not in {"projected", "atlas"}:
    raise ValueError(
        "PAYLOAD_CONSTRAINED_SPACE must be 'projected' or 'atlas', "
        f"got {PAYLOAD_CONSTRAINED_SPACE!r}"
    )
LOCK_SLIDE_YAW = os.environ.get("PAYLOAD_LOCK_SLIDE_YAW", "1").lower() not in {
    "0", "false", "no", "off"
}
SLIDE_YAW_BAND = float(os.environ.get("PAYLOAD_SLIDE_YAW_BAND", "0.0"))
PAYLOAD_TORQUE_DERATE = float(os.environ.get("PAYLOAD_TORQUE_DERATE", "0.55"))
PAYLOAD_CONTACT_LOCAL = MODEL_SUPPORT_CONTACT_LOCAL.tolist()
SUPPORT_NORMAL_WORLD = MODEL_SUPPORT_NORMAL_WORLD.tolist()
SUPPORT_FORCE_MAX = float(os.environ.get("PAYLOAD_SUPPORT_FORCE_MAX", "inf"))
# contact model: "normal" (vertical force, fast analytic, default) |
#                "force"  (normal + friction) | "wrench" (full 6-D: + tip/twist)
# NOTE: "force"/"wrench" use an SLSQP cone solve per call -- fine for analysis,
# slow inside planning; keep "normal" for the run, use analyze_contact_wrench.py
# for the rich force+moment picture.
SUPPORT_CONTACT_MODEL = os.environ.get("PAYLOAD_SUPPORT_CONTACT_MODEL", "normal")
SUPPORT_MU = float(os.environ.get("PAYLOAD_SUPPORT_MU", "0.0"))
SUPPORT_MU_TORSION = float(os.environ.get("PAYLOAD_SUPPORT_MU_TORSION", "0.0"))
SUPPORT_BASE_RADIUS = float(os.environ.get("PAYLOAD_SUPPORT_BASE_RADIUS", "0.0"))
SUPPORT_DERATE = float(os.environ.get("PAYLOAD_SUPPORT_DERATE", "1.0"))

# Two separated finite support surfaces.  By default these are two tables at
# z=0 separated by a gap around y=0.  Setting the PAYLOAD_SURF_* environment
# variables turns them into other layouts -- e.g. a two-level shelf, where the
# two surfaces sit at different heights and the object is slid off the lower
# level, carried up through free flight, and placed on the upper level.
def _surf_bounds(env_name, default):
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default
    vals = [float(x) for x in raw.replace(",", " ").split()]
    # expect xlo ylo xhi yhi
    return [[vals[0], vals[1]], [vals[2], vals[3]]]

TABLE_A = {
    "name": "A",
    "height": float(os.environ.get("PAYLOAD_SURF_A_Z", "0.0")),
    "xy_bounds": _surf_bounds("PAYLOAD_SURF_A_XY", [[0.09, 0.2], [0.38, 0.58]]),
}
TABLE_B = {
    "name": "B",
    "height": float(os.environ.get("PAYLOAD_SURF_B_Z", "0.0")),
    "xy_bounds": _surf_bounds("PAYLOAD_SURF_B_XY", [[0.09, -0.72], [0.38, -0.2]]),
}

SELF_COLLISION_FRAMES = [
    "fer_link1",
    "fer_link2",
    "fer_link3",
    "fer_link4",
    "fer_link5",
    "fer_link6",
    "fer_link7",
    "fer_link8",
    "fer_hand",
    "payload",
]
SELF_COLLISION_MIN_DISTANCE = 0.04


def _payload_points():
    return payload_points(COMPONENTS)


# Table A: x 0.05–0.40 (forward),  y 0.20–0.60 (sideways).  Table B = mirror (y negative).
# A pose only forces a SLIDE in the far-outer corner: y >= 0.55 AND x >= 0.35.
# Anything inboard of that is may-lift -> the robot just lifts it, no slide.
START_X = None   #  0.05 .. 0.40   | for a slide keep  0.35 .. 0.40 | 0.35 .. 0.40
START_Y = 0.75   #  0.20 .. 0.60   | for a slide use   0.55 .. 0.58 | 0.50 .. 0.58 
# Start EE orientation.  Roll & pitch are always locked (gripper straight
# down, cylinder vertical).  START_LOCK_YAW also pins the YAW so the start
# hand facing is deterministic / matches the physical setup photo;
# START_YAW_DEG rotates that facing about the vertical.  Yaw stays free
# during the motion regardless.  Falls back to yaw-free if unreachable.
START_LOCK_YAW = os.environ.get("PAYLOAD_START_LOCK_YAW", "1") == "1"
START_YAW_DEG = float(os.environ.get("PAYLOAD_START_YAW_DEG", "0.0"))
GOAL_X  = None   #  0.05 .. 0.40   | for a slide keep  0.35 .. 0.40 | 0.35 .. 0.40
GOAL_Y  = -0.7   # -0.60 .. -0.20  | for a slide use  -0.58 .. -0.55| -0.58 .. -0.50
# Goal EE orientation.  OFF by default: the goal yaw stays free, which gives the
# planner wrist freedom to meet torque limits with the unbalanced load.  Set
# PAYLOAD_GOAL_LOCK_YAW=1 to set the object down facing exactly as it started
# (goal yaw pinned to the START orientation).  Yaw stays free in between, and the
# planner falls back to a yaw-free goal if the pinned pose is unreachable, so this
# can never make planning strictly worse.
GOAL_LOCK_YAW = os.environ.get("PAYLOAD_GOAL_LOCK_YAW", "1") == "1"

WALL_ENABLED = os.environ.get("PAYLOAD_WALL", "0") == "1"
WALL_TOP = float(os.environ.get("PAYLOAD_WALL_TOP", "0.20"))
OBSTACLE_MARGIN = float(os.environ.get("PAYLOAD_OBSTACLE_MARGIN", "0.05"))
_wx_lo, _wx_hi, _wy_thick = 0.10, 0.60, 0.18
_wall_T = np.eye(4)
_wall_T[:3, 3] = [0.5 * (_wx_lo + _wx_hi), 0.0, 0.5 * WALL_TOP]
GAP_WALL = {
    "type": "box",
    "name": "gap_wall",
    "T": _wall_T,
    "xsize": (_wx_hi - _wx_lo) + 2.0 * OBSTACLE_MARGIN,
    "ysize": _wy_thick + 2.0 * OBSTACLE_MARGIN,
    "zsize": WALL_TOP + 2.0 * OBSTACLE_MARGIN,
}

CFG = {
    "urdf": URDF,
    "controlled_frame": FRAME,
    "payload_case": PAYLOAD_CASE,
    "payload_mass": TOTAL_MASS,
    "payload_com": COM,
    "payload_inertia": INERTIA,
    "torque_derate": PAYLOAD_TORQUE_DERATE,
    "collision_frames": COLLISION_FRAMES,
    "collision_frame_points": {
        "payload": _payload_points(),
    },
    "table_collision_margin": TABLE_COLLISION_MARGIN,
    "start_lock_yaw": START_LOCK_YAW,
    "start_yaw_deg": START_YAW_DEG,
    "goal_lock_yaw": GOAL_LOCK_YAW,
    "support_contact_local": PAYLOAD_CONTACT_LOCAL,
    "support_normal_world": SUPPORT_NORMAL_WORLD,
    "support_force_max": SUPPORT_FORCE_MAX,
    "support_contact_model": SUPPORT_CONTACT_MODEL,
    "support_mu": SUPPORT_MU,
    "support_mu_torsion": SUPPORT_MU_TORSION,
    "support_base_radius": SUPPORT_BASE_RADIUS,
    "support_derate": SUPPORT_DERATE,
    "self_collision_frames": SELF_COLLISION_FRAMES,
    "self_collision_min_distance": SELF_COLLISION_MIN_DISTANCE,
    "obstacles": [GAP_WALL] if WALL_ENABLED else [],
    "surfaces": [TABLE_A, TABLE_B],
}

PLAN_OUT = os.path.join(HERE, "panda_plan.npz")


class _Tee:
    """Mirror verbose planner output while retaining lines for failure summaries."""

    def __init__(self, stream):
        self.stream = stream
        self.lines: list[str] = []

    def write(self, text: str) -> int:
        self.stream.write(text)
        for line in text.splitlines():
            if line.strip():
                self.lines.append(line)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()


def _failure_lines(lines: list[str]) -> list[str]:
    """Extract per-pair failure messages emitted by planner verbose output."""
    failures = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and "failed:" in stripped:
            failures.append(stripped)
    return failures


def _project_flat(scene: Scene, x: float, y: float, z: float, seed: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """Project a flat controlled-frame placement for diagnostics only."""
    mode = Mode(
        "diagnostic_flat_place",
        equalities=[
            FrameConstraint(
                scene.robot,
                FRAME,
                target_position=[float(x), float(y), float(z)],
                pos_mask=(1, 1, 1),
                target_rotation=transport.R_FLAT,
                rot_mask=(1, 1, 0),
            )
        ],
    )
    q, ok = mode.project(seed)
    return (np.asarray(q, dtype=float), True) if ok else (None, False)


def _project_flat_any_seed(scene: Scene, x: float, y: float, z: float,
                           seeds: list[np.ndarray]) -> tuple[np.ndarray | None, bool]:
    for seed in seeds:
        q, ok = _project_flat(scene, x, y, z, seed)
        if ok and q is not None:
            return q, True
    return None, False


def _diagnostic_sweep(scene: Scene, seeds: list[np.ndarray] | None = None) -> None:
    """Print reach/torque checks across both tables and the gap.

    This is a cheap sanity check that the finite support patches straddle the
    free-lift boundary.  It does not plan a path; it only projects flat
    placements and evaluates the existing torque oracle.
    """
    modes = transport.build_modes(scene)
    torque = modes["torque"]
    support = modes["support_torque"]
    qlo, qhi = scene.robot.joint_limits
    seed = 0.5 * (np.asarray(qlo, dtype=float) + np.asarray(qhi, dtype=float))
    seeds = seeds or [seed]
    print("=" * 70)
    print("diagnostic flat placement sweep")
    print(" region   x[m]    y[m]    reach   util_free   util_support    lambda    verdict")
    samples: list[tuple[str, float, float, float]] = []
    support_effect_count = 0
    support_worse_count = 0
    for table in (TABLE_A, TABLE_B):
        (xlo, ylo), (xhi, yhi) = np.asarray(table["xy_bounds"], dtype=float)
        for x in np.linspace(xlo, xhi, 3):
            for y in np.linspace(ylo, yhi, 3):
                samples.append((str(table["name"]), float(x), float(y), float(table["height"])))
    xlo = max(float(TABLE_A["xy_bounds"][0][0]), float(TABLE_B["xy_bounds"][0][0]))
    xhi = min(float(TABLE_A["xy_bounds"][1][0]), float(TABLE_B["xy_bounds"][1][0]))
    for x in np.linspace(xlo, xhi, 5):
        samples.append(("gap", float(x), 0.0, 0.5 * (float(TABLE_A["height"]) + float(TABLE_B["height"]))))

    for region, x, y, z in samples:
        q, ok = _project_flat(scene, x, y, z, seed)
        if not ok or q is None:
            print(
                f" {region:>5s}  {x:+.3f}  {y:+.3f}   no      --         --          --      unreachable"
            )
            continue
        seed = q
        util_free = float(np.max(torque.utilisation(q)))
        support_sol = support.solve(q)
        util_support = float(np.max(support_sol["utilisation"]))
        lambda_value = float(support_sol["lambda"])
        if support_sol["feasible"] and np.isfinite(lambda_value) and lambda_value > 1.0e-9:
            support_effect_count += 1
            if util_support > util_free + 1.0e-9:
                support_worse_count += 1
        if util_free <= 1.0:
            verdict = "MAY-LIFT"
        elif support_sol["feasible"]:
            verdict = "SUPPORTED-SLIDE"
        else:
            verdict = "INFEASIBLE"
        print(
            f" {region:>5s}  {x:+.3f}  {y:+.3f}   yes    "
            f"{util_free:9.3f}  {util_support:11.3f}  {lambda_value:8.3f}   {verdict}"
        )
    if support_effect_count > 0 and support_worse_count == support_effect_count:
        print(
            "WARNING: support load-sharing increased max joint utilisation at every "
            "sample where lambda>0. The contact-force sign may be wrong."
        )

    table = TABLE_A
    (_, ylo), (_, yhi) = np.asarray(table["xy_bounds"], dtype=float)
    y_mid = 0.5 * (float(ylo) + float(yhi))
    z = float(table["height"])
    first_slide_x = None
    largest_reachable_x = None
    print()
    print("table-A outward torque boundary scan (legacy carried-seed posture)")
    print(f" y={y_mid:+.3f} m, z={z:+.3f} m, torque_derate={CFG['torque_derate']:.2f}")
    print(" x[m]    reach   util_max   verdict")
    for x in np.linspace(0.30, 0.70, 17):
        q, ok = _project_flat_any_seed(scene, float(x), y_mid, z, seeds)
        if not ok or q is None:
            print(f" {x:+.3f}   no      --       unreachable")
            continue
        util = float(np.max(torque.utilisation(q)))
        verdict = "MAY-LIFT" if util <= 1.0 else "SLIDE"
        largest_reachable_x = float(x)
        if first_slide_x is None and util > 1.0:
            first_slide_x = float(x)
        print(f" {x:+.3f}   yes    {util:7.3f}   {verdict}")
    if first_slide_x is None:
        print(" legacy must-slide boundary: none in scanned reachable range")
    else:
        print(f" legacy must-slide boundary: smallest SLIDE x = {first_slide_x:.3f} m")
    if largest_reachable_x is None:
        print(" legacy largest reachable x: none in scan")
    else:
        print(f" legacy largest reachable x: {largest_reachable_x:.3f} m")

    best_first_slide_x = None
    best_largest_reachable_x = None
    best_rng = np.random.default_rng(424242)
    carry_seed = np.asarray(seeds[0], dtype=float)
    print()
    print("table-A outward torque boundary scan (best over posture, 40 restarts)")
    print(f" y={y_mid:+.3f} m, z={z:+.3f} m, torque_derate={CFG['torque_derate']:.2f}")
    print(" x[m]    reach   best_util  verdict")
    for x in np.linspace(0.30, 0.70, 17):
        util, q_best = best_utilisation(
            scene,
            float(x),
            y_mid,
            z,
            n_restarts=40,
            seed=carry_seed,
            rng=best_rng,
        )
        if util is None or q_best is None:
            print(f" {x:+.3f}   no      --       unreachable")
            continue
        carry_seed = q_best
        verdict = "MAY-LIFT" if util <= 1.0 else "SLIDE"
        best_largest_reachable_x = float(x)
        if best_first_slide_x is None and util > 1.0:
            best_first_slide_x = float(x)
        print(f" {x:+.3f}   yes    {util:7.3f}   {verdict}")
    if best_first_slide_x is None:
        print(" TRUE must-slide boundary: none in scanned reachable range")
    else:
        print(f" TRUE must-slide boundary: smallest SLIDE x = {best_first_slide_x:.3f} m")
    if best_largest_reachable_x is None:
        print(" TRUE largest reachable x: none in scan")
    else:
        print(f" TRUE largest reachable x: {best_largest_reachable_x:.3f} m")
    if first_slide_x is None or best_first_slide_x is None:
        print(" comparison to legacy boundary: one scan found no SLIDE boundary")
    else:
        delta = best_first_slide_x - first_slide_x
        print(
            " comparison to legacy boundary: "
            f"legacy={first_slide_x:.3f} m, true-best={best_first_slide_x:.3f} m, "
            f"delta={delta:+.3f} m"
        )

    print()
    print("gap free-carry height/pull-in torque scan")
    print(" y=+0.000 m, flat carry, torque-feasible means util_max <= 1")
    print(" x[m]    z[m]    reach   util_max   verdict")
    gap_xs = [0.32, 0.42, 0.52]
    gap_zs = np.linspace(0.40, 0.85, 10)
    table_height_feasible = []
    first_feasible = None
    for x in gap_xs:
        carry_seed = np.asarray(seeds[0], dtype=float)
        for z in gap_zs:
            q, ok = _project_flat(scene, float(x), 0.0, float(z), carry_seed)
            if not ok or q is None:
                q, ok = _project_flat_any_seed(scene, float(x), 0.0, float(z), seeds)
            if not ok or q is None:
                print(f" {x:+.3f}  {z:+.3f}   no      --       unreachable")
                if np.isclose(z, 0.40):
                    table_height_feasible.append(False)
                continue
            carry_seed = q
            util = float(np.max(torque.utilisation(q)))
            feasible = util <= 1.0
            verdict = "feasible" if feasible else "infeasible"
            if np.isclose(z, 0.40):
                table_height_feasible.append(feasible)
            if feasible and (first_feasible is None or z < first_feasible[0]):
                first_feasible = (float(z), float(x), util)
            print(f" {x:+.3f}  {z:+.3f}   yes    {util:7.3f}   {verdict}")
    if table_height_feasible:
        all_table_height_infeasible = not any(table_height_feasible)
        print(
            " gap crossing at table height z=0.400 infeasible for all sampled x: "
            f"{all_table_height_infeasible}"
        )
    else:
        print(" gap crossing at table height z=0.400 infeasible for all sampled x: True")
    if first_feasible is None:
        print(" up-and-in corridor: no torque-feasible carry found in scanned heights")
    else:
        z, x, util = first_feasible
        print(
            " up-and-in corridor begins at "
            f"z={z:.3f} m, x={x:.3f} m (utilisation={util:.3f})"
        )


def _seed_candidates(scene: Scene, count: int = 8) -> list[np.ndarray]:
    """Deterministic small seed set: joint midpoint plus reproducible randoms."""
    qlo, qhi = scene.robot.joint_limits
    qlo = np.asarray(qlo, dtype=float)
    qhi = np.asarray(qhi, dtype=float)
    rng = np.random.default_rng(20260608)
    seeds = [0.5 * (qlo + qhi)]
    seeds.extend(rng.uniform(qlo, qhi) for _ in range(max(0, int(count) - 1)))
    return [np.asarray(q, dtype=float) for q in seeds]


def _joint_path_length(path: np.ndarray) -> float:
    arr = np.asarray(path, dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def _cartesian_path_length(scene: Scene, path: np.ndarray) -> float:
    arr = np.asarray(path, dtype=float)
    if len(arr) < 2:
        return 0.0
    xyz = np.asarray(
        [scene.robot.frame_pose(FRAME, q)[:3, 3] for q in arr],
        dtype=float,
    )
    return float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))


def _collision_constraint_from_mode(mode: Mode):
    for constraint in mode.inequalities:
        if getattr(constraint, "name", "") == "collision_free":
            return constraint
    return None


def _project_flat_search(
    scene: Scene,
    x: float,
    y: float,
    z: float,
    seeds: list[np.ndarray],
    *,
    mode: Mode | None = None,
    require_feasible: bool = False,
) -> tuple[np.ndarray | None, bool]:
    qlo, qhi = scene.robot.joint_limits
    rng = np.random.default_rng(20260611)
    candidates = [np.asarray(seed, dtype=float) for seed in seeds]
    candidates.extend(rng.uniform(qlo, qhi) for _ in range(80))
    for seed in candidates:
        q, ok = _project_flat(scene, x, y, z, seed)
        if not ok or q is None:
            continue
        q = transport.wrap_to_limits(q, qlo, qhi)
        if not (np.all(q >= qlo - 1.0e-9) and np.all(q <= qhi + 1.0e-9)):
            continue
        if require_feasible and mode is not None and not mode.feasible(q):
            continue
        return q, True
    return None, False


def _print_collision_diagnostics(scene: Scene, free_mode: Mode, collision) -> bool:
    print("=" * 70)
    print("collision wiring diagnostics")
    print(
        " mode.feasible call path: "
        "constraints.Mode.feasible -> constraints.Mode.inequality_values -> "
        "constraints.CollisionConstraint.value -> "
        "constraints.CollisionConstraint.hit_info -> collision.py::*Collision.is_in_collision"
    )
    print(
        " collision constraint class: "
        f"{collision.__class__.__module__}.{collision.__class__.__name__}"
    )
    object_names = []
    for obj in getattr(collision, "objects", []):
        spec = getattr(obj, "obstacle", {})
        name = str(spec.get("name", "<unnamed>"))
        object_names.append(name)
        size = ""
        if obj.__class__.__name__ == "BoxCollision":
            size = (
                f" size=({float(spec.get('xsize', float('nan'))):.3f},"
                f"{float(spec.get('ysize', float('nan'))):.3f},"
                f"{float(spec.get('zsize', float('nan'))):.3f})"
            )
        print(
            " collision body: "
            f"name={name}, module={obj.__class__.__module__}, "
            f"class={obj.__class__.__name__}{size}"
        )
    expected = set()
    missing = sorted(expected - set(object_names))
    print(
        " expected environment bodies present: "
        f"{'YES' if not missing else 'NO, missing ' + ', '.join(missing)}"
    )
    point_counts = {
        frame: len(points)
        for frame, points in getattr(collision, "frame_points", {}).items()
    }
    print(
        " environment query frames: "
        f"{', '.join(getattr(collision, 'frames', []))}"
    )
    print(f" per-frame local sample points: {point_counts}")
    print(
        " self-collision frames: "
        f"{', '.join(getattr(collision, 'self_collision_frames', [])) or '<none>'}; "
        f"min_distance={getattr(collision, 'self_collision_min_distance', 0.0):.3f}"
    )
    print(
        " free_flight inequalities: "
        f"{', '.join(getattr(c, 'name', c.__class__.__name__) for c in free_mode.inequalities)}"
    )
    return not missing


def _collision_self_check(scene: Scene, seeds: list[np.ndarray]) -> None:
    modes = transport.build_modes(scene, support_margin=SUPPORT_MARGIN)
    free_mode = modes["free"]
    collision = _collision_constraint_from_mode(free_mode)
    if collision is None:
        print("LOUD FAILURE: free_flight has no CollisionConstraint in its inequalities.")
        raise SystemExit(1)
    bodies_present = _print_collision_diagnostics(scene, free_mode, collision)
    if not bodies_present:
        print("LOUD FAILURE: expected environment bodies are absent from free_flight collision query.")
        raise SystemExit(1)
    if TABLE_COLLISION_MARGIN <= 0.0:
        print("COLLISION SELF-CHECK: skipped (tables at base plane, no solid blocks)")
        return
    # Probe the center of the synthesized table-A collision box.  This keeps
    # the wiring check meaningful after removing the explicit gap wall.
    table = TABLE_A
    (xlo, ylo), (xhi, yhi) = np.asarray(table["xy_bounds"], dtype=float)
    bad_center = np.asarray(
        [
            0.5 * (float(xlo) + float(xhi)),
            0.5 * (float(ylo) + float(yhi)),
            0.5 * max(0.0, float(table["height"]) - TABLE_COLLISION_MARGIN),
        ],
        dtype=float,
    )
    q_bad, bad_projected = _project_flat_search(
        scene,
        float(bad_center[0]),
        float(bad_center[1]),
        float(bad_center[2]),
        seeds,
    )
    if not bad_projected or q_bad is None:
        print(
            "LOUD FAILURE: could not project diagnostic table-solid "
            "flat-carry config."
        )
        raise SystemExit(1)

    bad_hit = collision.hit_info(q_bad)
    bad_feasible = bool(free_mode.feasible(q_bad))
    print(
        "COLLISION SELF-CHECK: "
        f"table-solid config feasible={bad_feasible} (should be False), "
        f"raw_hit={bad_hit is not None}, hit={bad_hit}"
    )
    if bad_feasible or bad_hit is None:
        print("LOUD FAILURE: collision is still not wired correctly; refusing to plan.")
        raise SystemExit(1)
    print("COLLISION SELF-CHECK: PASS")


def _discover_start(scene: Scene, seeds: list[np.ndarray]) -> tuple[transport.Placement | None, float | None]:
    """Pick the outermost reachable table-A pose that is genuinely must-slide."""
    torque = transport.build_modes(scene)["torque"]
    table = TABLE_A
    (xlo, ylo), (xhi, yhi) = np.asarray(table["xy_bounds"], dtype=float)
    if START_X is not None and START_Y is not None:
        for seed in seeds:
            q, ok = _project_flat(scene, float(START_X), float(START_Y), float(table["height"]), seed)
            if ok and q is not None:
                util = float(np.max(torque.utilisation(q)))
                verdict = "SLIDE" if util > 1.0 else "MAY-LIFT"
                print(f"PINNED START: x={START_X:.3f}, y={START_Y:.3f}, utilisation={util:.3f} -> {verdict}")
                return transport.Placement(str(table["name"]), float(START_X), float(START_Y)), util
        print(f"LOUD: pinned START ({START_X}, {START_Y}) is not reachable / flat-projectable.")
        return None, None
    y_nominal = 0.5 * (float(ylo) + float(yhi)) if START_Y is None else float(START_Y)
    y = float(np.clip(y_nominal, ylo + SUPPORT_MARGIN, yhi - SUPPORT_MARGIN))
    # Outermost along the table reach direction: scan far edge inward.
    x_inner = float(xlo + SUPPORT_MARGIN)
    if OUTER_HALF_ONLY:
        x_inner = max(x_inner, 0.5 * (float(xlo) + float(xhi)))
    xs = np.linspace(float(xhi - SUPPORT_MARGIN), x_inner, 41)
    best = None
    for x in xs:
        for seed in seeds:
            q, ok = _project_flat(scene, float(x), y, float(table["height"]), seed)
            if not ok or q is None:
                continue
            util = float(np.max(torque.utilisation(q)))
            if util > 1.0:
                best = (transport.Placement(str(table["name"]), float(x), y), util)
                break
        if best is not None:
            break
    if best is None:
        print(
            "No reachable must-slide START exists on table A at "
            f"torque_derate={CFG['torque_derate']:.2f}, payload mass from {URDF}."
        )
        print("The scenario has no must-slide region under the honest torque setting.")
        return None, None
    placement, util = best
    print(
        "discovered START: "
        f"surface={placement.surface}, x={placement.x:.3f}, y={placement.y:.3f}, "
        f"utilisation={util:.3f} -> SLIDE"
    )
    return placement, util


def _best_flat_utilisation(scene: Scene, placement: transport.Placement,
                           seeds: list[np.ndarray]) -> tuple[float | None, int | None]:
    torque = transport.build_modes(scene)["torque"]
    best_util = None
    best_seed = None
    for seed_idx, seed in enumerate(seeds):
        q, ok = _project_flat(
            scene,
            placement.x,
            placement.y,
            scene.surface(placement.surface).height,
            seed,
        )
        if ok and q is not None:
            util = float(np.max(torque.utilisation(q)))
            if best_util is None or util < best_util:
                best_util = util
                best_seed = seed_idx
    return best_util, best_seed


def _discover_goal(scene: Scene, seeds: list[np.ndarray]) -> tuple[transport.Placement | None, float | None, int | None]:
    """Outermost reachable table-B placement in the FAR half; SLIDE goals allowed."""
    torque = transport.build_modes(scene)["torque"]
    table = TABLE_B
    (xlo, ylo), (xhi, yhi) = np.asarray(table["xy_bounds"], dtype=float)

    if GOAL_X is not None and GOAL_Y is not None:
        for seed_idx, seed in enumerate(seeds):
            q, ok = _project_flat(scene, float(GOAL_X), float(GOAL_Y), float(table["height"]), seed)
            if ok and q is not None:
                util = float(np.max(torque.utilisation(q)))
                verdict = "SLIDE" if util > 1.0 else "MAY-LIFT"
                print(f"PINNED GOAL: x={GOAL_X:.3f}, y={GOAL_Y:.3f}, utilisation={util:.3f} -> {verdict}")
                return transport.Placement(str(table["name"]), float(GOAL_X), float(GOAL_Y)), util, seed_idx
        print(f"LOUD: pinned GOAL ({GOAL_X}, {GOAL_Y}) is not reachable / flat-projectable.")
        return None, None, None

    x_min = float(xlo + SUPPORT_MARGIN)
    x_max = float(xhi - SUPPORT_MARGIN)
    if OUTER_HALF_ONLY:
        x_min = max(x_min, 0.5 * (float(xlo) + float(xhi)))
    if x_min > x_max:
        x_min = x_max = 0.5 * (float(xlo) + float(xhi))
    y_min = float(ylo + SUPPORT_MARGIN)
    y_max = float(yhi - SUPPORT_MARGIN)
    if y_min > y_max:
        y_min = y_max = 0.5 * (float(ylo) + float(yhi))

    goal_y = 0.5 * (float(ylo) + float(yhi)) if GOAL_Y is None else float(GOAL_Y)
    y_nominal = float(np.clip(goal_y, y_min, y_max))

    # Prefer the requested y first, then sweep y; at each y scan from the far
    # edge inward and take the FIRST reachable placement (outermost wins).
    for y in [y_nominal, *np.linspace(y_min, y_max, 7)]:
        for x in np.linspace(x_max, x_min, 21):
            for seed_idx, seed in enumerate(seeds):
                q, ok = _project_flat(scene, float(x), float(y), float(table["height"]), seed)
                if not ok or q is None:
                    continue
                util = float(np.max(torque.utilisation(q)))
                verdict = "MAY-LIFT" if util <= 1.0 else "SLIDE"
                placement = transport.Placement(str(table["name"]), float(x), float(y))
                print(
                    "discovered GOAL: "
                    f"surface={placement.surface}, x={placement.x:.3f}, y={placement.y:.3f}, "
                    f"utilisation={util:.3f} -> {verdict}"
                )
                return placement, util, seed_idx
    print("No reachable GOAL placement exists in the far half of table B.")
    return None, None, None
    


def _print_summary(scene: Scene, res) -> None:
    """Print segment counts and transition payload positions/margins."""
    torque = transport.build_modes(scene)["torque"]
    robot = scene.robot
    print()
    print("result:", res.message)
    print(
        "transitions sampled: "
        f"lift-off={res.n_lift}, set-down={res.n_set}, pairs tried={res.pairs_tried}"
    )
    if not res.success:
        return
    result_by_label = {
        label: seg for label, seg in getattr(res, "segment_results", [])
    }
    for label, path in res.segments:
        seg = result_by_label.get(label)
        planner = getattr(seg, "planner", "unknown") if seg is not None else "unknown"
        space = getattr(seg, "constrained_space", "unknown") if seg is not None else "unknown"
        print(
            f"  segment {label:14s}: {len(path):3d} pts, "
            f"planner={planner}, space={space}, "
            f"joint_len={_joint_path_length(path):.3f}, "
            f"cart_len={_cartesian_path_length(scene, path):.3f}"
        )
    if getattr(res, "segment_results", None):
        spaces = [
            f"{label}={getattr(seg, 'constrained_space', 'unknown')}"
            for label, seg in res.segment_results
        ]
        print("constrained spaces actually used:", ", ".join(spaces))
    full = res.full_path()
    print(f"full path: {0 if full is None else len(full)} configs")
    if res.lift_off is not None:
        pos = robot.frame_pose(FRAME, res.lift_off)[:3, 3]
        print(f"lift-off object at {pos}  torque margin={torque.margin(res.lift_off):+.3f}")
    if res.set_down is not None:
        pos = robot.frame_pose(FRAME, res.set_down)[:3, 3]
        print(f"set-down object at {pos}  torque margin={torque.margin(res.set_down):+.3f}")
    if res.q_start is not None and res.q_goal is not None and res.lift_off is not None and res.set_down is not None:
        p_start = robot.frame_pose(FRAME, res.q_start)[:3, 3]
        p_lift = robot.frame_pose(FRAME, res.lift_off)[:3, 3]
        p_set = robot.frame_pose(FRAME, res.set_down)[:3, 3]
        p_goal = robot.frame_pose(FRAME, res.q_goal)[:3, 3]
        d0 = float(np.linalg.norm(p_start - p_goal))
        d1 = float(np.linalg.norm(p_lift - p_goal))
        d2 = float(np.linalg.norm(p_set - p_goal))
        moved_away = d1 > d0
        print(f"distance-to-goal d0/d1/d2: {d0:.3f}/{d1:.3f}/{d2:.3f}")
        print(f"slide moved away from goal: {moved_away}")
    for label, path in res.segments:
        if label != "free_flight" or len(path) == 0:
            continue
        object_xyz = np.asarray(
            [robot.frame_pose(FRAME, q)[:3, 3] for q in path],
            dtype=float,
        )
        min_z = float(np.min(object_xyz[:, 2]))
        apex_z = float(np.max(object_xyz[:, 2]))
        table_top = float(max(surface.height for surface in scene.surfaces))
        print(
            "flight object-bottom z min/apex: "
            f"{min_z:.3f}/{apex_z:.3f} m, "
            f"table top max z={table_top:.3f} m"
        )
        if min_z < table_top - TABLE_COLLISION_MARGIN:
            print(
                "WARNING: free-flight path dips below table collision top; "
                "solid table collision should reject this path."
            )
        margins = np.asarray([torque.margin(q) for q in path], dtype=float)
        i_min = int(np.argmin(margins))
        print(f"min flight margin: {margins[i_min]:+.3f}")
        print(f"min flight margin config: {np.asarray(path[i_min], dtype=float)}")
        break


def main() -> int:
    np.set_printoptions(precision=3, suppress=True)
    scene = Scene(CFG)
    print("=" * 70)
    print("Separated-table Panda payload plan")
    print(f"payload case: {PAYLOAD_CASE}")
    print(f"payload mass: {TOTAL_MASS:.3f} kg")
    print(f"payload COM: {np.round(COM, 5)}")
    print(f"payload URDF generated: {URDF}")
    print(f"URDF : {URDF}")
    print(f"frame: {FRAME}")
    print(f"table A bounds: {TABLE_A['xy_bounds']} at z={TABLE_A['height']:.3f}")
    print(f"table B bounds: {TABLE_B['xy_bounds']} at z={TABLE_B['height']:.3f}")
    print(f"table collision margin: {TABLE_COLLISION_MARGIN:.3f} m")
    print(f"obstacle keep-out margin: {OBSTACLE_MARGIN:.3f} m")
    print(
        "support model: "
        f"model={SUPPORT_CONTACT_MODEL}, contact_local={PAYLOAD_CONTACT_LOCAL}, "
        f"normal_world={SUPPORT_NORMAL_WORLD}, mu={SUPPORT_MU}, "
        f"base_radius={SUPPORT_BASE_RADIUS}, mu_torsion={SUPPORT_MU_TORSION}, "
        f"support_derate={SUPPORT_DERATE}, support_force_max={SUPPORT_FORCE_MAX}"
    )
    if WALL_ENABLED:
        print(
            "gap wall enabled: "
            f"top={WALL_TOP:.3f} m, x_span=[{_wx_lo:.3f}, {_wx_hi:.3f}], "
            f"y_thickness={_wy_thick:.3f} m"
        )
    print(f"support margin: {SUPPORT_MARGIN:.3f} m")
    print(
        f"planner: {PAYLOAD_PLANNER_NAME}, space={PAYLOAD_CONSTRAINED_SPACE}, "
        f"solve_time={PAYLOAD_SOLVE_TIME:.1f}s, "
        f"lock_slide_yaw={LOCK_SLIDE_YAW}, slide_yaw_band={SLIDE_YAW_BAND:.3g}"
    )
    seeds = _seed_candidates(scene)
    _collision_self_check(scene, seeds)
    start, start_util = _discover_start(scene, seeds)
    if start is None:
        print("=" * 70)
        print("Run commands:")
        print(f"  python -B {os.path.join(HERE, 'run_panda.py')}")
        print(f"  python -B {os.path.join(HERE, 'animate_pyvista.py')}")
        return 1
    print(f"start: ({start.surface!r}, {start.x:.3f}, {start.y:.3f})")
    goal, goal_util, goal_seed = _discover_goal(scene, seeds)
    if goal is None:
        print("=" * 70)
        print("Run commands:")
        print(f"  python -B {os.path.join(HERE, 'run_panda.py')}")
        print(f"  python -B {os.path.join(HERE, 'animate_pyvista.py')}")
        return 1
    print(f"goal : ({goal.surface!r}, {goal.x:.3f}, {goal.y:.3f})")
    if goal_seed is not None and goal_seed != 0:
        seeds = [seeds[goal_seed], *seeds[:goal_seed], *seeds[goal_seed + 1:]]
        print(f"planning seeds reordered to try holdable GOAL seed {goal_seed} first")
    _diagnostic_sweep(scene, seeds)

    res = transport.plan(
        scene,
        start,
        goal,
        planner=PAYLOAD_PLANNER_NAME,
        solve_time=PAYLOAD_SOLVE_TIME,
        delta=PAYLOAD_DELTA,
        best_of_n=PAYLOAD_BEST_OF_N,
        budget=int(os.environ.get("PAYLOAD_BUDGET", "80")),
        constrained_space=PAYLOAD_CONSTRAINED_SPACE,
    )

    print("=" * 70)
    _print_summary(scene, res)
    if res.success:
        full = np.asarray(res.full_path(), dtype=float)
        dense = transport.uniform_resample_result(
            scene,
            res,
            n_points=300,
            support_margin=SUPPORT_MARGIN,
            yaw_band=SLIDE_YAW_BAND,
        )
        ee_xyz = np.asarray(
            [scene.robot.frame_pose(FRAME, q)[:3, 3] for q in dense],
            dtype=float,
        )
        np.savez(
            PLAN_OUT,
            path=full,
            dense=dense,
            ee_xyz=ee_xyz,
            payload_case=np.asarray(PAYLOAD_CASE),
            payload_mass=np.asarray(TOTAL_MASS, dtype=float),
            payload_com=np.asarray(COM, dtype=float),
            payload_inertia=np.asarray(INERTIA, dtype=float),
            payload_component_names=np.asarray([c[0] for c in COMPONENTS]),
            payload_component_sizes=np.asarray([c[2] for c in COMPONENTS], dtype=float),
            payload_component_centers=np.asarray([c[3] for c in COMPONENTS], dtype=float),
            lift_off=np.asarray(res.lift_off, dtype=float),
            set_down=np.asarray(res.set_down, dtype=float),
            seg_lengths=np.asarray([len(p) for _, p in res.segments], dtype=int),
            segments=np.array([p for _, p in res.segments], dtype=object),
            seg_labels=np.array([lbl for lbl, _ in res.segments]),
            tableA=np.asarray(TABLE_A["xy_bounds"], dtype=float),
            hA=np.asarray(float(TABLE_A["height"])),
            tableB=np.asarray(TABLE_B["xy_bounds"], dtype=float),
            hB=np.asarray(float(TABLE_B["height"])),
            table_collision_margin=np.asarray(TABLE_COLLISION_MARGIN, dtype=float),
            support_contact_local=np.asarray(PAYLOAD_CONTACT_LOCAL, dtype=float),
            support_normal_world=np.asarray(SUPPORT_NORMAL_WORLD, dtype=float),
            support_force_max=np.asarray(SUPPORT_FORCE_MAX, dtype=float),
            support_contact_model=np.asarray(SUPPORT_CONTACT_MODEL),
            support_mu=np.asarray(SUPPORT_MU, dtype=float),
            support_mu_torsion=np.asarray(SUPPORT_MU_TORSION, dtype=float),
            support_base_radius=np.asarray(SUPPORT_BASE_RADIUS, dtype=float),
            support_derate=np.asarray(SUPPORT_DERATE, dtype=float),
            gap_wall_enabled=np.asarray(bool(WALL_ENABLED)),
            obstacle_center=np.asarray(_wall_T[:3, 3], dtype=float),
            obstacle_half_extents=np.asarray(
                [(_wx_hi - _wx_lo) / 2.0, _wy_thick / 2.0, WALL_TOP / 2.0],
                dtype=float,
            ),
            frame=np.asarray(FRAME),
            urdf=np.asarray(URDF),
        )
        print(f"saved plan -> {PLAN_OUT}")
    else:
        print("no feasible transport plan was produced.")
    print("=" * 70)
    print("Run commands:")
    print(f"  python -B {os.path.join(HERE, 'run_panda.py')}")
    print(f"  python -B {os.path.join(HERE, 'animate_pyvista.py')}")
    return 0 if res.success else 1


if __name__ == "__main__":
    raise SystemExit(main())