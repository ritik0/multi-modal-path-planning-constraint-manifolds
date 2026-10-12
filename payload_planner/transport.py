"""Clean joint-space multimodal transport planner for the payload demo."""

import os

from scene import Scene
from constraints import (
    Mode,
    FrameConstraint,
    PlaneConstraint,
    FlatCarryConstraint,
    TorqueFeasibility,
    ContactSupportFeasibility,
    JointLimit,
    WorkspaceBoundary,
    CollisionConstraint,
)
from ompl_planner import plan_in_mode
import numpy as np


# --------------------------------------------------------------------------- #
# OPT-IN transfer-height foliation for the free-flight segment.
#
# When PAYLOAD_FOLIATION=1, the single free-flight plan is replaced by a search
# over a family of horizontal transfer planes (leaves): for the given
# (q_lift, q_set) pair we pick the carry height whose cross-leaf path has the
# lowest torque utilisation (path length as a mild tiebreaker), then return that
# leaf's free-flight path.  When the flag is off, nothing here runs and the
# planner behaves exactly as before.
# --------------------------------------------------------------------------- #
def _foliation_enabled():
    return os.environ.get("PAYLOAD_FOLIATION", "0").lower() in {"1", "true", "yes", "on"}


def _plan_free_flight_foliated(scene, q_lift, q_set, planner, solve_time, delta,
                               constrained_space):
    """
    Plan the free-flight as the best leaf of a transfer-height foliation.
    Returns a PlanResult-like object (the cross-leaf segment) on success, or a
    failed-segment sentinel on failure.  Falls back to the plain free-flight if
    the foliation module is unavailable, so this can never make planning worse.
    """
    try:
        import transfer_foliation as fol
    except Exception:
        # module missing -> behave exactly like the plain planner
        return _plan_segment(scene.robot, free_flight_mode(scene),
                             q_lift, q_set, planner, solve_time, delta,
                             constrained_space)

    w_torque = float(os.environ.get("PAYLOAD_W_TORQUE", "10.0"))
    w_path = float(os.environ.get("PAYLOAD_W_PATH", "1.0"))
    best_h, info = fol.select_transfer_height(
        scene, q_lift, q_set,
        planner=planner, solve_time=solve_time, delta=delta,
        space=constrained_space, w_torque=w_torque, w_path=w_path,
        verbose=False,
    )
    if best_h is None or info is None:
        # no feasible leaf -> signal failure the same way a failed segment does
        return _plan_segment(scene.robot, free_flight_mode(scene),
                             q_lift, q_set, planner, solve_time, delta,
                             constrained_space)

    # Stitch the chosen leaf's three sub-paths (lift->leaf, cross, leaf->set)
    # into a single free-flight path.  Drop duplicated junction rows so the
    # concatenation is clean, and force the true endpoints so this segment
    # connects to slide_A (ends at q_lift) and slide_B (starts at q_set).
    segs = info.get("segments", [])
    paths = [np.asarray(p, float) for (_, p) in segs
             if p is not None and len(np.asarray(p, float)) > 0]
    if not paths:
        return _plan_segment(scene.robot, free_flight_mode(scene),
                             q_lift, q_set, planner, solve_time, delta,
                             constrained_space)
    stitched = [paths[0]]
    for nxt in paths[1:]:
        prev = stitched[-1]
        # if the next sub-path starts where the previous ended, skip the dup row
        if np.allclose(prev[-1], nxt[0], atol=1e-6):
            stitched.append(nxt[1:])
        else:
            stitched.append(nxt)
    full = np.vstack([s for s in stitched if len(s) > 0])
    # hard-guarantee the endpoints match the lift-off / set-down configs
    full[0] = np.asarray(q_lift, float).reshape(-1)
    full[-1] = np.asarray(q_set, float).reshape(-1)

    class _FoliatedSeg:
        success = True
        path = full
        message = f"foliated free-flight at h={best_h:.3f} (util_peak={info.get('util_peak',0):.3f})"
        planner = "RRTConnect"
    seg = _FoliatedSeg()
    return seg


R_FLAT = np.diag([1.0, -1.0, -1.0])


class Placement:
    def __init__(self, surface, x, y):
        self.surface = str(surface)
        self.x = float(x)
        self.y = float(y)


class TransportResult:
    def __init__(
        self,
        success,
        message,
        segments=(),
        lift_off=None,
        set_down=None,
        q_start=None,
        q_goal=None,
        n_lift=0,
        n_set=0,
        pairs_tried=0,
        segment_results=(),
        dense_path=None,
    ):
        self.success = bool(success)
        self.message = str(message)
        self.segments = [(str(label), np.asarray(path, dtype=float)) for label, path in segments]
        self.lift_off = None if lift_off is None else np.asarray(lift_off, dtype=float)
        self.set_down = None if set_down is None else np.asarray(set_down, dtype=float)
        self.q_start = None if q_start is None else np.asarray(q_start, dtype=float)
        self.q_goal = None if q_goal is None else np.asarray(q_goal, dtype=float)
        self.n_lift = int(n_lift)
        self.n_set = int(n_set)
        self.pairs_tried = int(pairs_tried)
        self.segment_results = list(segment_results)
        self.dense_path = None if dense_path is None else np.asarray(dense_path, dtype=float)

    def full_path(self):
        if not self.segments:
            return None
        return np.vstack([path for _, path in self.segments])


def coll(scene):
    return CollisionConstraint(
        scene.robot,
        scene.collision_frames,
        scene.obstacles,
        frame_points=scene.collision_frame_points,
        self_collision_frames=scene.self_collision_frames,
        self_collision_min_distance=scene.self_collision_min_distance,
    )


def support_torque(scene):
    # Single construction point: reads the support_* knobs (contact_model, mu,
    # base_radius, mu_torsion, support_derate, ...) straight from scene.config.
    return ContactSupportFeasibility.from_scene(scene)


def on_surface_mode(scene, surface):
    r = scene.robot
    frame = scene.controlled_frame
    return Mode(
        f"on_{surface.name}",
        equalities=[
            PlaneConstraint(r, frame, surface.height),
            FlatCarryConstraint(r, frame, R_FLAT),
        ],
        inequalities=[
            JointLimit(r),
            WorkspaceBoundary(
                r,
                frame,
                [surface.xy_bounds[0][0], surface.xy_bounds[0][1], -1.0e3],
                [surface.xy_bounds[1][0], surface.xy_bounds[1][1], 1.0e3],
            ),
            coll(scene),
            support_torque(scene),
        ],
    )


def free_flight_mode(scene):
    r = scene.robot
    frame = scene.controlled_frame
    return Mode(
        "free_flight",
        equalities=[FlatCarryConstraint(r, frame, R_FLAT)],
        inequalities=[
            JointLimit(r),
            TorqueFeasibility(r, scene.tau_max, derate=scene.derate),
            coll(scene),
        ],
    )


def _inset_xy_bounds(surface, support_margin):
    (xlo, ylo), (xhi, yhi) = np.asarray(surface.xy_bounds, dtype=float)
    m = float(support_margin)
    return float(xlo + m), float(ylo + m), float(xhi - m), float(yhi - m)


def liftoff_mode(scene, surface, support_margin=0.0):
    r = scene.robot
    frame = scene.controlled_frame
    xlo, ylo, xhi, yhi = _inset_xy_bounds(surface, support_margin)
    return Mode(
        f"lift_{surface.name}",
        equalities=[
            PlaneConstraint(r, frame, surface.height),
            FlatCarryConstraint(r, frame, R_FLAT),
        ],
        inequalities=[
            JointLimit(r),
            WorkspaceBoundary(
                r,
                frame,
                [xlo, ylo, -1.0e3],
                [xhi, yhi, 1.0e3],
            ),
            coll(scene),
            TorqueFeasibility(r, scene.tau_max, derate=scene.derate),
        ],
    )


def build_modes(scene, support_margin=0.0):
    modes = {
        "free": free_flight_mode(scene),
        "torque": TorqueFeasibility(scene.robot, scene.tau_max, derate=scene.derate),
        "support_torque": support_torque(scene),
    }
    for surface in scene.surfaces:
        modes[surface.name] = on_surface_mode(scene, surface)
        modes[f"lift_{surface.name}"] = liftoff_mode(scene, surface, support_margin=support_margin)
    return modes


def wrap_to_limits(q, qlo, qhi):
    return np.minimum(np.maximum(np.asarray(q, dtype=float), qlo), qhi)


def _within_limits(q, qlo, qhi):
    q = np.asarray(q, dtype=float)
    return bool(np.all(q >= qlo - 1.0e-9) and np.all(q <= qhi + 1.0e-9))


def place(scene, surface, x, y, seed=None, rng=None, n_restarts=64,
          lock_yaw=False, target_rotation=None):
    r = scene.robot
    frame = scene.controlled_frame
    qlo, qhi = r.joint_limits
    qlo = np.asarray(qlo, dtype=float)
    qhi = np.asarray(qhi, dtype=float)
    rng = rng or np.random.default_rng(20260612)
    R_target = R_FLAT if target_rotation is None else np.asarray(target_rotation, dtype=float).reshape(3, 3)
    # roll & pitch always locked (gripper straight down); yaw locked only if asked
    rot_mask = (1, 1, 1) if lock_yaw else (1, 1, 0)
    mode = Mode(
        "place",
        equalities=[
            FrameConstraint(
                r,
                frame,
                target_position=[float(x), float(y), float(surface.height)],
                pos_mask=(1, 1, 1),
                target_rotation=R_target,
                rot_mask=rot_mask,
            )
        ],
    )
    seeds = []
    if seed is not None:
        seeds.append(np.asarray(seed, dtype=float))
    seeds.append(0.5 * (qlo + qhi))
    seeds.extend(rng.uniform(qlo, qhi) for _ in range(max(0, int(n_restarts))))
    for q0 in seeds:
        q, ok = mode.project(q0, qlo=qlo, qhi=qhi)
        if ok and _within_limits(q, qlo, qhi):
            return np.asarray(q, dtype=float), True
    return None, False


def _place_feasible(scene, surface, x, y, mode, rng, n_tries=96,
                    lock_yaw=False, target_rotation=None):
    qlo, qhi = scene.robot.joint_limits
    qlo = np.asarray(qlo, dtype=float)
    qhi = np.asarray(qhi, dtype=float)
    seeds = [0.5 * (qlo + qhi)]
    seeds.extend(rng.uniform(qlo, qhi) for _ in range(max(0, int(n_tries))))
    for seed in seeds:
        q, ok = place(scene, surface, x, y, seed=seed, rng=rng, n_restarts=0,
                      lock_yaw=lock_yaw, target_rotation=target_rotation)
        if ok and q is not None and mode.feasible(q):
            return q, True
    return None, False


def sample_transitions(scene, surface, n_tries, other_surface=None, rng=None, support_margin=0.0):
    rng = rng or np.random.default_rng(20260612)
    mode = liftoff_mode(scene, surface, support_margin=support_margin)
    xlo, ylo, xhi, yhi = _inset_xy_bounds(surface, support_margin)
    if xlo > xhi or ylo > yhi:
        return []
    found = []
    for _ in range(max(0, int(n_tries))):
        x = float(rng.uniform(xlo, xhi))
        if other_surface is not None and rng.random() < 0.6:
            other_y = float(np.mean(np.asarray(other_surface.xy_bounds, dtype=float)[:, 1]))
            edge_y = ylo if abs(ylo - other_y) < abs(yhi - other_y) else yhi
            span = float(abs(yhi - ylo))
            y = float(np.clip(edge_y + rng.normal(0.0, 0.18 * span), ylo, yhi))
        else:
            y = float(rng.uniform(ylo, yhi))
        q, ok = _place_feasible(scene, surface, x, y, mode, rng, n_tries=16)
        if ok:
            found.append(q)
    return found


def _joint_length(path):
    arr = np.asarray(path, dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def _total_joint_length(paths):
    return float(sum(_joint_length(path) for path in paths))


def _linear_densify_joint_path(path, max_step=0.01):
    path = np.asarray(path, dtype=float)
    if len(path) <= 1:
        return path.copy()
    dense = [path[0].copy()]
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        dist = float(np.linalg.norm(b - a))
        if dist < 1.0e-6:
            continue
        n = max(1, int(np.ceil(dist / float(max_step))))
        for k in range(1, n + 1):
            t = float(k) / float(n)
            dense.append((1.0 - t) * a + t * b)
    return np.asarray(dense, dtype=float)


def _drop_close_waypoints(path, tol=1.0e-3):
    q = np.asarray(path, dtype=float)
    if len(q) <= 2:
        return q.copy()
    kept = [q[0].copy()]
    for idx in range(1, len(q) - 1):
        if float(np.linalg.norm(q[idx] - kept[-1])) >= float(tol):
            kept.append(q[idx].copy())
    kept.append(q[-1].copy())
    return np.asarray(kept, dtype=float)


def best_utilisation(scene, x, y, z, n_restarts=40, seed=None, rng=None):
    surface = type("_Surface", (), {"height": float(z), "name": "diagnostic"})()
    torque = TorqueFeasibility(scene.robot, scene.tau_max, derate=scene.derate)
    qlo, qhi = scene.robot.joint_limits
    qlo = np.asarray(qlo, dtype=float)
    qhi = np.asarray(qhi, dtype=float)
    rng = rng or np.random.default_rng(20260612)
    seeds = []
    if seed is not None:
        seeds.append(np.asarray(seed, dtype=float))
    seeds.append(0.5 * (qlo + qhi))
    seeds.extend(rng.uniform(qlo, qhi) for _ in range(max(0, int(n_restarts))))
    best_util = None
    best_q = None
    for q0 in seeds:
        q, ok = place(scene, surface, x, y, seed=q0, rng=rng, n_restarts=0)
        if not ok or q is None:
            continue
        util = float(np.max(torque.utilisation(q)))
        if best_util is None or util < best_util:
            best_util = util
            best_q = q
    return best_util, best_q


def uniform_resample_result(scene, result, n_points=300, support_margin=0.0, yaw_band=0.0):
    dense_path = getattr(result, "dense_path", None)
    if dense_path is not None:
        return np.asarray(dense_path, dtype=float).copy()
    path = result.full_path()
    if path is None:
        return np.zeros((0, scene.robot.nq), dtype=float)
    path = np.asarray(path, dtype=float)
    if len(path) <= 1 or int(n_points) <= 1:
        return path.copy()
    diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = np.concatenate([[True], diffs > 1.0e-12])
    path = path[keep]
    if len(path) <= 1:
        return np.repeat(path, int(n_points), axis=0)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    if arc[-1] <= 1.0e-12:
        return np.repeat(path[:1], int(n_points), axis=0)
    targets = np.linspace(0.0, arc[-1], int(n_points))
    out = np.empty((len(targets), path.shape[1]), dtype=float)
    for j in range(path.shape[1]):
        out[:, j] = np.interp(targets, arc, path[:, j])
    return out


def _plan_segment(robot, mode, q0, q1, planner, solve_time, delta, constrained_space):
    return plan_in_mode(
        robot,
        mode,
        q0,
        q1,
        planner_name=planner,
        solve_time=solve_time,
        delta=delta,
        space=constrained_space,
        simplify=True,
    )


def plan(
    scene,
    start,
    goal,
    planner="RRTConnect",
    solve_time=6.0,
    delta=0.15,
    n_tries=300,
    best_of_n=8,
    budget=80,
    support_margin=None,
    constrained_space="projected",
):
    r = scene.robot
    constrained_space = str(constrained_space).strip().lower()
    if constrained_space not in {"projected", "atlas"}:
        return TransportResult(
            False,
            f"unknown constrained space {constrained_space!r}; use 'projected' or 'atlas'",
        )
    if support_margin is None:
        support_margin = float(os.environ.get("PAYLOAD_SUPPORT_MARGIN", "0.05"))
    rng = np.random.default_rng(int(os.environ.get("PAYLOAD_SEED", "20260612")))
    surface_a = scene.surface(start.surface)
    surface_b = scene.surface(goal.surface)
    mode_a = on_surface_mode(scene, surface_a)
    mode_b = on_surface_mode(scene, surface_b)
    mode_f = free_flight_mode(scene)
    torque = TorqueFeasibility(r, scene.tau_max, derate=scene.derate)
    support = support_torque(scene)

    # Start orientation: roll & pitch are always locked (gripper straight down,
    # cylinder vertical).  Optionally also pin the YAW so the start hand facing
    # matches the physical setup; yaw stays free during the motion so the planner
    # keeps wrist freedom.  Falls back to the yaw-free start if the pinned pose is
    # not reachable, so this can never make planning worse.
    start_lock_yaw = bool(scene.config.get("start_lock_yaw", False))
    start_yaw_deg = float(scene.config.get("start_yaw_deg", 0.0))
    R_start = R_FLAT
    if start_yaw_deg != 0.0:
        _a = np.radians(start_yaw_deg)
        _Rz = np.array([[np.cos(_a), -np.sin(_a), 0.0],
                        [np.sin(_a), np.cos(_a), 0.0],
                        [0.0, 0.0, 1.0]])
        R_start = _Rz @ R_FLAT
    q_start, ok_start = (None, False)
    if start_lock_yaw:
        q_start, ok_start = _place_feasible(
            scene, surface_a, start.x, start.y, mode_a, rng,
            lock_yaw=True, target_rotation=R_start)
        if ok_start and q_start is not None:
            print(f"start orientation: full flat-carry pinned "
                  f"(gripper straight down, yaw={start_yaw_deg:.1f} deg)")
        else:
            print("start orientation: yaw-locked start not reachable here; "
                  "using yaw-free start (gripper still straight down)")
    if not ok_start or q_start is None:
        q_start, ok_start = _place_feasible(scene, surface_a, start.x, start.y, mode_a, rng)
    if not ok_start or q_start is None:
        return TransportResult(False, "start placement is not feasible on its support surface")
    goal_lock_yaw = bool(scene.config.get("goal_lock_yaw", False))
    q_goal, ok_goal = (None, False)
    if goal_lock_yaw:
        q_goal, ok_goal = _place_feasible(
            scene, surface_b, goal.x, goal.y, mode_b, rng,
            lock_yaw=True, target_rotation=R_start)
        if ok_goal and q_goal is not None:
            print(f"goal orientation: pinned to start facing "
                  f"(gripper straight down, yaw={start_yaw_deg:.1f} deg)")
        else:
            print("goal orientation: yaw-locked goal not reachable here; "
                  "using yaw-free goal (gripper still straight down)")
    if not ok_goal or q_goal is None:
        q_goal, ok_goal = _place_feasible(scene, surface_b, goal.x, goal.y, mode_b, rng)
    if not ok_goal or q_goal is None:
        return TransportResult(False, "goal placement is not feasible on its support surface")

    lifts = sample_transitions(
        scene,
        surface_a,
        n_tries,
        other_surface=surface_b,
        rng=rng,
        support_margin=support_margin,
    )
    sets = sample_transitions(
        scene,
        surface_b,
        n_tries,
        other_surface=surface_a,
        rng=rng,
        support_margin=support_margin,
    )
    if not lifts or not sets:
        return TransportResult(
            False,
            f"no torque-feasible transitions: lift-off={len(lifts)}, set-down={len(sets)}",
            q_start=q_start,
            q_goal=q_goal,
            n_lift=len(lifts),
            n_set=len(sets),
        )

    pairs = []
    for i, q_lift in enumerate(lifts):
        for j, q_set in enumerate(sets):
            pairs.append((float(np.linalg.norm(q_lift - q_set)), i, j))
    pairs.sort(key=lambda item: item[0])

    accepted = []
    pairs_tried = 0
    for _, i, j in pairs[: max(0, int(budget))]:
        pairs_tried += 1
        q_lift = lifts[i]
        q_set = sets[j]
        seg_a = _plan_segment(
            r, mode_a, q_start, q_lift, planner, solve_time, delta, constrained_space
        )
        if not seg_a.success:
            continue
        if _foliation_enabled():
            seg_f = _plan_free_flight_foliated(
                scene, q_lift, q_set, planner, solve_time, delta, constrained_space
            )
        else:
            seg_f = _plan_segment(
                r, mode_f, q_lift, q_set, planner, solve_time, delta, constrained_space
            )
        if not seg_f.success:
            continue
        seg_b = _plan_segment(
            r, mode_b, q_set, q_goal, planner, solve_time, delta, constrained_space
        )
        if not seg_b.success:
            continue
        segments = [
            ("slide_A", _drop_close_waypoints(seg_a.path)),
            ("free_flight", _drop_close_waypoints(seg_f.path)),
            ("slide_B", _drop_close_waypoints(seg_b.path)),
        ]
        cost = _total_joint_length([path for _, path in segments])
        accepted.append((cost, segments, q_lift.copy(), q_set.copy(), [seg_a, seg_f, seg_b]))
        if len(accepted) >= max(1, int(best_of_n)):
            break

    if not accepted:
        return TransportResult(
            False,
            f"no feasible transport triple among {pairs_tried} candidate pairs",
            q_start=q_start,
            q_goal=q_goal,
            n_lift=len(lifts),
            n_set=len(sets),
            pairs_tried=pairs_tried,
        )

    cost, segments, q_lift, q_set, seg_results = min(accepted, key=lambda item: item[0])
    raw_full = np.vstack([path for _, path in segments])
    dense_path = _linear_densify_joint_path(raw_full, max_step=0.01)
    lift_util = float(np.max(torque.utilisation(q_lift)))
    set_util = float(np.max(torque.utilisation(q_set)))
    lengths = [_joint_length(path) for _, path in segments]
    print(
        "transport selected: "
        f"lift_util={lift_util:.3f}, set_util={set_util:.3f}, "
        f"joint_lens slide_A/free/slide_B={lengths[0]:.3f}/{lengths[1]:.3f}/{lengths[2]:.3f}, "
        f"total={cost:.3f}"
    )
    for name, q_cfg in [
        ("start", q_start),
        ("lift_off", q_lift),
        ("set_down", q_set),
        ("goal", q_goal),
    ]:
        sol = support.solve(q_cfg)
        print(
            f"  support {name:8s}: "
            f"lambda={sol['lambda']:.6g}, "
            f"max_util={float(np.max(sol['utilisation'])):.3f}, "
            f"feasible={sol['feasible']}, reason={sol['reason']}"
        )
    return TransportResult(
        True,
        f"transport planned; selected best of {len(accepted)} feasible triples",
        segments=segments,
        lift_off=q_lift,
        set_down=q_set,
        q_start=q_start,
        q_goal=q_goal,
        n_lift=len(lifts),
        n_set=len(sets),
        pairs_tried=pairs_tried,
        segment_results=list(zip(["slide_A", "free_flight", "slide_B"], seg_results)),
        dense_path=dense_path,
    )