"""
multimodal.py
=============
Chain the multi-surface payload plan from a start placement to a goal placement,
using constraint Modes + sample_transition + a per-mode planner (OMPL RRT-Connect
by default).  No segment shape is prescribed; the planner discovers each one.

    placement_start (on surface A)
        --slide on A-->  lift-off transition  (still on A AND torque-feasible)
        --free flight->  set-down transition  (torque-feasible AND on B)
        --slide on B-->  placement_goal

Modes
-----
on_<surface> : PlaneConstraint(surface.height) + FlatCarry.   ineq = [collision]
               While the object is supported the TABLE carries the load, so the
               torque test is NOT active here -- this is what lets the start sit
               in a torque-infeasible spot and simply slide out of it.
free_flight  : FlatCarry.   ineq = [torque, collision, (workspace)]
               No surface under the object; the arm must hold it alone, so the
               torque inequality is active EVERYWHERE along the arc.

Transitions (found by projection onto the stacked manifolds, never hand-placed)
    lift-off = sample_transition(on_A, free_flight)
    set-down = sample_transition(free_flight, on_B)

A thin discrete layer tries (lift-off, set-down) pairs until all three segments
plan.  Many transitions exist; a failed flight arc just means "try another pair".

The planner is INJECTED (default = ompl_planner.plan_in_mode) so the same
orchestration can run with RRT-Connect in production or with the bundled
`straightline_planner` for an OMPL-free smoke test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from constraints import (PlaneConstraint, FlatCarryConstraint, FrameConstraint,
                         TorqueFeasibility, WorkspaceBoundary, CollisionConstraint,
                         JointLimit, Mode, sample_transition)
from ompl_planner import plan_in_mode, PlanResult

# tool z pointing down -> a flat grasp; matches verify.py / visualize.py
R_FLAT = np.diag([1., -1., -1.])
COLLISION_CHECK_EE_STEP = 0.01


def wrap_to_limits(q, qlo, qhi):
    """
    Shift each revolute angle by multiples of 2*pi to land inside [qlo, qhi]
    when possible (kinematically identical pose, g(q) unchanged).  Projection
    can converge to an equivalent angle outside the modelled range; this brings
    it back so OMPL's bounds accept it.
    """
    q = np.array(q, float)
    two_pi = 2.0 * np.pi
    for i in range(len(q)):
        if q[i] < qlo[i]:
            q[i] += np.ceil((qlo[i] - q[i]) / two_pi) * two_pi
        elif q[i] > qhi[i]:
            q[i] -= np.ceil((q[i] - qhi[i]) / two_pi) * two_pi
    return q


def _in_limits(q, qlo, qhi):
    return bool(np.all(q >= qlo - 1e-9) and np.all(q <= qhi + 1e-9))


def best_utilisation(scene, x: float, y: float, z: float, n_restarts: int = 40,
                     seed=None, rng=None):
    """Approximate the best torque posture for a flat, yaw-free placement."""
    torque = build_modes(scene)["torque"]
    qlo, qhi = scene.robot.joint_limits
    qlo = np.asarray(qlo, dtype=float)
    qhi = np.asarray(qhi, dtype=float)
    rng = rng or np.random.default_rng(20260608)
    projection_seeds = []
    if seed is not None:
        projection_seeds.append(np.asarray(seed, dtype=float))
    projection_seeds.extend(rng.uniform(qlo, qhi) for _ in range(int(n_restarts)))

    mode = Mode("diagnostic_flat_place", equalities=[FrameConstraint(
        scene.robot,
        scene.controlled_frame,
        target_position=[float(x), float(y), float(z)],
        pos_mask=(1, 1, 1),
        target_rotation=R_FLAT,
        rot_mask=(1, 1, 0),
    )])

    best_util = None
    best_q = None
    for q_seed in projection_seeds:
        q, ok = mode.project(q_seed)
        if not ok or q is None:
            continue
        q = wrap_to_limits(q, qlo, qhi)
        if not _in_limits(q, qlo, qhi):
            continue
        util = float(np.max(torque.utilisation(q)))
        if best_util is None or util < best_util:
            best_util = util
            best_q = q
    return best_util, best_q


@dataclass
class Placement:
    """A query: put the controlled frame here, flat, on this surface."""
    surface: str
    x: float
    y: float
    R: np.ndarray = field(default_factory=lambda: R_FLAT.copy())


@dataclass
class MultiModalResult:
    success: bool
    message: str
    segments: list = field(default_factory=list)   # [(label, path ndarray)]
    segment_results: list = field(default_factory=list)  # [(label, PlanResult)]
    lift_off: "np.ndarray | None" = None
    set_down: "np.ndarray | None" = None
    q_start: "np.ndarray | None" = None
    q_goal: "np.ndarray | None" = None
    n_lift: int = 0
    n_set: int = 0
    pairs_tried: int = 0
    yaw_locks: dict = field(default_factory=dict)
    slide_yaw_band: float = 0.0

    def full_path(self):
        """Concatenate segments, dropping the duplicated boundary configs."""
        if not self.segments:
            return None
        chunks = [self.segments[0][1]]
        for _, p in self.segments[1:]:
            chunks.append(p[1:])               # first row == previous last row
        return np.vstack(chunks)


def uniform_resample_result(scene, result, n_points=300, flat_tol=0.0,
                            workspace=None, support_margin=0.0,
                            yaw_band=0.0):
    """Uniformly resample a multimodal result while re-projecting per segment."""
    if result is None or not getattr(result, "segments", None):
        return np.empty((0, scene.robot.nq), dtype=float)
    yaw_locks = getattr(result, "yaw_locks", {}) or None
    effective_yaw_band = (
        getattr(result, "slide_yaw_band", yaw_band)
        if yaw_locks else float(yaw_band)
    )
    modes = build_modes(scene, flat_tol=flat_tol, workspace=workspace,
                        support_margin=support_margin, yaw_locks=yaw_locks,
                        yaw_band=effective_yaw_band)

    def mode_for_label(label):
        if label == "free_flight":
            return modes["free"]
        if label.startswith("slide_"):
            return modes["on"][label[len("slide_"):]]
        raise KeyError(f"unknown segment label {label!r}")

    edges = []
    total = 0.0
    first = None
    last = None
    for label, path in result.segments:
        mode = mode_for_label(label)
        arr = np.asarray(path, dtype=float)
        if len(arr) == 0:
            continue
        if first is None:
            first = arr[0].copy()
        last = arr[-1].copy()
        for a, b in zip(arr[:-1], arr[1:]):
            length = float(np.linalg.norm(b - a))
            if length <= 1.0e-12:
                continue
            edges.append((total, total + length, a.copy(), b.copy(), mode))
            total += length

    n_points = max(2, int(n_points))
    if first is None:
        return np.empty((0, scene.robot.nq), dtype=float)
    if total <= 1.0e-12 or not edges:
        return np.repeat(first.reshape(1, -1), n_points, axis=0)

    out = []
    edge_idx = 0
    for target in np.linspace(0.0, total, n_points):
        while edge_idx < len(edges) - 1 and target > edges[edge_idx][1]:
            edge_idx += 1
        s0, s1, a, b, mode = edges[edge_idx]
        alpha = 0.0 if s1 <= s0 else (target - s0) / (s1 - s0)
        if alpha <= 1.0e-12:
            q = a.copy()
        elif alpha >= 1.0 - 1.0e-12:
            q = b.copy()
        else:
            q = _project_valid(mode, (1.0 - alpha) * a + alpha * b)
            if q is None:
                q = a.copy() if alpha < 0.5 else b.copy()
        out.append(q)
    dense = np.asarray(out, dtype=float)
    dense[0] = first
    dense[-1] = last
    return dense


def _project_valid(mode, q, tol=1.0e-4):
    q_proj, ok = mode.project(q)
    if not ok or q_proj is None:
        return None
    q_proj = np.asarray(q_proj, dtype=float)
    if not mode.on_manifold(q_proj, tol=tol):
        return None
    if not mode.feasible(q_proj):
        return None
    return q_proj


def _cartesian_ee_positions(robot, frame, path):
    arr = np.asarray(path, dtype=float)
    return np.asarray(
        [np.asarray(robot.frame_pose(frame, q), dtype=float)[:3, 3] for q in arr],
        dtype=float,
    )


def _cartesian_polyline_length(positions):
    pos = np.asarray(positions, dtype=float)
    if len(pos) <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))


def _cartesian_ee_length(robot, frame, path):
    return _cartesian_polyline_length(_cartesian_ee_positions(robot, frame, path))


def _ee_distance(robot, frame, a, b):
    pa = np.asarray(robot.frame_pose(frame, a), dtype=float)[:3, 3]
    pb = np.asarray(robot.frame_pose(frame, b), dtype=float)[:3, 3]
    return float(np.linalg.norm(pb - pa))


def _local_substeps_for_edge(robot, frame, a, b, substeps=10,
                             ee_step=COLLISION_CHECK_EE_STEP):
    distance = _ee_distance(robot, frame, a, b)
    return max(int(substeps), int(np.ceil(distance / max(float(ee_step), 1.0e-9))), 1)


def _shortcut_segment_on_manifold(path, mode, rng, robot, frame, iterations=None,
                                  substeps=10, tol=1.0e-4):
    arr = np.asarray(path, dtype=float).copy()
    if len(arr) <= 2:
        return arr
    first = arr[0].copy()
    last = arr[-1].copy()
    ee = _cartesian_ee_positions(robot, frame, arr)
    iterations = (max(150, 2 * len(arr)) if iterations is None
                  else max(0, int(iterations)))
    substeps = max(2, int(substeps))
    for _ in range(iterations):
        if len(arr) <= 2:
            break
        i = int(rng.integers(0, len(arr) - 2))
        j = int(rng.integers(i + 2, len(arr)))
        a = arr[i]
        b = arr[j]
        replaced_len = _cartesian_polyline_length(ee[i:j + 1])
        if replaced_len <= 1.0e-12:
            continue
        candidate = [a.copy()]
        candidate_ee = [ee[i].copy()]
        ok = True
        local_substeps = _local_substeps_for_edge(robot, frame, a, b, substeps=substeps)
        for k in range(1, local_substeps):
            alpha = float(k) / float(local_substeps)
            q = _project_valid(mode, (1.0 - alpha) * a + alpha * b, tol=tol)
            if q is None:
                ok = False
                break
            candidate.append(q)
            candidate_ee.append(
                np.asarray(robot.frame_pose(frame, q), dtype=float)[:3, 3]
            )
        if not ok:
            continue
        candidate.append(b.copy())
        candidate_ee.append(ee[j].copy())
        candidate_ee = np.asarray(candidate_ee, dtype=float)
        candidate_len = _cartesian_polyline_length(candidate_ee)
        if not candidate_len <= 0.99 * replaced_len:
            continue
        arr = np.vstack([arr[:i], np.asarray(candidate), arr[j + 1:]])
        ee = np.vstack([ee[:i], candidate_ee, ee[j + 1:]])
    arr[0] = first
    arr[-1] = last
    return arr


def _densify_segment_on_manifold(path, mode, robot=None, frame=None, substeps=10, tol=1.0e-4):
    arr = np.asarray(path, dtype=float)
    if len(arr) <= 1:
        return arr.copy()
    substeps = max(1, int(substeps))
    out = [arr[0].copy()]
    for a, b in zip(arr[:-1], arr[1:]):
        local_substeps = substeps
        if robot is not None and frame is not None:
            local_substeps = _local_substeps_for_edge(robot, frame, a, b, substeps=substeps)
        for k in range(1, local_substeps):
            alpha = float(k) / float(local_substeps)
            q = _project_valid(mode, (1.0 - alpha) * a + alpha * b, tol=tol)
            if q is not None:
                out.append(q)
        out.append(b.copy())
    dense = np.asarray(out, dtype=float)
    dense[0] = arr[0]
    dense[-1] = arr[-1]
    return dense


def _smooth_segment_on_manifold(path, mode, rng, robot, frame,
                                shortcut_iterations=None,
                                shortcut_substeps=10, densify_substeps=10,
                                tol=1.0e-4):
    """Shortcut and densify a solved segment without leaving its Mode."""
    arr = np.asarray(path, dtype=float)
    if len(arr) == 0:
        return arr.copy()
    first = arr[0].copy()
    last = arr[-1].copy()
    try:
        shortened = _shortcut_segment_on_manifold(
            arr, mode, rng, robot, frame,
            iterations=shortcut_iterations,
            substeps=shortcut_substeps,
            tol=tol,
        )
        shortened[0] = first
        shortened[-1] = last
        dense = _densify_segment_on_manifold(
            shortened, mode,
            robot=robot,
            frame=frame,
            substeps=densify_substeps,
            tol=tol,
        )
        if len(dense) == 0:
            return arr.copy()
        dense[0] = arr[0]
        dense[-1] = arr[-1]

        before_second_len = _cartesian_ee_length(robot, frame, dense)
        second = _shortcut_segment_on_manifold(
            dense, mode, rng, robot, frame,
            iterations=max(150, len(dense)),
            substeps=shortcut_substeps,
            tol=tol,
        )
        if len(second) == 0:
            return arr.copy()
        second[0] = first
        second[-1] = last
        after_second_len = _cartesian_ee_length(robot, frame, second)
        if after_second_len < before_second_len - 1.0e-12:
            light_substeps = max(2, min(3, int(densify_substeps)))
            final = _densify_segment_on_manifold(
                second, mode,
                robot=robot,
                frame=frame,
                substeps=light_substeps,
                tol=tol,
            )
            if len(final) == 0:
                return arr.copy()
            final[0] = first
            final[-1] = last
            return final
        return second
    except Exception as exc:
        print(f"WARNING: on-manifold smoothing skipped for {mode.name}: {exc}")
        return arr.copy()


def _densely_verify_segment(path, mode, robot, frame, ee_step=COLLISION_CHECK_EE_STEP,
                            tol=1.0e-4):
    """Verify feasibility with controlled-frame samples spaced by at most ee_step."""
    arr = np.asarray(path, dtype=float)
    if len(arr) == 0:
        return False
    for q in arr:
        if not mode.on_manifold(q, tol=tol) or not mode.feasible(q):
            return False
    for a, b in zip(arr[:-1], arr[1:]):
        local_substeps = _local_substeps_for_edge(
            robot, frame, a, b, substeps=1, ee_step=ee_step
        )
        for k in range(1, local_substeps):
            alpha = float(k) / float(local_substeps)
            q = _project_valid(mode, (1.0 - alpha) * a + alpha * b, tol=tol)
            if q is None:
                return False
    return True


# --------------------------------------------------------------------------- #
#  Mode construction for a scene
# --------------------------------------------------------------------------- #
def build_modes(scene, flat_tol=0.0, workspace=None, support_margin=0.0,
                yaw_locks=None, yaw_band=0.0):
    """
    Returns dict: {"on": {surface_name: Mode}, "free": Mode, "torque": obj}.

    flat_tol > 0 relaxes "held flat" into a tilt band (a TSR), letting the
    planner CHOOSE a carry orientation within +-flat_tol rad on roll/pitch.
    workspace = (lo, hi) adds a WorkspaceBoundary to free_flight (keep the arc
    in front of the base), per the README's "Next" list.
    """
    r = scene.robot
    f = scene.controlled_frame
    torque = TorqueFeasibility(r, scene.tau_max, derate=scene.derate)
    joint_limit = JointLimit(r)
    collision = (
        CollisionConstraint(
            r,
            scene.collision_frames,
            scene.obstacles,
            frame_points=getattr(scene, "collision_frame_points", {}),
            self_collision_frames=getattr(scene, "self_collision_frames", []),
            self_collision_min_distance=getattr(scene, "self_collision_min_distance", 0.0),
        )
        if scene.obstacles or getattr(scene, "self_collision_frames", [])
        else None
    )

    def with_collision(extra):
        xs = [joint_limit] + list(extra)
        if collision is not None:
            xs.append(collision)
        return xs

    yaw_locks = yaw_locks or {}
    on = {}
    for s in scene.surfaces:
        if s.name in yaw_locks:
            eqs = [
                PlaneConstraint(r, f, s.height),
                FrameConstraint(
                    r,
                    f,
                    target_rotation=np.asarray(yaw_locks[s.name], dtype=float),
                    rot_mask=(1, 1, 1),
                    rot_tol=[flat_tol, flat_tol, float(yaw_band)],
                    name=f"flat_yawlock_{s.name}",
                ),
            ]
        else:
            eqs = [PlaneConstraint(r, f, s.height),
                   FlatCarryConstraint(r, f, R_flat=R_FLAT, tol=flat_tol)]
        ineqs = with_collision([])
        if s.xy_bounds is not None:
            (xlo, ylo), (xhi, yhi) = np.asarray(s.xy_bounds, float).reshape(2, 2)
            margin = max(0.0, float(support_margin))
            xlo_i, xhi_i = float(xlo) + margin, float(xhi) - margin
            ylo_i, yhi_i = float(ylo) + margin, float(yhi) - margin
            if xlo_i > xhi_i:
                cx = 0.5 * (float(xlo) + float(xhi))
                xlo_i, xhi_i = cx - 5.0e-7, cx + 5.0e-7
            if ylo_i > yhi_i:
                cy = 0.5 * (float(ylo) + float(yhi))
                ylo_i, yhi_i = cy - 5.0e-7, cy + 5.0e-7
            ineqs.append(WorkspaceBoundary(
                r, f,
                lo=[xlo_i, ylo_i, -1.0e3],
                hi=[xhi_i, yhi_i, 1.0e3],
                name=f"inside_{s.name}_footprint"))
        on[s.name] = Mode(f"on_{s.name}", equalities=eqs,
                          inequalities=ineqs)

    free_ineq = with_collision([torque])
    if workspace is not None:
        lo, hi = workspace
        free_ineq.append(WorkspaceBoundary(r, f, lo, hi))
    free = Mode("free_flight",
                equalities=[FlatCarryConstraint(r, f, R_flat=R_FLAT, tol=flat_tol)],
                inequalities=free_ineq)
    return {"on": on, "free": free, "torque": torque}


def place_config(scene, placement, seed, restarts=24, rng=None):
    """
    Project onto the full-pose placement -> a joint config (IK-free).  Tries the
    given seed first, then random restarts within joint limits, so a single
    shared seed need not serve every placement.
    """
    rng = rng or np.random.default_rng(0)
    r = scene.robot
    s = scene.surface(placement.surface)
    m = Mode("place", equalities=[FrameConstraint(
        r, scene.controlled_frame,
        target_position=[placement.x, placement.y, s.height], pos_mask=(1, 1, 1),
        target_rotation=placement.R, rot_mask=(1, 1, 0))])   # flat; yaw free
    q, ok = m.project(seed)
    if ok:
        return q, True
    qlo, qhi = r.joint_limits
    for _ in range(int(restarts)):
        q, ok = m.project(rng.uniform(qlo, qhi))
        if ok:
            return q, True
    return q, False


# --------------------------------------------------------------------------- #
#  The plan
# --------------------------------------------------------------------------- #
def plan(scene, start, goal,
         planner=None,
         flat_tol=0.0,
         workspace=None,
         support_margin=0.0,
         seed_q=None,
         n_transition_tries=60,
         perturb=0.25,
         max_pairs=10,
         lift_margin=0.0,
         w_backward=3.0,
         w_torque=0.5,
         refine_top_k=1,
         lock_slide_yaw=False,
         slide_yaw_band=0.0,
         rng=None,
         verbose=False,
         **planner_kw):
    """
    Plan from `start` Placement to `goal` Placement across the scene's surfaces.
    Returns a MultiModalResult.  `planner(robot, mode, q0, q1, **kw)` defaults to
    OMPL RRT-Connect (ompl_planner.plan_in_mode).

    w_backward penalizes transition pairs whose lift/set locations move away
    from the goal.  w_torque prefers transition pairs with lower worst-joint
    free-carry utilisation.  Pair search attributes failures to bad lifts, bad
    sets, or bad flight pairs, then resamples from the remaining candidates.
    """
    planner = planner or plan_in_mode
    rng = rng or np.random.default_rng(0)
    r = scene.robot
    f = scene.controlled_frame
    if seed_q is None:
        qlo, qhi = r.joint_limits
        seed_q = 0.5 * (np.asarray(qlo, float) + np.asarray(qhi, float))
    seed_q = np.asarray(seed_q, float)

    def log(*a):
        if verbose:
            print(*a)

    # Backward-compatible no-op for older scripts; the free flight is a single
    # unconstrained transport segment again.
    planner_kw.pop("transport_clearance", None)

    modes = build_modes(scene, flat_tol=flat_tol, workspace=workspace,
                        support_margin=support_margin)
    on, free, torque = modes["on"], modes["free"], modes["torque"]
    if start.surface not in on or goal.surface not in on:
        return MultiModalResult(False, "start/goal surface not in scene.")
    on_A, on_B = on[start.surface], on[goal.surface]

    # 1) resolve endpoint configs (IK-free projection onto the placements)
    q_start, ok_s = place_config(scene, start, seed_q, rng=rng)
    q_goal, ok_g = place_config(scene, goal, seed_q, rng=rng)
    if not ok_s or not ok_g:
        return MultiModalResult(False, "could not reach start/goal placement.",
                                q_start=q_start if ok_s else None,
                                q_goal=q_goal if ok_g else None)
    qlo, qhi = r.joint_limits
    q_start = wrap_to_limits(q_start, qlo, qhi)
    q_goal = wrap_to_limits(q_goal, qlo, qhi)
    if not (_in_limits(q_start, qlo, qhi) and _in_limits(q_goal, qlo, qhi)):
        return MultiModalResult(False, "start/goal config outside joint limits.",
                                q_start=q_start, q_goal=q_goal)
    log(f"   start config feasible on A: {on_A.feasible(q_start)} "
        f"(torque margin {torque.margin(q_start):+.3f})")
    log(f"   goal  config feasible on B: {on_B.feasible(q_goal)} "
        f"(torque margin {torque.margin(q_goal):+.3f})")
    yaw_locks = {}
    if lock_slide_yaw:
        R_A = r.frame_pose(scene.controlled_frame, q_start)[:3, :3].copy()
        R_B = r.frame_pose(scene.controlled_frame, q_goal)[:3, :3].copy()
        yaw_locks = {start.surface: R_A}
        if goal.surface != start.surface:
            yaw_locks[goal.surface] = R_B
        locked_modes = build_modes(
            scene,
            flat_tol=flat_tol,
            workspace=workspace,
            support_margin=support_margin,
            yaw_locks=yaw_locks,
            yaw_band=slide_yaw_band,
        )
        locked_on, locked_free, locked_torque = (
            locked_modes["on"],
            locked_modes["free"],
            locked_modes["torque"],
        )
        locked_on_A = locked_on[start.surface]
        locked_on_B = locked_on[goal.surface]
        if (
            locked_on_A.on_manifold(q_start)
            and locked_on_A.feasible(q_start)
            and locked_on_B.on_manifold(q_goal)
            and locked_on_B.feasible(q_goal)
        ):
            modes = locked_modes
            on, free, torque = locked_on, locked_free, locked_torque
            on_A, on_B = locked_on_A, locked_on_B
            log(
                "   slide yaw locked to endpoint orientations "
                f"(yaw_band={float(slide_yaw_band):.3g})"
            )
        else:
            yaw_locks = {}
            print("WARNING: slide yaw-lock check failed; planning with yaw-free slides.")

    # 2) sample transitions (projected onto the stacked manifolds)
    lifts = sample_transition(on_A, free, q_start, n_tries=n_transition_tries,
                              perturb=perturb, rng=rng, q_lo=qlo, q_hi=qhi)
    sets = sample_transition(free, on_B, q_goal, n_tries=n_transition_tries,
                             perturb=perturb, rng=rng, q_lo=qlo, q_hi=qhi)
    lifts = [wrap_to_limits(q, qlo, qhi) for q in lifts]
    sets = [wrap_to_limits(q, qlo, qhi) for q in sets]
    lifts = [q for q in lifts if _in_limits(q, qlo, qhi)]
    sets = [q for q in sets if _in_limits(q, qlo, qhi)]
    if not lifts or not sets:
        free_only = Mode(
            "free_diag",
            equalities=[FlatCarryConstraint(r, f, R_flat=R_FLAT, tol=flat_tol)],
            inequalities=[torque, JointLimit(r)],
        )
        n_free_feasible = sum(1 for q in lifts + sets if free_only.feasible(q))
        hint = (
            "Lift region infeasible (mass too high for any free carry)."
            if n_free_feasible == 0
            else "Sampling missed the feasible set (try larger n_transition_tries or different perturb)."
        )
        return MultiModalResult(
            False,
            f"no transitions: lift-off={len(lifts)}, set-down={len(sets)}, "
            f"free-mode-feasible-among-sampled={n_free_feasible}. {hint}",
            q_start=q_start, q_goal=q_goal, n_lift=len(lifts), n_set=len(sets))

    requested_lift_margin = max(0.0, float(lift_margin))
    if requested_lift_margin > 0.0:
        raw_lift_count, raw_set_count = len(lifts), len(sets)
        lifts = [q for q in lifts if torque.margin(q) >= requested_lift_margin]
        sets = [q for q in sets if torque.margin(q) >= requested_lift_margin]
        log(
            f"   lift_margin={requested_lift_margin:.3f}: "
            f"lift-off {raw_lift_count}->{len(lifts)}, set-down {raw_set_count}->{len(sets)}"
        )
        if not lifts or not sets:
            return MultiModalResult(
                False,
                f"no transitions after lift_margin={requested_lift_margin} filter: "
                f"lift-off={len(lifts)}, set-down={len(sets)}.",
                q_start=q_start, q_goal=q_goal, n_lift=len(lifts), n_set=len(sets))

    log(f"   sampled {len(lifts)} lift-off and {len(sets)} set-down transitions")

    def frame_position(q):
        return np.asarray(r.frame_pose(scene.controlled_frame, q)[:3, 3], dtype=float)

    p_start = frame_position(q_start)
    p_goal = frame_position(q_goal)
    p_lifts = [frame_position(q) for q in lifts]
    p_sets = [frame_position(q) for q in sets]
    util_lifts = [float(np.max(torque.utilisation(q))) for q in lifts]
    util_sets = [float(np.max(torque.utilisation(q))) for q in sets]
    d0 = float(np.linalg.norm(p_start - p_goal))

    # 3) discrete search over (lift-off, set-down) pairs.
    #    Slides don't depend on the other transition, so cache them.  Failed
    #    slides poison their transition index; failed flights poison their pair.
    seg1_cache = {}
    seg3_cache = {}
    seg2_attempts = {}
    all_pairs = [(i, j) for i in range(len(p_lifts)) for j in range(len(p_sets))]
    rng_pairs = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
    rejected_lifts = set()
    rejected_sets = set()
    rejected_pairs = set()
    pair_attempt_counts = {}
    tried = 0
    n_total_budget = max(1, int(max_pairs)) * 4

    def _pair_metrics(i, j):
        p_lift = p_lifts[i]
        p_set = p_sets[j]
        path_len = float(np.linalg.norm(p_start - p_lift) +
                         np.linalg.norm(p_lift - p_set) +
                         np.linalg.norm(p_set - p_goal))
        d1 = float(np.linalg.norm(p_lift - p_goal))
        d2 = float(np.linalg.norm(p_set - p_goal))
        backward = max(0.0, d1 - d0) + max(0.0, d2 - d1)
        return path_len, backward, d1, d2

    def _pair_score(i, j):
        path_len, backward, _, _ = _pair_metrics(i, j)
        flight_len = float(np.linalg.norm(p_lifts[i] - p_sets[j]))
        util = max(util_lifts[i], util_sets[j])
        return float(
            flight_len + 0.25 * path_len +
            float(w_backward) * backward + float(w_torque) * util
        )

    def _sample_next_pair():
        candidates = [
            (i, j) for (i, j) in all_pairs
            if i not in rejected_lifts
            and j not in rejected_sets
            and (i, j) not in rejected_pairs
        ]
        revisit = (
            len(candidates) < len(all_pairs)
            and (not candidates or rng_pairs.random() < 0.1)
        )
        if revisit:
            candidates = list(all_pairs)
        if not candidates:
            return None
        candidates.sort(key=lambda ij: (pair_attempt_counts.get(ij, 0), _pair_score(*ij)))
        return candidates[0]

    while tried < n_total_budget:
        pick = _sample_next_pair()
        if pick is None:
            break
        i, j = pick
        pair_attempt_counts[pick] = pair_attempt_counts.get(pick, 0) + 1
        tried += 1
        ql = lifts[i]
        qs = sets[j]
        if i not in seg1_cache:
            seg1_cache[i] = planner(r, on_A, q_start, ql, **planner_kw)
        seg1 = seg1_cache[i]
        if not seg1.success:
            log(f"   [lift {i}, set {j}] slide-A failed: {seg1.message} "
                f"-> rejecting lift {i}")
            rejected_lifts.add(i)
            continue
        if (i, j) not in seg2_attempts:
            seg2_attempts[(i, j)] = planner(r, free, ql, qs, **planner_kw)
        seg2 = seg2_attempts[(i, j)]
        if not seg2.success:
            log(f"   [lift {i}, set {j}] flight failed: {seg2.message}")
            rejected_pairs.add((i, j))
            flight_fails_for_lift_i = sum(
                1 for (ii, _), result in seg2_attempts.items()
                if ii == i and not result.success
            )
            if flight_fails_for_lift_i >= 3:
                log(f"   -> lift {i} has {flight_fails_for_lift_i} "
                    f"flight fails; rejecting lift {i}")
                rejected_lifts.add(i)
            continue
        if j not in seg3_cache:
            seg3_cache[j] = planner(r, on_B, qs, q_goal, **planner_kw)
        seg3 = seg3_cache[j]
        if not seg3.success:
            log(f"   [lift {i}, set {j}] slide-B failed: {seg3.message} "
                f"-> rejecting set {j}")
            rejected_sets.add(j)
            continue
        path_len, backward, d1, d2 = _pair_metrics(i, j)
        log(
            f"   [lift {i}, set {j}] all three segments solved "
            f"(path_len {path_len:.3f}, backward {backward:.3f}, "
            f"d0/d1/d2 {d0:.3f}/{d1:.3f}/{d2:.3f})"
        )
        labels = [f"slide_{start.surface}", "free_flight", f"slide_{goal.surface}"]
        results = [seg1, seg2, seg3]
        segment_modes = [on_A, free, on_B]
        smoothed_paths = []
        for label, result, mode in zip(labels, results, segment_modes):
            original = np.asarray(result.path, dtype=float)
            smoothed = _smooth_segment_on_manifold(
                result.path, mode, rng, scene.robot, scene.controlled_frame
            )
            if _densely_verify_segment(
                smoothed, mode, scene.robot, scene.controlled_frame
            ):
                path = smoothed
            else:
                path = original
                log(f"   smoothing broke feasibility for {label} -> kept certified planner path")
            result.path = path
            result.path_points = int(len(path))
            if len(path):
                try:
                    result.max_equality_residual = max(
                        float(np.linalg.norm(mode.equality_residual(q))) for q in path
                    )
                    result.feasible_along_path = _densely_verify_segment(
                        path, mode, scene.robot, scene.controlled_frame
                    )
                except Exception:
                    pass
            smoothed_paths.append(path)
            log(f"   smoothed {label}: {len(path)} pts")
        return MultiModalResult(
            True, "multimodal plan found (informed search).",
            segments=[(label, path) for label, path in zip(labels, smoothed_paths)],
            segment_results=[(label, result) for label, result in zip(labels, results)],
            lift_off=ql, set_down=qs, q_start=q_start, q_goal=q_goal,
            n_lift=len(lifts), n_set=len(sets), pairs_tried=tried,
            yaw_locks={k: np.asarray(v, dtype=float).copy()
                       for k, v in yaw_locks.items()},
            slide_yaw_band=float(slide_yaw_band))

    return MultiModalResult(
        False, f"no feasible (lift-off, set-down) pair in {tried} tries.",
        q_start=q_start, q_goal=q_goal, n_lift=len(lifts), n_set=len(sets),
        pairs_tried=tried)


# --------------------------------------------------------------------------- #
#  OMPL-free smoke-test planner (NOT a search planner; no obstacle avoidance)
# --------------------------------------------------------------------------- #
def straightline_planner(robot, mode, q_start, q_goal, steps=40, **_):
    """
    Joint-space lerp with per-waypoint projection onto the mode manifold and a
    feasibility check.  Lets the whole pipeline run without OMPL installed.
    Use ompl_planner.plan_in_mode (RRT-Connect) for real planning.
    """
    qs, ok0 = mode.project(q_start)
    qg, ok1 = mode.project(q_goal)
    if not (ok0 and ok1):
        return PlanResult(False, None, "straightline: endpoint projection failed",
                          planner="straightline")
    path = []
    for t in np.linspace(0.0, 1.0, steps):
        q, ok = mode.project((1 - t) * qs + t * qg)
        if not ok:
            return PlanResult(False, None, f"straightline: projection failed @ t={t:.2f}",
                              planner="straightline")
        if not mode.feasible(q):
            return PlanResult(False, None, f"straightline: infeasible @ t={t:.2f}",
                              planner="straightline")
        path.append(q)
    path = np.asarray(path)
    res = max(float(np.linalg.norm(mode.equality_residual(q))) for q in path)
    return PlanResult(True, path, "straightline (offline smoke test)",
                      path_points=len(path), max_equality_residual=res,
                      feasible_along_path=True, planner="straightline")


__all__ = ["Placement", "MultiModalResult", "build_modes", "best_utilisation",
           "place_config", "plan", "straightline_planner",
           "uniform_resample_result", "R_FLAT"]