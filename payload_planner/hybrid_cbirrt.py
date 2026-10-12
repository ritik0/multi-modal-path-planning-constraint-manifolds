"""Experimental CBiRRT-style hybrid planner for the payload scene.

This module deliberately treats the existing Mode objects as black boxes.  It
does not change torque, collision, table, obstacle, or projection definitions;
it only changes the search strategy used to find a joint-space path.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from types import SimpleNamespace

import numpy as np

import multimodal as mm

MAX_DEBUG_POINTS = int(os.environ.get("PAYLOAD_MAX_DEBUG_POINTS", "5000"))
PAYLOAD_HYBRID_SMOOTHING = int(os.environ.get("PAYLOAD_HYBRID_SMOOTHING", "100"))
PAYLOAD_SHORTCUT_WJOINT = float(os.environ.get("PAYLOAD_SHORTCUT_WJOINT", "0.05"))


@dataclass
class HybridNode:
    q: np.ndarray
    mode_name: str
    parent: int | None


def _mode_for_name(modes, mode_name: str):
    if mode_name == "free":
        return modes["free"]
    if mode_name.startswith("on_"):
        return modes["on"][mode_name[3:]]
    raise KeyError(f"unknown hybrid mode {mode_name!r}")


def _segment_label(mode_name: str) -> str:
    if mode_name == "free":
        return "free_flight"
    if mode_name.startswith("on_"):
        return "slide_" + mode_name[3:]
    return mode_name


def _constraint_values(mode, q):
    return [
        (getattr(c, "name", c.__class__.__name__), float(c.value(q)))
        for c in mode.inequalities
    ]


def _violations(mode, q, tol=1.0e-9):
    return [(name, value) for name, value in _constraint_values(mode, q) if value > tol]


def _collision_hit(mode, q):
    for c in mode.inequalities:
        if getattr(c, "name", "") == "collision_free" and hasattr(c, "hit_info"):
            hit = c.hit_info(q)
            if hit is not None:
                return hit
    return None


def _bump(counter, key, amount=1):
    counter[key] = int(counter.get(key, 0)) + int(amount)


def _new_stats(scene):
    supported = {s.name: 0 for s in scene.surfaces}
    return {
        "accepted_free": 0,
        "accepted_supported": supported,
        "rejected": {},
        "collision_hits": {},
        "failed_edges": 0,
        "connect_attempts": 0,
        "debug": {
            "ambient_payload_points": [],
            "accepted_payload_points": [],
            "accepted_modes": [],
            "rejected_payload_points": [],
            "rejected_reasons": [],
            "tree_edges": [],
            "tree_edge_modes": [],
            "final_mode_labels": [],
            "final_switch_indices": [],
        },
    }


def _debug_add(debug, key, value):
    if len(debug.get(key, [])) < MAX_DEBUG_POINTS:
        debug[key].append(value)


def _payload_pos(scene, q):
    return np.asarray(scene.robot.frame_pose(scene.controlled_frame, q), dtype=float)[:3, 3]


def _record_reject(stats, reason, mode=None, q=None, scene=None):
    _bump(stats["rejected"], reason)
    if q is not None and scene is not None:
        _debug_add(stats["debug"], "rejected_payload_points", _payload_pos(scene, q))
        _debug_add(stats["debug"], "rejected_reasons", str(reason))
    if mode is not None and q is not None:
        hit = _collision_hit(mode, q)
        if hit is not None:
            if hit.get("kind") == "environment":
                key = str(hit.get("object_name", "<unknown>"))
            else:
                key = f"self:{hit.get('frame_a', '?')}:{hit.get('frame_b', '?')}"
            _bump(stats["collision_hits"], key)


def _top(counter, limit=6):
    items = sorted(counter.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return ", ".join(f"{k}={v}" for k, v in items[:limit]) or "none"


def _valid_in_mode(mode, q, qlo, qhi):
    return (
        mm._in_limits(q, qlo, qhi)
        and mode.on_manifold(q)
        and mode.feasible(q)
    )


def _project_mode(mode, q_raw, qlo, qhi):
    q, ok = mode.project(q_raw, qlo=qlo, qhi=qhi)
    if not ok or q is None:
        return None
    q = mm.wrap_to_limits(np.asarray(q, dtype=float), qlo, qhi)
    return q


def constrain_config(scene, modes, q_old, q_raw, previous_mode=None,
                     qlo=None, qhi=None, rng=None, stats=None):
    """Project q_raw to whichever existing mode can make it valid."""
    qlo = np.asarray(qlo if qlo is not None else scene.robot.joint_limits[0], dtype=float)
    qhi = np.asarray(qhi if qhi is not None else scene.robot.joint_limits[1], dtype=float)
    q_raw = np.asarray(q_raw, dtype=float)
    attempts = [("free", modes["free"])]
    attempts.extend((f"on_{surface.name}", modes["on"][surface.name]) for surface in scene.surfaces)

    valid = []
    for mode_name, mode in attempts:
        q = _project_mode(mode, q_raw, qlo, qhi)
        if q is None:
            if stats is not None:
                _record_reject(stats, f"{mode_name}:equality", scene=scene, q=q_raw)
            continue
        if not mm._in_limits(q, qlo, qhi):
            if stats is not None:
                _record_reject(stats, f"{mode_name}:joint", scene=scene, q=q)
            continue
        if not mode.on_manifold(q):
            if stats is not None:
                _record_reject(stats, f"{mode_name}:manifold", scene=scene, q=q)
            continue
        if not mode.feasible(q):
            if stats is not None:
                violations = _violations(mode, q)
                reason = f"{mode_name}:feasible"
                if violations:
                    reason += ":" + ",".join(name for name, _ in violations[:2])
                _record_reject(stats, reason, mode, q, scene=scene)
            continue
        valid.append((float(np.linalg.norm(q - q_raw)), mode_name, q))

    if not valid:
        return None
    valid.sort(key=lambda item: item[0])
    _, mode_name, q = valid[0]
    if stats is not None:
        if mode_name == "free":
            stats["accepted_free"] += 1
        elif mode_name.startswith("on_"):
            _bump(stats["accepted_supported"], mode_name[3:])
        _debug_add(stats["debug"], "accepted_payload_points", _payload_pos(scene, q))
        _debug_add(stats["debug"], "accepted_modes", mode_name)
    return q, mode_name


def edge_valid(scene, modes, q_a, mode_a, q_b, mode_b, step_size,
               qlo=None, qhi=None, rng=None, stats=None):
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    distance = float(np.linalg.norm(q_b - q_a))
    steps = max(1, int(np.ceil(distance / max(float(step_size), 1.0e-9))))
    qs = []
    labels = []
    previous = mode_a
    q_prev = q_a
    for k in range(1, steps):
        alpha = float(k) / float(steps)
        q_raw = (1.0 - alpha) * q_a + alpha * q_b
        constrained = constrain_config(
            scene, modes, q_prev, q_raw, previous_mode=previous,
            qlo=qlo, qhi=qhi, rng=rng, stats=stats,
        )
        if constrained is None:
            if stats is not None:
                stats["failed_edges"] += 1
            return False, [], []
        q_prev, previous = constrained
        qs.append(q_prev)
        labels.append(previous)
    target_mode = _mode_for_name(modes, mode_b)
    if not _valid_in_mode(target_mode, q_b, qlo, qhi):
        if stats is not None:
            _record_reject(stats, f"{mode_b}:endpoint", target_mode, q_b, scene=scene)
            stats["failed_edges"] += 1
        return False, [], []
    qs.append(q_b.copy())
    labels.append(mode_b)
    return True, qs, labels


def _nearest(tree, q):
    distances = [float(np.linalg.norm(node.q - q)) for node in tree]
    return int(np.argmin(distances))


def _add_node(tree, q, mode_name, parent):
    tree.append(HybridNode(np.asarray(q, dtype=float), str(mode_name), int(parent)))
    return len(tree) - 1


def constrained_extend(tree, q_target, scene, modes, step_size, max_extend_steps,
                       qlo, qhi, rng, stats):
    nearest = _nearest(tree, q_target)
    current = nearest
    reached = False
    for _ in range(int(max_extend_steps)):
        q_near = tree[current].q
        direction = np.asarray(q_target, dtype=float) - q_near
        distance = float(np.linalg.norm(direction))
        if distance <= 1.0e-12:
            reached = True
            break
        step = min(float(step_size), distance)
        q_raw = q_near + (step / distance) * direction
        constrained = constrain_config(
            scene, modes, q_near, q_raw,
            previous_mode=tree[current].mode_name,
            qlo=qlo, qhi=qhi, rng=rng, stats=stats,
        )
        if constrained is None:
            break
        q_new, mode_new = constrained
        ok, qs, labels = edge_valid(
            scene, modes, q_near, tree[current].mode_name,
            q_new, mode_new, step_size=max(float(step_size) * 0.5, 0.02),
            qlo=qlo, qhi=qhi, rng=rng, stats=stats,
        )
        if not ok:
            break
        parent = current
        for q_edge, mode_edge in zip(qs, labels):
            parent_q = tree[parent].q
            parent_mode = tree[parent].mode_name
            parent = _add_node(tree, q_edge, mode_edge, parent)
            debug = stats["debug"]
            _debug_add(
                debug,
                "tree_edges",
                np.asarray([_payload_pos(scene, parent_q), _payload_pos(scene, q_edge)], dtype=float),
            )
            _debug_add(debug, "tree_edge_modes", f"{parent_mode}->{mode_edge}")
        current = parent
        if float(np.linalg.norm(tree[current].q - q_target)) <= float(step_size):
            reached = True
            break
    return current, reached


def _trace_tree(tree, idx):
    out = []
    while idx is not None:
        out.append(idx)
        idx = tree[idx].parent
    out.reverse()
    return out


def _path_from_indices(tree, indices):
    return [tree[i].q for i in indices], [tree[i].mode_name for i in indices]


def _combine_paths(tree_start, idx_start, tree_goal, idx_goal):
    start_indices = _trace_tree(tree_start, idx_start)
    goal_indices = _trace_tree(tree_goal, idx_goal)
    qs_a, modes_a = _path_from_indices(tree_start, start_indices)
    qs_b, modes_b = _path_from_indices(tree_goal, goal_indices)
    qs = qs_a + list(reversed(qs_b))[1:]
    labels = modes_a + list(reversed(modes_b))[1:]
    return np.asarray(qs, dtype=float), labels


def _split_segments(qs, mode_labels):
    segments = []
    if len(qs) == 0:
        return segments
    start = 0
    for i in range(1, len(qs) + 1):
        if i == len(qs) or mode_labels[i] != mode_labels[start]:
            label = _segment_label(mode_labels[start])
            segments.append((label, np.asarray(qs[start:i], dtype=float)))
            start = i
    return segments


def _switch_indices(mode_labels):
    return np.asarray(
        [i for i in range(1, len(mode_labels)) if mode_labels[i] != mode_labels[i - 1]],
        dtype=int,
    )


def _transition_config(qs, mode_labels, from_mode, to_mode):
    for i in range(1, len(mode_labels)):
        if mode_labels[i - 1] == from_mode and mode_labels[i] == to_mode:
            return qs[i]
    return None


def _path_lengths(scene, qs):
    arr = np.asarray(qs, dtype=float)
    if len(arr) <= 1:
        return 0.0, 0.0
    ee = np.asarray(
        [scene.robot.frame_pose(scene.controlled_frame, q)[:3, 3] for q in arr],
        dtype=float,
    )
    ee_len = float(np.sum(np.linalg.norm(np.diff(ee, axis=0), axis=1)))
    joint_len = float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
    return ee_len, joint_len


def _path_cost(scene, qs, w_joint=PAYLOAD_SHORTCUT_WJOINT):
    ee_len, joint_len = _path_lengths(scene, qs)
    return float(ee_len + float(w_joint) * joint_len)


def _final_dense_verify(scene, modes, segments):
    for label, path in segments:
        mode_name = "free" if label == "free_flight" else "on_" + label[len("slide_"):]
        mode = _mode_for_name(modes, mode_name)
        if mm._densely_verify_segment(path, mode, scene.robot, scene.controlled_frame):
            continue
        arr = np.asarray(path, dtype=float)
        for i, q in enumerate(arr):
            residual = float(np.linalg.norm(mode.equality_residual(q)))
            violations = _violations(mode, q)
            hit = _collision_hit(mode, q)
            if residual > 1.0e-4 or violations or hit is not None:
                return False, (
                    f"dense validation failed in {label} at waypoint {i}: "
                    f"equality_residual={residual:.3e}, "
                    f"violations={violations}, collision_hit={hit}"
                )
        return False, f"dense validation failed in {label} between stored waypoints"
    return True, ""


def _shortcut_path(scene, modes, qs, mode_labels, rng, step_size, qlo, qhi, stats,
                   attempts=100):
    qs = [np.asarray(q, dtype=float) for q in qs]
    labels = list(mode_labels)
    start_ee_len, start_joint_len = _path_lengths(scene, qs)
    start_waypoints = len(qs)
    if len(qs) <= 2:
        print(
            "shortcut: "
            f"ee_len {start_ee_len:.3f}->{start_ee_len:.3f} m, "
            f"joints {start_joint_len:.3f}->{start_joint_len:.3f}, "
            f"waypoints {start_waypoints}->{start_waypoints}"
        )
        return np.asarray(qs, dtype=float), labels
    current_cost = _path_cost(scene, qs)
    no_improve = 0
    early_stop = max(50, int(attempts) // 4)
    for _ in range(int(attempts)):
        if len(qs) <= 2:
            break
        i = int(rng.integers(0, len(qs) - 2))
        j = int(rng.integers(i + 2, len(qs)))
        ok, mid_qs, mid_labels = edge_valid(
            scene, modes, qs[i], labels[i], qs[j], labels[j],
            step_size=max(float(step_size) * 0.5, 0.02),
            qlo=qlo, qhi=qhi, rng=rng, stats=stats,
        )
        if not ok:
            no_improve += 1
            if no_improve >= early_stop:
                break
            continue
        candidate_qs = qs[: i + 1] + mid_qs + qs[j + 1 :]
        candidate_labels = labels[: i + 1] + mid_labels + labels[j + 1 :]
        candidate_cost = _path_cost(scene, candidate_qs)
        if candidate_cost < current_cost - 1.0e-6:
            qs, labels = candidate_qs, candidate_labels
            current_cost = candidate_cost
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop:
                break
    end_ee_len, end_joint_len = _path_lengths(scene, qs)
    print(
        "shortcut: "
        f"ee_len {start_ee_len:.3f}->{end_ee_len:.3f} m, "
        f"joints {start_joint_len:.3f}->{end_joint_len:.3f}, "
        f"waypoints {start_waypoints}->{len(qs)}"
    )
    return np.asarray(qs, dtype=float), labels


def _print_diagnostics(stats, tree_start, tree_goal, qs=None, mode_labels=None, torque=None):
    print("hybrid_cbirrt diagnostics")
    print(f"  tree nodes: start={len(tree_start)}, goal={len(tree_goal)}")
    print(f"  accepted free nodes: {stats['accepted_free']}")
    print(f"  accepted supported nodes: {stats['accepted_supported']}")
    print(f"  rejected attempts: {_top(stats['rejected'])}")
    print(f"  collision hits: {_top(stats['collision_hits'])}")
    print(f"  failed edges: {stats['failed_edges']}, connect_attempts={stats['connect_attempts']}")
    if qs is not None and mode_labels is not None:
        switches = _switch_indices(mode_labels)
        sequence = [_segment_label(label) for label in mode_labels]
        compact = [sequence[0]] if sequence else []
        for label in sequence[1:]:
            if label != compact[-1]:
                compact.append(label)
        print(f"  mode switches: {len(switches)}")
        print(f"  final mode sequence: {' -> '.join(compact)}")
        if torque is not None:
            free_margins = [float(torque.margin(q)) for q, m in zip(qs, mode_labels) if m == "free"]
            if free_margins:
                print(f"  min free-flight torque margin: {min(free_margins):+.3f}")


def plan_hybrid_cbirrt(
    scene,
    start,
    goal,
    seed_q=None,
    support_margin=0.0,
    flat_tol=0.0,
    workspace=None,
    rng=None,
    max_time=60.0,
    step_size=0.18,
    goal_tolerance=0.08,
    goal_bias=0.20,
    connect_attempts=12,
    max_iterations=20000,
    smoothing_attempts=None,
    verbose=False,
):
    rng = rng or np.random.default_rng(0)
    modes = mm.build_modes(
        scene,
        flat_tol=flat_tol,
        workspace=workspace,
        support_margin=support_margin,
    )
    on_modes = modes["on"]
    free_mode = modes["free"]
    torque = modes["torque"]
    qlo, qhi = scene.robot.joint_limits
    qlo = np.asarray(qlo, dtype=float)
    qhi = np.asarray(qhi, dtype=float)
    if seed_q is None:
        seed_q = 0.5 * (qlo + qhi)
    seed_q = np.asarray(seed_q, dtype=float)

    if start.surface not in on_modes or goal.surface not in on_modes:
        return mm.MultiModalResult(False, "start/goal surface not in scene")

    q_start, ok_s, diag_s = mm._select_supported_config(
        scene, start, on_modes[start.surface], seed_q, rng, torque,
        require_torque_infeasible=False,
    )
    q_goal, ok_g, diag_g = mm._select_supported_config(
        scene, goal, on_modes[goal.surface], seed_q, rng, torque,
        require_torque_infeasible=False,
    )
    if not ok_s or not ok_g:
        return mm.MultiModalResult(
            False,
            "hybrid_cbirrt could not make start/goal supported-feasible. "
            f"start: {diag_s}; goal: {diag_g}",
            q_start=q_start if ok_s else None,
            q_goal=q_goal if ok_g else None,
        )
    print(
        "hybrid start/goal: "
        f"start_supported={on_modes[start.surface].on_manifold(q_start) and on_modes[start.surface].feasible(q_start)} "
        f"torque_margin={torque.margin(q_start):+.3f}; "
        f"goal_supported={on_modes[goal.surface].on_manifold(q_goal) and on_modes[goal.surface].feasible(q_goal)} "
        f"torque_margin={torque.margin(q_goal):+.3f}"
    )

    stats = _new_stats(scene)
    tree_start = [HybridNode(q_start, f"on_{start.surface}", None)]
    tree_goal = [HybridNode(q_goal, f"on_{goal.surface}", None)]
    tic = time.perf_counter()

    for iteration in range(int(max_iterations)):
        if time.perf_counter() - tic > float(max_time):
            break
        if rng.random() < float(goal_bias):
            q_sample = tree_goal[int(rng.integers(0, len(tree_goal)))].q
        else:
            q_sample = rng.uniform(qlo, qhi)
            _debug_add(stats["debug"], "ambient_payload_points", _payload_pos(scene, q_sample))

        active_start = (iteration % 2 == 0)
        tree_a = tree_start if active_start else tree_goal
        tree_b = tree_goal if active_start else tree_start
        idx_a, _ = constrained_extend(
            tree_a, q_sample, scene, modes, step_size, connect_attempts,
            qlo, qhi, rng, stats,
        )
        q_reached = tree_a[idx_a].q
        idx_b, reached_b = constrained_extend(
            tree_b, q_reached, scene, modes, step_size, connect_attempts,
            qlo, qhi, rng, stats,
        )
        stats["connect_attempts"] += 1
        if not reached_b:
            continue
        if float(np.linalg.norm(tree_a[idx_a].q - tree_b[idx_b].q)) > float(goal_tolerance):
            continue
        ok, conn_qs, conn_modes = edge_valid(
            scene, modes,
            tree_a[idx_a].q, tree_a[idx_a].mode_name,
            tree_b[idx_b].q, tree_b[idx_b].mode_name,
            step_size=max(float(step_size) * 0.5, 0.02),
            qlo=qlo, qhi=qhi, rng=rng, stats=stats,
        )
        if not ok:
            continue

        if active_start:
            start_indices = _trace_tree(tree_start, idx_a)
            goal_indices = _trace_tree(tree_goal, idx_b)
            qs_a, modes_a = _path_from_indices(tree_start, start_indices)
            qs_b, modes_b = _path_from_indices(tree_goal, goal_indices)
            qs = qs_a + conn_qs + list(reversed(qs_b[:-1]))
            mode_labels = modes_a + conn_modes + list(reversed(modes_b[:-1]))
        else:
            start_indices = _trace_tree(tree_start, idx_b)
            goal_indices = _trace_tree(tree_goal, idx_a)
            qs_s, modes_s = _path_from_indices(tree_start, start_indices)
            qs_g, modes_g = _path_from_indices(tree_goal, goal_indices)
            qs = qs_s + list(reversed(conn_qs))[1:] + list(reversed(qs_g[:-1]))
            mode_labels = modes_s + list(reversed(conn_modes))[1:] + list(reversed(modes_g[:-1]))
        qs = np.asarray(qs, dtype=float)
        qs, mode_labels = _shortcut_path(
            scene,
            modes,
            qs,
            mode_labels,
            rng,
            step_size,
            qlo,
            qhi,
            stats,
            attempts=PAYLOAD_HYBRID_SMOOTHING if smoothing_attempts is None else smoothing_attempts,
        )
        segments = _split_segments(qs, mode_labels)
        valid, message = _final_dense_verify(scene, modes, segments)
        if not valid:
            print("hybrid_cbirrt dense validation rejected candidate:", message)
            continue

        lift = _transition_config(qs, mode_labels, f"on_{start.surface}", "free")
        set_down = _transition_config(qs, mode_labels, "free", f"on_{goal.surface}")
        result = mm.MultiModalResult(
            True,
            f"hybrid_cbirrt found path in {iteration + 1} iterations.",
            segments=segments,
            segment_results=[
                (
                    label,
                    SimpleNamespace(
                        success=True,
                        path=path,
                        message="hybrid_cbirrt segment",
                        feasible_along_path=True,
                        planner="hybrid_cbirrt",
                        constrained_space="hybrid",
                    ),
                )
                for label, path in segments
            ],
            lift_off=lift,
            set_down=set_down,
            q_start=q_start,
            q_goal=q_goal,
            n_lift=1 if lift is not None else 0,
            n_set=1 if set_down is not None else 0,
            pairs_tried=int(stats["connect_attempts"]),
        )
        result.mode_labels = np.asarray(mode_labels, dtype=object)
        result.hybrid_mode_sequence = np.asarray([_segment_label(m) for m in mode_labels], dtype=object)
        result.hybrid_switch_indices = _switch_indices(mode_labels)
        stats["debug"]["final_mode_labels"] = list(mode_labels)
        stats["debug"]["final_switch_indices"] = list(result.hybrid_switch_indices)
        result.hybrid_stats = stats
        result.hybrid_debug = stats["debug"]
        _print_diagnostics(stats, tree_start, tree_goal, qs, mode_labels, torque)
        return result

    _print_diagnostics(stats, tree_start, tree_goal)
    return mm.MultiModalResult(
        False,
        "hybrid_cbirrt failed to connect trees. "
        f"dominant rejections=[{_top(stats['rejected'])}], "
        f"collision_hits=[{_top(stats['collision_hits'])}]",
        q_start=q_start,
        q_goal=q_goal,
        pairs_tried=int(stats["connect_attempts"]),
    )


__all__ = [
    "HybridNode",
    "constrain_config",
    "edge_valid",
    "constrained_extend",
    "plan_hybrid_cbirrt",
]
