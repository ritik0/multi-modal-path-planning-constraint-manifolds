"""Export an executable Panda payload trajectory and a decimated viz copy.

The geometric path is loaded from ``panda_plan.npz`` and is not modified here.
This script only applies time parameterization and writes CSV files.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_PLAN = HERE / "panda_plan.npz"
DEFAULT_EXEC_CSV = HERE / "system_traj.csv"
DEFAULT_VIZ_CSV = HERE / "traj_viz.csv"
DEFAULT_CONTACT_DIAG_CSV = HERE / "traj_contact_diagnostics.csv"

DT = 1.0e-3
VIZ_DT = 0.0333
SAFETY_FRACTION = 0.3
STOP_SPEED_THRESHOLD = 0.02
STOP_WINDOW_S = 0.25
JOIN_TOL = 1.0e-3

FRANKA_QDOT_MAX = np.array(
    [2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610],
    dtype=float,
)

# Franka Panda datasheet acceleration limits, rad/s^2.  URDFs generally do not
# carry acceleration limits.  Confirm these against the exact robot variant in
# use; Panda and FR3 limits differ slightly.
FRANKA_QDDOT_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], dtype=float)
R_FLAT = np.diag([1.0, -1.0, -1.0])


def _validate_joint_path(path, name: str) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if path.ndim != 2:
        raise ValueError(f"{name} must be a 2D joint path, got shape {path.shape}")
    if len(path) < 2:
        raise ValueError(f"{name} must contain at least two configurations")
    return path


def _is_grasped_plan(data) -> bool:
    files = set(data.files)
    return (
        "path" in files
        and (
            "segments" not in files
            or "regrasp_indices" in files
            or "grasp_yaw" in files
        )
    )


def _normalise_grasped_label(label: str) -> str:
    text = str(label).strip()
    low = text.lower()
    if low == "flight":
        return "free_flight"
    if low.startswith("slide_"):
        return low
    return low or "segment"


def _as_index(data, key: str, n: int) -> int | None:
    if key not in data.files:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size == 0:
        return None
    idx = int(arr[0])
    if idx < 0 or idx >= n:
        raise ValueError(f"{key}={idx} is outside path index range [0, {n - 1}]")
    return idx


def _warn_if_nonzero_grasp_tilt(data) -> None:
    tilt_keys = [
        "grasp_tilt",
        "grasp_tilts",
        "chosen_grasp_tilt",
        "chosen_grasp_tilts",
        "tilt",
        "tilts",
    ]
    tilts = []
    for key in tilt_keys:
        if key in data.files:
            values = np.asarray(data[key], dtype=float).reshape(-1)
            if values.size:
                tilts.append(values)
    if not tilts:
        return
    all_tilts = np.concatenate(tilts)
    if np.any(np.abs(all_tilts) > 1.0e-10):
        print(
            "WARNING: grasped plan contains non-zero grasp tilt. "
            "The baked-URDF dynamic torque check is exact only for pure-yaw "
            "grasps; use the per-grasp model for tilted grasps."
        )


def _load_grasped_plan_segments(data, plan_path: Path):
    q = _validate_joint_path(data["path"], "path")
    n = len(q)
    boundary_set = {0, n - 1}
    for key in ("lift_off_index", "set_down_index"):
        idx = _as_index(data, key, n)
        if idx is not None:
            boundary_set.add(idx)
    if "regrasp_indices" in data.files:
        for raw in np.asarray(data["regrasp_indices"], dtype=int).reshape(-1):
            idx = int(raw)
            if idx < 0 or idx >= n:
                raise ValueError(f"regrasp index {idx} is outside path index range [0, {n - 1}]")
            boundary_set.add(idx)
    boundaries = sorted(boundary_set)

    labels_arr = None
    if "labels" in data.files:
        labels_arr = np.asarray(data["labels"]).astype(str).reshape(-1)
        if len(labels_arr) != n:
            raise ValueError(f"labels length {len(labels_arr)} does not match path length {n}")

    segments = []
    labels = []
    recovered_spans = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 1:
            continue
        segment = _validate_joint_path(q[start:end + 1], f"path[{start}:{end + 1}]")
        midpoint = int((start + end) // 2)
        if labels_arr is None:
            label = f"segment_{len(segments)}"
        else:
            label = _normalise_grasped_label(labels_arr[midpoint])
        segments.append(segment)
        labels.append(label)
        recovered_spans.append((int(start), int(end)))

    if not segments:
        raise ValueError(f"{plan_path} did not yield any multi-sample grasped segments")
    for idx in range(len(segments) - 1):
        gap = float(np.linalg.norm(segments[idx][-1] - segments[idx + 1][0]))
        if gap > JOIN_TOL:
            raise ValueError(
                "recovered grasped segments are discontinuous: "
                f"segment {idx} end and segment {idx + 1} start differ by {gap:.3e}. "
                "Regenerate the grasped plan with continuous stitching."
            )
        segments[idx + 1][0] = segments[idx][-1]

    _warn_if_nonzero_grasp_tilt(data)
    print(
        "segments recovered from grasped stitched path: "
        + ", ".join(
            f"{label}[{start}:{end}]" for label, (start, end) in zip(labels, recovered_spans)
        )
    )
    return segments, labels, True


def _load_plan_segments(plan_path: Path):
    data = np.load(plan_path, allow_pickle=True)
    if _is_grasped_plan(data):
        return _load_grasped_plan_segments(data, plan_path)

    if "segments" in data.files:
        segments = [
            _validate_joint_path(segment, f"segments[{i}]")
            for i, segment in enumerate(np.asarray(data["segments"], dtype=object))
        ]
        if len(segments) != 3:
            raise ValueError(
                f"expected exactly three top-level segments, got {len(segments)}"
            )
        if "seg_labels" in data.files:
            labels = [str(label) for label in np.asarray(data["seg_labels"]).tolist()]
        else:
            labels = [f"segment_{i}" for i in range(len(segments))]
        if len(labels) != len(segments):
            raise ValueError(
                f"seg_labels length {len(labels)} does not match segments length {len(segments)}"
            )
        for idx in range(len(segments) - 1):
            gap = float(np.linalg.norm(segments[idx][-1] - segments[idx + 1][0]))
            if gap > JOIN_TOL:
                raise ValueError(
                    f"segments[{idx}] end and segments[{idx + 1}] start differ by {gap:.3e}"
                )
            segments[idx + 1][0] = segments[idx][-1]
        return segments, labels, True

    key = "dense" if "dense" in data.files else "path"
    if key not in data.files:
        raise KeyError(f"{plan_path} must contain key 'dense' or fallback key 'path'")
    print(
        f"NOTE: {plan_path} has no saved per-segment paths; "
        f"falling back to whole-path time parameterization from key '{key}'."
    )
    return [_validate_joint_path(data[key], key)], ["whole_path"], False


def _build_scene(plan_data=None):
    from scene import Scene
    import importlib

    if plan_data is not None and "payload_case" in plan_data.files:
        payload_case = str(np.asarray(plan_data["payload_case"]).item())
        os.environ["PAYLOAD_CASE"] = payload_case

    if "payload_model" in sys.modules:
        importlib.reload(sys.modules["payload_model"])
    if "run_panda" in sys.modules:
        run_panda = importlib.reload(sys.modules["run_panda"])
    else:
        import run_panda

    config = dict(run_panda.CFG)
    if plan_data is not None:
        if "support_contact_local" in plan_data.files:
            config["support_contact_local"] = np.asarray(
                plan_data["support_contact_local"], dtype=float
            ).reshape(3)
        if "support_normal_world" in plan_data.files:
            config["support_normal_world"] = np.asarray(
                plan_data["support_normal_world"], dtype=float
            ).reshape(3)
        if "support_force_max" in plan_data.files:
            config["support_force_max"] = float(np.asarray(plan_data["support_force_max"]).reshape(-1)[0])
        if "support_contact_model" in plan_data.files:
            config["support_contact_model"] = str(plan_data["support_contact_model"])
        for _k in ("support_mu", "support_mu_torsion", "support_base_radius", "support_derate"):
            if _k in plan_data.files:
                config[_k] = float(np.asarray(plan_data[_k]).reshape(-1)[0])
    # Let the contact headroom be retuned at export time (no re-plan needed): the
    # supported-slide load-sharing reserves this fraction for the dynamics so the
    # torque-limited timing has room to work.
    _sd_env = os.environ.get("PAYLOAD_SUPPORT_DERATE")
    if _sd_env is not None:
        config["support_derate"] = float(_sd_env)
    print(
        "export payload case: "
        f"{config.get('payload_case', os.environ.get('PAYLOAD_CASE', '<unknown>'))}"
    )
    return Scene(config)


def _time_parameterize(q_path: np.ndarray, vmax: np.ndarray, amax: np.ndarray):
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from primitive_manifold_planner.trajectory.time_parameterization import (
        time_parameterize_cubic_spline_path,
        time_parameterize_envelope_following_path,
    )

    trajectory = time_parameterize_envelope_following_path(
        q_path,
        max_joint_velocity=vmax,
        max_joint_acceleration=amax,
        dt=DT,
    )
    ok, reason = _trajectory_local_continuity_ok(trajectory, amax)
    if ok:
        return trajectory
    print(
        "WARNING: envelope-following segment timing failed local continuity "
        f"({reason}); using cubic-spline rest-to-rest timing for this segment."
    )
    return time_parameterize_cubic_spline_path(
        q_path,
        max_joint_velocity=vmax,
        max_joint_acceleration=amax,
        dt=DT,
    )


def _torque_stretch_factor(traj, scene, tau_fraction):
    """Smallest time-stretch s >= 1 that brings the segment's peak dynamic torque
    down to tau_fraction * (derate * tau_max) -- WHEN that is achievable.

    Stretching by s scales velocity by 1/s and acceleration by 1/s^2, so the
    inertial + Coriolis torque (M ddq + c) scales by 1/s^2 while the quasi-static
    part (gravity, contact-shared on support) is invariant.  As s -> infinity the
    segment peak approaches the static FLOOR = max |g_eff| / budget.  Therefore:

      * if that floor is already at/above the target, slowing can never bring the
        peak under it -> we do NOT stretch (the only cure is contact headroom,
        support_derate < 1), and flag static_limited so the caller can say so;
      * otherwise every sample has headroom below the target, so the required
        stretch is finite and we slow just enough to hit the target.

    Returns (s_required, static_limited).
    """
    robot = scene.robot
    if not hasattr(robot, "update_dynamics"):
        return 1.0, False
    q = np.asarray(traj.q, dtype=float)
    dq = np.asarray(traj.dq, dtype=float)
    ddq = np.asarray(traj.ddq, dtype=float)
    if len(q) == 0:
        return 1.0, False
    nq = q.shape[1]
    budget = np.asarray(scene.derate * scene.tau_max, dtype=float).reshape(-1)  # raw derated budget
    if budget.shape[0] < nq:
        return 1.0, False
    budget = budget[:nq]
    target = float(tau_fraction)

    support = None
    try:
        from constraints import ContactSupportFeasibility
        support = ContactSupportFeasibility.from_scene(scene)
    except Exception:
        support = None

    floor_util = 0.0
    s_req = 1.0
    any_over = False
    try:
        for k in range(len(q)):
            M, c, g = robot.update_dynamics(q[k], dq[k])
            g = np.asarray(g, dtype=float)
            # On a support surface the table shares the gravity load; that
            # contact-supported torque is the time-invariant part.
            if support is not None and _supported_surface(scene, q[k]) is not None:
                sol = support.solve(q[k])
                if sol["feasible"] and np.isfinite(sol["lambda"]):
                    g = np.asarray(sol["tau"], dtype=float)
            g_eff = g[:nq]                                            # invariant under time scaling
            dyn = (np.asarray(M, dtype=float) @ ddq[k]
                   + np.asarray(c, dtype=float))[:nq]                 # scales as 1/s^2
            for j in range(nq):
                Bj = float(budget[j])
                gj = float(g_eff[j])
                dj = float(dyn[j])
                fj = abs(gj) / Bj                                     # irreducible static floor
                if fj > floor_util:
                    floor_util = fj
                if abs(gj + dj) <= target * Bj + 1.0e-9:
                    continue                                         # within target at s = 1
                any_over = True
                if fj >= target - 1.0e-9 or abs(dj) < 1.0e-9:
                    continue                                         # irreducible here; floor gate handles it
                Tj = target * Bj
                x_hi = max((Tj - gj) / dj, (-Tj - gj) / dj)          # x = 1/s^2 upper edge
                if x_hi >= 1.0 - 1.0e-12 or x_hi <= 1.0e-12:
                    continue
                s_j = 1.0 / float(np.sqrt(x_hi))
                if s_j > s_req:
                    s_req = s_j
    except Exception as exc:
        print(f"torque-limited timing: inverse dynamics failed ({exc}); leaving segment unscaled.")
        return 1.0, False

    # Segment-level floor gate: a static floor already at/above the target cannot
    # be beaten by slowing, so do not stretch -- reserve contact headroom instead.
    if floor_util >= target - 1.0e-9:
        return 1.0, bool(any_over)
    return s_req, False


def _time_parameterize_torque_limited(q_path, vmax, amax, scene,
                                      tau_fraction=0.99, max_stretch=4.0, max_iters=3):
    """Mentor's time-parameterization, then stretch the segment's DURATION just
    enough that the full dynamic torque stays within the derated budget.

    The stretch re-runs the SAME parameterizer with scaled limits (vmax/s,
    amax/s^2): identical geometry, identical quintic, identical rest-to-rest
    (zero velocity / zero acceleration) boundaries -- only slower.  It is a no-op
    (returns the mentor's trajectory unchanged) when nothing is over budget.

    Returns (trajectory, total_stretch, static_limited).
    """
    vmax = np.asarray(vmax, dtype=float)
    amax = np.asarray(amax, dtype=float)
    traj = _time_parameterize(q_path, vmax, amax)
    s_total = 1.0
    static_limited = False
    for _ in range(int(max_iters)):
        s_step, seg_static_limited = _torque_stretch_factor(traj, scene, tau_fraction)
        static_limited = static_limited or seg_static_limited
        if s_step <= 1.0 + 1.0e-4:
            break
        s_total = min(float(max_stretch), s_total * float(s_step))
        traj = _time_parameterize(q_path, vmax / s_total, amax / (s_total * s_total))
        if s_total >= float(max_stretch) - 1.0e-9:
            break
    return traj, s_total, static_limited


def _geometric_path_class():
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from primitive_manifold_planner.trajectory.time_parameterization import GeometricPath

    return GeometricPath


def _clean_segment_waypoints(segment: np.ndarray, label: str, tol: float = 1.0e-3) -> np.ndarray:
    q = _validate_joint_path(segment, label)
    kept = [q[0].copy()]
    for idx in range(1, len(q) - 1):
        if float(np.linalg.norm(q[idx] - kept[-1])) >= float(tol):
            kept.append(q[idx].copy())
    kept.append(q[-1].copy())
    cleaned = np.asarray(kept, dtype=float)
    chords = np.linalg.norm(np.diff(cleaned, axis=0), axis=1) if len(cleaned) >= 2 else np.zeros(0)
    min_chord = float(np.min(chords)) if len(chords) else 0.0
    print(
        "waypoint_clean: "
        f"{label}: {len(q)}->{len(cleaned)} waypoints, "
        f"min consecutive chord after={min_chord:.6g} rad"
    )
    return cleaned


def _print_envelope_diagnostic(label: str, segment: np.ndarray, vmax: np.ndarray) -> None:
    GeometricPath = _geometric_path_class()
    geo = GeometricPath(np.asarray(segment, dtype=float))
    if float(geo.arc_length) <= 1.0e-12:
        print(
            "envelope_diag: "
            f"{label}: arc_length=0, min_sdot_max=inf, sigma=0, joint=none, "
            "implied_duration=0"
        )
        return
    n_samples = max(400, int(np.ceil(float(geo.arc_length) / 1.0e-3)) + 1)
    sigma = np.linspace(0.0, float(geo.arc_length), n_samples)
    dq_ds = np.asarray(geo.dq_dsigma(sigma), dtype=float)
    ratios = np.asarray(vmax, dtype=float).reshape(1, -1) / np.maximum(np.abs(dq_ds), 1.0e-9)
    sdot_max = np.min(ratios, axis=1)
    sample_idx = int(np.argmin(sdot_max))
    joint_idx = int(np.argmin(ratios[sample_idx]))
    min_sdot = float(sdot_max[sample_idx])
    duration = float(geo.arc_length) / max(min_sdot, 1.0e-12)
    print(
        "envelope_diag: "
        f"{label}: arc_length={float(geo.arc_length):.6g} rad, "
        f"min_sdot_max={min_sdot:.6g} at sigma={float(sigma[sample_idx]):.6g}, "
        f"dominating_joint={joint_idx + 1}, "
        f"implied_duration={duration:.6g}s"
    )


def _trajectory_local_continuity_ok(trajectory, amax: np.ndarray):
    dq = np.asarray(trajectory.dq, dtype=float)
    if len(dq) < 2:
        return False, "fewer than two samples"
    jump = np.abs(np.diff(dq, axis=0))
    max_jump = float(np.max(jump))
    jump_limit = float(1.5 * np.max(amax) * DT)
    if max_jump > jump_limit:
        flat_idx = int(np.argmax(jump))
        sample_idx, joint_idx = np.unravel_index(flat_idx, jump.shape)
        return (
            False,
            f"max velocity step {max_jump:.6g} > {jump_limit:.6g} "
            f"at segment samples {sample_idx}->{sample_idx + 1}, joint {joint_idx + 1}",
        )
    return True, "ok"


def _concatenate_segment_trajectories(segment_trajectories):
    q_parts = []
    dq_parts = []
    ddq_parts = []
    durations = []
    junction_indices = []
    for idx, trajectory in enumerate(segment_trajectories):
        q = np.asarray(trajectory.q, dtype=float)
        dq = np.asarray(trajectory.dq, dtype=float)
        ddq = np.asarray(trajectory.ddq, dtype=float)
        if not (q.shape == dq.shape == ddq.shape):
            raise ValueError(f"segment {idx} q/dq/ddq shapes do not match")
        start = 1 if idx > 0 else 0
        if idx > 0 and q_parts:
            previous_q = q_parts[-1][-1]
            boundary_gap = float(np.linalg.norm(previous_q - q[0]))
            if boundary_gap > JOIN_TOL:
                raise ValueError(
                    f"timed segment {idx - 1}/{idx} boundary gap {boundary_gap:.3e}"
                )
            q = q.copy()
            q[0] = previous_q
        q_parts.append(q[start:])
        dq_parts.append(dq[start:])
        ddq_parts.append(ddq[start:])
        durations.append(float(max(0, len(q) - 1) * DT))
        if idx < len(segment_trajectories) - 1:
            junction_indices.append(sum(len(part) for part in q_parts) - 1)
    q = np.vstack(q_parts)
    dq = np.vstack(dq_parts)
    ddq = np.vstack(ddq_parts)
    t = np.arange(len(q), dtype=float) * DT
    junction_indices = np.asarray(junction_indices, dtype=int)
    junction_times = t[junction_indices] if len(junction_indices) else np.zeros(0, dtype=float)
    return (
        SimpleNamespace(t=t, q=q, dq=dq, ddq=ddq),
        durations,
        junction_indices,
        junction_times,
    )


def _robot_velocity_limits(robot, dof: int) -> np.ndarray:
    model = getattr(robot, "pin_model", None)
    if model is not None and hasattr(model, "velocityLimit"):
        limits = np.asarray(model.velocityLimit, dtype=float).reshape(-1)
        if limits.shape[0] >= dof and np.all(np.isfinite(limits[:dof])) and np.all(limits[:dof] > 0.0):
            return limits[:dof].copy()
    return FRANKA_QDOT_MAX[:dof].copy()


def _write_csv(
    path: Path,
    t: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    tau: np.ndarray | None = None,
    ee_xyz: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        ["time_s"]
        + [f"q{i}" for i in range(1, q.shape[1] + 1)]
        + [f"dq{i}" for i in range(1, q.shape[1] + 1)]
        + [f"ddq{i}" for i in range(1, q.shape[1] + 1)]
    )
    tau_arr = None if tau is None else np.asarray(tau, dtype=float)
    ee_arr = None if ee_xyz is None else np.asarray(ee_xyz, dtype=float)
    if tau_arr is not None:
        if tau_arr.shape != q.shape:
            raise ValueError(f"tau shape {tau_arr.shape} does not match q shape {q.shape}")
        header += [f"torque{i}" for i in range(1, q.shape[1] + 1)]
    if ee_arr is not None:
        if ee_arr.shape != (len(t), 3):
            raise ValueError(f"ee_xyz shape {ee_arr.shape} does not match expected {(len(t), 3)}")
        header += ["ee_x", "ee_y", "ee_z"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k in range(len(t)):
            row = (
                [float(t[k])]
                + [float(x) for x in q[k]]
                + [float(x) for x in dq[k]]
                + [float(x) for x in ddq[k]]
            )
            if tau_arr is not None:
                row += [float(x) for x in tau_arr[k]]
            if ee_arr is not None:
                row += [float(x) for x in ee_arr[k]]
            writer.writerow(row)


def _write_viz_csv(path: Path, t: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["time_s"] + [f"q{i}" for i in range(1, q.shape[1] + 1)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for k in range(len(t)):
            writer.writerow([float(t[k])] + [float(x) for x in q[k]])


def _write_contact_diagnostics_csv(
    path: Path,
    t: np.ndarray,
    q: np.ndarray,
    scene,
    labels=None,
    junction_indices=None,
) -> None:
    from constraints import ContactSupportFeasibility, TorqueFeasibility

    path.parent.mkdir(parents=True, exist_ok=True)
    unsupported_torque = TorqueFeasibility(scene.robot, scene.tau_max, derate=scene.derate)
    support_torque = ContactSupportFeasibility.from_scene(scene)
    ranges = _segment_sample_ranges(
        len(q),
        np.zeros(0, dtype=int) if junction_indices is None else np.asarray(junction_indices, dtype=int),
    )
    sample_labels = ["whole_path"] * len(q)
    if labels is not None:
        labels = list(labels)
        if len(labels) == len(ranges):
            for label, (start, end) in zip(labels, ranges):
                for idx in range(int(start), int(end) + 1):
                    sample_labels[idx] = str(label)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample",
                "t",
                "label",
                "lambda",
                "max_supported_util",
                "max_unsupported_util",
                "contact_feasible",
            ]
        )
        for idx, (time_s, qk) in enumerate(zip(np.asarray(t, dtype=float), np.asarray(q, dtype=float))):
            unsupported_util = float(np.max(unsupported_torque.utilisation(qk)))
            surface = _supported_surface(scene, qk)
            if surface is None:
                writer.writerow(
                    [idx, float(time_s), sample_labels[idx], 0.0, "", unsupported_util, ""]
                )
                continue
            sol = support_torque.solve(qk)
            writer.writerow(
                [
                    idx,
                    float(time_s),
                    sample_labels[idx],
                    float(sol["lambda"]),
                    float(np.max(sol["utilisation"])),
                    unsupported_util,
                    bool(sol["feasible"]),
                ]
            )


def _viz_indices(n: int, dt: float) -> np.ndarray:
    stride = max(1, int(round(VIZ_DT / float(dt))))
    idx = np.arange(0, int(n), stride, dtype=int)
    if len(idx) == 0 or idx[-1] != n - 1:
        idx = np.concatenate([idx, np.asarray([n - 1], dtype=int)])
    return np.unique(idx)


def _peak_from_util(util: np.ndarray, offset: int = 0):
    flat_idx = int(np.argmax(util))
    sample_idx, joint_idx = np.unravel_index(flat_idx, util.shape)
    return {
        "peak": float(util[sample_idx, joint_idx]),
        "sample": int(sample_idx + offset),
        "joint": int(joint_idx),
    }


def _segment_sample_ranges(n_samples: int, junction_indices: np.ndarray):
    junctions = [int(idx) for idx in np.asarray(junction_indices, dtype=int)]
    if not junctions:
        return [(0, int(n_samples) - 1)]
    ranges = []
    start = 0
    for junction in junctions:
        ranges.append((start, junction))
        start = junction
    ranges.append((start, int(n_samples) - 1))
    return ranges


def _dynamic_torque_check(
    scene,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    segment_labels=None,
    junction_indices=None,
):
    robot = scene.robot
    if not hasattr(robot, "update_dynamics"):
        print("Dynamic torque check skipped: robot.update_dynamics is not exposed.")
        return None

    tau_limit = np.asarray(scene.derate * scene.tau_max, dtype=float).reshape(-1)
    if tau_limit.shape[0] < q.shape[1]:
        print("Dynamic torque check skipped: tau_max does not cover all joints.")
        return None
    tau_limit = tau_limit[:q.shape[1]]

    # Contact load-sharing.  On a support surface the table carries part of the
    # gravity load, so the honest arm torque is  M ddq + c + (g - J_c^T lambda n);
    # in free flight there is no contact and the raw inverse-dynamics torque stands.
    support = None
    try:
        from constraints import ContactSupportFeasibility
        support = ContactSupportFeasibility.from_scene(scene)
    except Exception as exc:
        print(f"Dynamic torque check: contact load-sharing disabled ({exc}); using unsupported torque.")

    util_rows = []
    tau_rows = []
    try:
        for k in range(len(q)):
            M, c, g = robot.update_dynamics(q[k], dq[k])
            g = np.asarray(g, dtype=float)
            # If the payload rests on a table here, share the gravity load with the
            # surface: replace g(q) by the contact-supported gravity torque g - a*lambda.
            if support is not None and _supported_surface(scene, q[k]) is not None:
                sol = support.solve(q[k])
                if sol["feasible"] and np.isfinite(sol["lambda"]):
                    g = np.asarray(sol["tau"], dtype=float)
            tau = np.asarray(M, dtype=float) @ ddq[k] + np.asarray(c, dtype=float) + g
            tau = np.asarray(tau[:q.shape[1]], dtype=float)
            util = np.abs(tau[:q.shape[1]]) / tau_limit
            tau_rows.append(tau)
            util_rows.append(util)
    except Exception as exc:
        print(f"Dynamic torque check skipped: inverse dynamics failed ({exc}).")
        return None

    util_arr = np.asarray(util_rows, dtype=float)
    tau_arr = np.asarray(tau_rows, dtype=float)
    global_peak = _peak_from_util(util_arr)
    labels = list(segment_labels or ["whole_path"])
    ranges = _segment_sample_ranges(
        len(q),
        np.zeros(0, dtype=int) if junction_indices is None else np.asarray(junction_indices, dtype=int),
    )
    if len(labels) != len(ranges):
        labels = [f"segment_{idx}" for idx in range(len(ranges))]
    segment_peaks = []
    for label, (start, end) in zip(labels, ranges):
        peak = _peak_from_util(util_arr[start:end + 1], offset=start)
        peak["label"] = str(label)
        peak["start"] = int(start)
        peak["end"] = int(end)
        segment_peaks.append(peak)

    flight_peak = next(
        (peak for peak in segment_peaks if "flight" in peak["label"].lower()),
        global_peak,
    )
    if flight_peak["peak"] > 1.0:
        print(
            "LOUD WARNING: free-flight dynamic torque utilisation exceeds 1.0. "
            "Lower SAFETY_FRACTION before running on hardware."
        )
    return {"global": global_peak, "segments": segment_peaks, "tau": tau_arr}


def _compute_ee_positions(scene, q: np.ndarray, frame: str = "fer_hand_tcp") -> np.ndarray | None:
    robot = scene.robot
    if not hasattr(robot, "frame_pose"):
        print("End-effector position export skipped: robot.frame_pose is not exposed.")
        return None
    try:
        return np.asarray([robot.frame_pose(frame, qk)[:3, 3] for qk in np.asarray(q, dtype=float)], dtype=float)
    except Exception as exc:
        print(f"End-effector position export skipped: FK failed for frame {frame!r} ({exc}).")
        return None


def _print_payload_centerline_crossing(scene, t: np.ndarray, q: np.ndarray) -> None:
    try:
        payload_xyz = np.asarray(
            [scene.robot.frame_pose("payload", qk)[:3, 3] for qk in np.asarray(q, dtype=float)],
            dtype=float,
        )
    except Exception as exc:
        print(f"payload centerline crossing: skipped (FK failed: {exc})")
        return
    if len(payload_xyz) == 0:
        print("payload centerline crossing: skipped (empty trajectory)")
        return
    idx = int(np.argmin(np.abs(payload_xyz[:, 1])))
    p = payload_xyz[idx]
    print(
        "payload centerline crossing: "
        f"sample={idx}, t={float(t[idx]):.3f}s, "
        f"xyz=[{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]"
    )


def _low_speed_clusters(mask: np.ndarray):
    clusters = []
    start = None
    for idx, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            clusters.append((start, idx - 1))
            start = None
    if start is not None:
        clusters.append((start, len(mask) - 1))
    return clusters


def _continuity_self_check(trajectory, amax: np.ndarray, junction_times: np.ndarray) -> None:
    t = np.asarray(trajectory.t, dtype=float)
    dq = np.asarray(trajectory.dq, dtype=float)
    if len(t) < 2:
        print("continuity_check: FAIL trajectory has fewer than two samples")
        return

    jump = np.abs(np.diff(dq, axis=0))
    flat_idx = int(np.argmax(jump))
    sample_idx, joint_idx = np.unravel_index(flat_idx, jump.shape)
    worst_jump = float(jump[sample_idx, joint_idx])
    jump_limit = float(1.5 * np.max(amax) * DT)
    jump_pass = bool(worst_jump <= jump_limit)
    print(
        "velocity_step_check: "
        f"{'PASS' if jump_pass else 'FAIL'} "
        f"max |dq[i+1]-dq[i]|={worst_jump:.6g} "
        f"limit={jump_limit:.6g} at samples {sample_idx}->{sample_idx + 1}, "
        f"joint {joint_idx + 1}, t={t[sample_idx]:.3f}->{t[sample_idx + 1]:.3f}s"
    )

    speed = np.linalg.norm(dq, axis=1)
    low_speed = speed < STOP_SPEED_THRESHOLD
    allowed_centers = np.concatenate(
        [
            np.asarray([float(t[0])], dtype=float),
            np.asarray(junction_times, dtype=float),
            np.asarray([float(t[-1])], dtype=float),
        ]
    )
    allowed = np.zeros_like(low_speed, dtype=bool)
    for center in allowed_centers:
        allowed |= np.abs(t - center) <= STOP_WINDOW_S
    outside = low_speed & ~allowed
    clusters = _low_speed_clusters(low_speed)
    interior_clusters = 0
    disallowed_clusters = []
    for start, end in clusters:
        center_time = float(0.5 * (t[start] + t[end]))
        near_endpoint = (
            abs(center_time - float(t[0])) <= STOP_WINDOW_S
            or abs(center_time - float(t[-1])) <= STOP_WINDOW_S
        )
        near_junction = np.any(np.abs(np.asarray(junction_times, dtype=float) - center_time) <= STOP_WINDOW_S)
        if near_junction and not near_endpoint:
            interior_clusters += 1
        if np.any(outside[start:end + 1]):
            disallowed_clusters.append((start, end))

    stutter_pass = bool(not np.any(outside) and interior_clusters == len(junction_times))
    if disallowed_clusters:
        offender_start, offender_end = disallowed_clusters[0]
        offender_slice = slice(offender_start, offender_end + 1)
        local = int(np.argmin(speed[offender_slice])) + offender_start
        offender = (
            f" first spurious stop cluster samples {offender_start}->{offender_end}, "
            f"min speed {speed[local]:.6g} at t={t[local]:.3f}s"
        )
    else:
        offender = (
            f" low-speed clusters={len(clusters)}, "
            f"interior clusters near junctions={interior_clusters}, "
            f"expected interior={len(junction_times)}"
        )
    print(
        "stutter_check: "
        f"{'PASS' if stutter_pass else 'FAIL'} "
        f"speed<{STOP_SPEED_THRESHOLD:.3f} rad/s only within +/-{STOP_WINDOW_S:.3f}s "
        f"of endpoints/junctions;{offender}"
    )


def _print_segment_summary(labels, durations, segment_trajectories, vmax, amax) -> None:
    print("segment_durations_s:")
    for label, duration, trajectory in zip(labels, durations, segment_trajectories):
        dq = np.asarray(trajectory.dq, dtype=float)
        ddq = np.asarray(trajectory.ddq, dtype=float)
        peak_vel_frac = float(np.max(np.abs(dq) / vmax.reshape(1, -1)))
        peak_acc_frac = float(np.max(np.abs(ddq) / amax.reshape(1, -1)))
        print(
            f"  {label}: {duration:.3f}s "
            f"(peak vel frac {peak_vel_frac:.4f}, peak acc frac {peak_acc_frac:.4f})"
        )


def _print_summary(trajectory, vmax: np.ndarray, amax: np.ndarray, torque_result) -> None:
    q = np.asarray(trajectory.q, dtype=float)
    dq = np.asarray(trajectory.dq, dtype=float)
    ddq = np.asarray(trajectory.ddq, dtype=float)
    vel_frac = np.max(np.abs(dq), axis=0) / vmax
    acc_frac = np.max(np.abs(ddq), axis=0) / amax
    viz_rows = len(_viz_indices(len(trajectory.t), DT))

    print(f"total_duration_s: {float(trajectory.t[-1] - trajectory.t[0]):.3f}")
    print(f"total_exec_rows: {len(trajectory.t)}")
    print(f"total_viz_rows: {viz_rows}")
    print(f"velocity_limits_used_rad_s: {np.array2string(vmax, precision=4)}")
    print(f"acceleration_limits_used_rad_s2: {np.array2string(amax, precision=4)}")
    print(f"peak_velocity_fraction_by_joint: {np.array2string(vel_frac, precision=4)}")
    print(f"peak_acceleration_fraction_by_joint: {np.array2string(acc_frac, precision=4)}")
    if torque_result is None:
        print("peak_torque_utilisation: skipped")
    else:
        global_peak = torque_result["global"]
        print(
            "peak_torque_utilisation: "
            f"{global_peak['peak']:.4f} at sample {global_peak['sample']} "
            f"joint {global_peak['joint'] + 1}"
        )
        print("peak_torque_utilisation_by_segment:")
        for peak in torque_result["segments"]:
            print(
                f"  {peak['label']}: {peak['peak']:.4f} "
                f"at joint {peak['joint'] + 1} sample {peak['sample']}"
            )


def _supported_surface(scene, q: np.ndarray):
    p = scene.robot.frame_pose("payload", q)[:3, 3]
    best = None
    best_dz = float("inf")
    for surface in getattr(scene, "surfaces", []):
        xy_bounds = getattr(surface, "xy_bounds", None)
        if xy_bounds is None:
            continue
        (xlo, ylo), (xhi, yhi) = np.asarray(xy_bounds, dtype=float)
        dz = abs(float(p[2]) - float(surface.height))
        if xlo <= p[0] <= xhi and ylo <= p[1] <= yhi and dz < 1.0e-2 and dz < best_dz:
            best = surface
            best_dz = dz
    return best


def _first_indices(indices, limit: int = 8) -> str:
    values = [int(i) for i in list(indices)[:limit]]
    suffix = "" if len(indices) <= limit else ", ..."
    return "[" + ", ".join(str(i) for i in values) + suffix + "]"


def _audit_geometric_path(scene, q: np.ndarray, label: str) -> bool:
    from constraints import (
        CollisionConstraint,
        ContactSupportFeasibility,
        FlatCarryConstraint,
        JointLimit,
        TorqueFeasibility,
    )

    q = _validate_joint_path(q, label)
    robot = scene.robot
    print("=" * 70)
    print(f"geometric audit: {label} ({len(q)} configs)")

    step = np.linalg.norm(np.diff(q, axis=0), axis=1)
    repetition_idx = np.flatnonzero(step < 1.0e-6)
    print(
        " repetitions: "
        f"count={len(repetition_idx)}, first={_first_indices(repetition_idx)}"
    )

    payload_xyz = np.asarray(
        [robot.frame_pose("payload", qk)[:3, 3] for qk in q],
        dtype=float,
    )
    backtracks = []
    for i in range(1, len(payload_xyz) - 1):
        prev_step = payload_xyz[i] - payload_xyz[i - 1]
        next_step = payload_xyz[i + 1] - payload_xyz[i]
        if (
            np.linalg.norm(prev_step) > 1.0e-4
            and np.linalg.norm(next_step) > 1.0e-4
            and float(np.dot(next_step, prev_step)) < 0.0
        ):
            backtracks.append(i)
    print(
        " backtracks: "
        f"count={len(backtracks)}, first={_first_indices(backtracks)}"
    )

    qlo, qhi = robot.joint_limits
    qlo = np.asarray(qlo, dtype=float)
    qhi = np.asarray(qhi, dtype=float)
    joint_limit = JointLimit(robot)
    flat = FlatCarryConstraint(robot, "payload", R_FLAT)
    collision = CollisionConstraint(
        robot,
        scene.collision_frames,
        scene.obstacles,
        frame_points=scene.collision_frame_points,
        self_collision_frames=scene.self_collision_frames,
        self_collision_min_distance=scene.self_collision_min_distance,
    )
    unsupported_torque = TorqueFeasibility(robot, scene.tau_max, derate=scene.derate)
    support_torque = ContactSupportFeasibility.from_scene(scene)

    offenders = {
        "joint_limits": [],
        "ee_down": [],
        "collision_free": [],
        "torque_unsupported": [],
        "support_infeasible_slide": [],
    }
    hit_details = []
    unsupported_free_util_max = 0.0
    supported_slide_util_max = 0.0
    supported_lambda_max = 0.0
    for idx, qk in enumerate(q):
        if not (
            np.all(qk >= qlo - 1.0e-9)
            and np.all(qk <= qhi + 1.0e-9)
            and joint_limit.value(qk) <= 1.0e-9
        ):
            offenders["joint_limits"].append(idx)
        if float(np.linalg.norm(flat.value(qk))) >= 1.0e-3:
            offenders["ee_down"].append(idx)
        hit = collision.hit_info(qk)
        if hit is not None:
            offenders["collision_free"].append(idx)
            if len(hit_details) < 3:
                hit_details.append((idx, hit))
        surface = _supported_surface(scene, qk)
        if surface is None:
            unsupported_util = float(np.max(unsupported_torque.utilisation(qk)))
            unsupported_free_util_max = max(unsupported_free_util_max, unsupported_util)
            if unsupported_util > 1.0 + 1.0e-9:
                offenders["torque_unsupported"].append(idx)
        else:
            sol = support_torque.solve(qk)
            supported_slide_util_max = max(
                supported_slide_util_max, float(np.max(sol["utilisation"]))
            )
            if np.isfinite(sol["lambda"]):
                supported_lambda_max = max(supported_lambda_max, float(sol["lambda"]))
            if not sol["feasible"]:
                offenders["support_infeasible_slide"].append(idx)

    valid_ok = True
    for name, idxs in offenders.items():
        ok = len(idxs) == 0
        valid_ok = valid_ok and ok
        print(
            f" {name}: {'PASS' if ok else 'FAIL'} "
            f"count={len(idxs)}, first={_first_indices(idxs)}"
        )
    for idx, hit in hit_details:
        print(f"   collision detail sample {idx}: {hit}")
    print(f" max unsupported free-flight utilisation: {unsupported_free_util_max:.3f}")
    print(f" max honest supported-slide utilisation: {supported_slide_util_max:.3f}")
    print(f" max lambda during supported slides: {supported_lambda_max:.6g}")
    print(
        " support-infeasible slide samples: "
        f"{len(offenders['support_infeasible_slide'])}"
    )
    print(
        " NOTE: slide-phase torque uses contact load-sharing: "
        "tau_arm = g(q) - J_contact(q)^T lambda n."
    )

    ok = bool(len(repetition_idx) == 0 and len(backtracks) == 0 and valid_ok)
    print(f" geometric audit result: {'PASS' if ok else 'FAIL'}")
    return ok


def _assert_time_grid_clean(t: np.ndarray, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> None:
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        raise ValueError("time grid has fewer than two samples")
    dt = np.diff(t)
    if not np.all(dt > 0.0):
        raise ValueError("time grid is not strictly increasing")
    if not np.allclose(dt, DT):
        raise ValueError(f"time grid is not uniformly {DT:g}s")
    for name, arr in [("q", q), ("dq", dq), ("ddq", ddq)]:
        if not np.all(np.isfinite(np.asarray(arr, dtype=float))):
            raise ValueError(f"{name} contains non-finite values")
    print("time grid: OK (1 ms uniform, monotonic)")


def _safe_normalize(values: np.ndarray, limit) -> np.ndarray:
    """values / limit per joint, with 0 where the limit is 0 / non-finite."""
    values = np.asarray(values, dtype=float)
    limit = np.asarray(limit, dtype=float).reshape(-1)
    out = np.zeros_like(values)
    for j in range(values.shape[1]):
        lj = float(limit[j]) if j < limit.size else 0.0
        if np.isfinite(lj) and abs(lj) > 1.0e-12:
            out[:, j] = values[:, j] / lj
    return out


def _plot_trajectory(t, q, dq, ddq, vmax, amax, tau, scene, outdir: Path) -> None:
    """Write traj_all.png.

    Position / velocity / acceleration panels show the raw values (with their
    limit lines), exactly as before.  Only the torque panel is normalised: each
    joint's torque is divided by the DERATED budget, so +/-1 == derate * tau_max
    -- the fraction of rated torque this run was allowed to use (0.40 by default).
    The +/-1 line follows whatever derate the run used.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        t = np.asarray(t, dtype=float)
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        ddq = np.asarray(ddq, dtype=float)
        vmax = np.asarray(vmax, dtype=float).reshape(-1)
        amax = np.asarray(amax, dtype=float).reshape(-1)
        nq = q.shape[1]
        joint_labels = [f"j{j + 1}" for j in range(nq)]

        derate = float(getattr(scene, "derate", 1.0))
        tau_budget = derate * np.asarray(scene.tau_max, dtype=float).reshape(-1)
        have_tau = tau is not None and np.size(tau) > 0
        taun = _safe_normalize(np.asarray(tau, dtype=float), tau_budget) if have_tau else None

        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

        # raw panels (unchanged from before)
        raw_panels = [
            ("Joint Position", "q [rad]", q, None),
            ("Joint Velocity", "dq [rad/s]", dq, vmax),
            ("Joint Acceleration", "ddq [rad/s^2]", ddq, amax),
        ]
        for ax, (title, ylabel, values, limits) in zip(axes[:3], raw_panels):
            for j, name in enumerate(joint_labels):
                ax.plot(t, values[:, j], linewidth=1.0, label=name)
                if limits is not None:
                    ax.axhline(float(limits[j]), color="black", linewidth=0.45, alpha=0.25)
                    ax.axhline(-float(limits[j]), color="black", linewidth=0.45, alpha=0.25)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
        axes[0].legend(ncol=nq, fontsize=8, loc="upper right")

        # torque panel: normalised to the derated budget (+/-1 = derate * tau_max)
        ax_tau = axes[3]
        if have_tau:
            for j, name in enumerate(joint_labels):
                ax_tau.plot(t, taun[:, j], linewidth=1.0, label=name)
        else:
            ax_tau.text(0.5, 0.5, "no torque data exported",
                        ha="center", va="center", transform=ax_tau.transAxes)
        ax_tau.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax_tau.axhline(-1.0, color="black", linestyle="--", linewidth=0.8)
        ax_tau.set_ylim(-1.2, 1.2)
        ax_tau.set_title(f"Normalized Joint Torques  (+/-1 = derate {derate:.2f} x tau_max)")
        ax_tau.set_ylabel("tau / (derate * tau_max)")
        ax_tau.set_xlabel("t [s]")
        ax_tau.grid(True, alpha=0.25)

        fig.tight_layout()
        out_path = outdir / "traj_all.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

        # velocity / acceleration exceedances in raw terms (as before)
        exceedances = []
        for kind, values, limits in [("velocity", dq, vmax), ("acceleration", ddq, amax)]:
            mask = np.abs(values) > limits.reshape(1, -1) + 1.0e-9
            sample_idx, joint_idx = np.nonzero(mask)
            for i, j in zip(sample_idx, joint_idx):
                exceedances.append(
                    (kind, int(j) + 1, float(t[i]), float(values[i, j]), float(limits[j]))
                )
        if exceedances:
            print("velocity/acceleration limit exceedances:")
            for kind, joint, time_s, value, limit in exceedances[:40]:
                print(f"  {kind} joint {joint} at t={time_s:.3f}s: "
                      f"value={value:+.6g}, limit={limit:.6g}")
            if len(exceedances) > 40:
                print(f"  ... {len(exceedances) - 40} more exceedance samples")
        else:
            print("no velocity/acceleration limit exceedances")

        # torque exceedance vs the derated budget (normalised > 1)
        if have_tau:
            mask = np.abs(taun) > 1.0 + 1.0e-6
            si, ji = np.nonzero(mask)
            if si.size == 0:
                print("no torque exceedances over the derated budget (normalised <= 1.0)")
            else:
                worst = {}
                for i, j in zip(si, ji):
                    v = abs(float(taun[i, j]))
                    if j not in worst or v > worst[j][1]:
                        worst[j] = (float(t[i]), v)
                print("torque over derated budget on "
                      + ", ".join(f"j{j + 1} (peak {val:.4f} @ t={tt:.3f}s)"
                                  for j, (tt, val) in sorted(worst.items())))
        print(f"wrote trajectory plot: {out_path}")
    except Exception as exc:
        print(f"trajectory plotting skipped: {exc}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--exec-csv", type=Path, default=DEFAULT_EXEC_CSV)
    parser.add_argument("--viz-csv", type=Path, default=DEFAULT_VIZ_CSV)
    parser.add_argument("--contact-diag-csv", type=Path, default=DEFAULT_CONTACT_DIAG_CSV)
    parser.add_argument("--safety-fraction", type=float, default=SAFETY_FRACTION)
    args = parser.parse_args(argv)

    segments, labels, used_segments = _load_plan_segments(args.plan)
    with np.load(args.plan, allow_pickle=True) as plan_data:
        scene = _build_scene(plan_data)
        dof = segments[0].shape[1]
        if dof != int(scene.robot.nq):
            raise ValueError(f"path has {dof} joints but scene robot has nq={scene.robot.nq}")
        for idx, segment in enumerate(segments):
            if segment.shape[1] != dof:
                raise ValueError(f"segments[{idx}] has dof {segment.shape[1]}, expected {dof}")
        for label, segment in zip(labels, segments):
            _audit_geometric_path(scene, segment, f"raw segment {label}")
        if "dense" in plan_data.files:
            dense_plan = _validate_joint_path(plan_data["dense"], "dense")
            _audit_geometric_path(scene, dense_plan, "npz dense")

    safety = float(args.safety_fraction)
    if not (0.0 < safety <= 1.0):
        raise ValueError("--safety-fraction must be in (0, 1]")
    vmax_nominal = _robot_velocity_limits(scene.robot, dof)
    amax_nominal = FRANKA_QDDOT_MAX[:dof].copy()
    vmax = safety * vmax_nominal
    amax = safety * amax_nominal

    cleaned_segments = []
    for label, segment in zip(labels, segments):
        cleaned = _clean_segment_waypoints(segment, label)
        _print_envelope_diagnostic(label, cleaned, vmax)
        cleaned_segments.append(cleaned)
    segments = cleaned_segments

    if os.environ.get("PAYLOAD_TORQUE_LIMITED_TIMING", "1") == "1":
        _tau_frac = float(os.environ.get("PAYLOAD_TORQUE_TIMING_FRACTION", "0.99"))
        _max_stretch = float(os.environ.get("PAYLOAD_TORQUE_TIMING_MAX_STRETCH", "4.0"))
        segment_trajectories = []
        for _seg_label, _segment in zip(labels, segments):
            _seg_traj, _seg_s, _static_limited = _time_parameterize_torque_limited(
                _segment, vmax, amax, scene,
                tau_fraction=_tau_frac, max_stretch=_max_stretch,
            )
            if _seg_s > 1.0 + 1.0e-3:
                print(f"torque-limited timing: segment '{_seg_label}' stretched "
                      f"x{_seg_s:.3f} so the dynamic torque fits within "
                      f"{_tau_frac:.3f} of the derated budget")
            if _static_limited:
                print(f"torque-limited timing: segment '{_seg_label}' has samples whose "
                      f"quasi-static supported torque is already at the budget; slowing "
                      f"cannot reduce those -- reserve contact headroom "
                      f"(PAYLOAD_SUPPORT_DERATE < 1) to bring them under.")
            segment_trajectories.append(_seg_traj)
    else:
        segment_trajectories = [
            _time_parameterize(segment, vmax, amax)
            for segment in segments
        ]
    trajectory, durations, junction_indices, junction_times = _concatenate_segment_trajectories(
        segment_trajectories
    )
    t = np.asarray(trajectory.t, dtype=float)
    q = np.asarray(trajectory.q, dtype=float)
    dq = np.asarray(trajectory.dq, dtype=float)
    ddq = np.asarray(trajectory.ddq, dtype=float)
    _assert_time_grid_clean(t, q, dq, ddq)
    _print_payload_centerline_crossing(scene, t, q)

    torque_result = _dynamic_torque_check(
        scene,
        q,
        dq,
        ddq,
        segment_labels=labels,
        junction_indices=junction_indices,
    )
    tau = None if torque_result is None else np.asarray(torque_result.get("tau"), dtype=float)
    ee_xyz = _compute_ee_positions(scene, q)

    if tau is not None:
        print(
            "NOTE: exported tau1..tau7 are feed-forward inverse-dynamics torques "
            "M(q)ddq + c(q,dq) + g(q); on support surfaces g(q) is replaced by the "
            "contact-shared gravity torque g(q) - J_contact(q)^T lambda n (honest slide torque)."
        )
    _write_csv(args.exec_csv, t, q, dq, ddq, tau=tau, ee_xyz=ee_xyz)
    viz_idx = _viz_indices(len(t), DT)
    _write_viz_csv(args.viz_csv, t[viz_idx], q[viz_idx], dq[viz_idx], ddq[viz_idx])
    _write_contact_diagnostics_csv(
        args.contact_diag_csv,
        t,
        q,
        scene,
        labels=labels,
        junction_indices=junction_indices,
    )
    _plot_trajectory(t, q, dq, ddq, vmax, amax, tau, scene, HERE)

    total_geometric_points = int(sum(len(segment) for segment in segments))
    mode = "per-segment" if used_segments else "whole-path fallback"
    print(
        f"loaded path: {args.plan} "
        f"({len(segments)} {mode} path(s), {total_geometric_points} geometric points)"
    )
    print(f"wrote exec csv: {args.exec_csv}")
    print(f"wrote viz csv: {args.viz_csv} (stride {max(1, int(round(VIZ_DT / DT)))})")
    print(f"wrote contact diagnostics csv: {args.contact_diag_csv}")
    print(
        "NOTE: slide-phase torque uses contact load-sharing: "
        "tau_arm = g(q) - J_contact(q)^T lambda n."
    )
    print(
        "FRANKA_QDDOT_MAX is a Panda datasheet constant; confirm against your "
        "specific robot variant before hardware execution."
    )
    _print_segment_summary(labels, durations, segment_trajectories, vmax, amax)
    if len(junction_times):
        print(
            "junction_times_s: "
            + ", ".join(
                f"{time:.3f} (sample {idx})"
                for time, idx in zip(junction_times, junction_indices)
            )
        )
    _continuity_self_check(trajectory, amax, junction_times)
    _print_summary(trajectory, vmax, amax, torque_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
