"""Surface-exploiting / torque-aware transport experiment.

Drop this file into ``payload_planner/`` and run it from that folder.

Purpose
-------
The thesis planner uses two choices that are deliberately changed here:

1. On a support surface the thesis oracle uses the *minimum* normal force
   required for feasibility.  This experiment instead chooses, at each
   supported configuration, the normal force lambda that MINIMISES THE WORST
   NORMALISED JOINT TORQUE:

       lambda*(q) = argmin_lambda max_j |g_j(q)-lambda*a_j(q)| / B_j

   where a(q)=J_c(q)^T n and B_j=rho*tau_max,j.

2. The thesis selects the shortest complete successful route.  This experiment
   selects successful routes lexicographically by

       (peak worst-joint utilisation,
        free-flight joint-space length,
        mean worst-joint utilisation,
        total joint-space length)

   so actuator loading is the first priority and unsupported carrying is the
   second priority.  This tends to favour "slide longer, lift later, set down
   earlier" when that actually reduces robot effort.

The constrained planning machinery itself is unchanged: transition sampling,
Atlas/Projected state spaces, RRT-Connect, collision checking, and geometric
certification are reused from the thesis stack.

This is a MODEL-BASED / QUASI-STATIC route-selection experiment.  It does not
replace the post-timing full inverse-dynamics check in export_trajectory.py.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import run_panda as rp
import transport
from constraints import ContactSupportFeasibility, TorqueFeasibility
from scene import Scene


HERE = Path(__file__).resolve().parent
PLAN_OUT = HERE / os.environ.get("EFFORT_PLAN_OUT", "panda_plan_effort.npz")
CSV_OUT = HERE / os.environ.get("EFFORT_CSV_OUT", "panda_plan_effort_metrics.csv")

# Number of complete successful routes to compare before selecting a winner.
# A larger value gives the torque-aware objective more alternatives to choose
# from, at the cost of planning time.
EFFORT_BEST_OF_N = int(os.environ.get("EFFORT_BEST_OF_N", "30"))
EFFORT_BUDGET = int(os.environ.get("EFFORT_BUDGET", "120"))

# Joint-space spacing used when evaluating route utilisation.
EFFORT_SAMPLE_STEP = float(os.environ.get("EFFORT_SAMPLE_STEP", "0.02"))


# ---------------------------------------------------------------------------
# 1) Alternative support allocation: minimise the worst utilised robot joint.
# ---------------------------------------------------------------------------

def minimax_support(scene: Scene, support: ContactSupportFeasibility, q: np.ndarray) -> dict:
    """Return the scalar normal force that minimises peak joint utilisation.

    Solves the linear program

        minimise t
        s.t. |g_j - lambda a_j| <= t B_j  for every joint j
             lambda >= 0
             lambda <= support.max_lambda (if finite)

    Variables are x=[lambda, t].  Because lambda is scalar and every torque is
    affine in lambda, this is a small convex LP.
    """
    if support.contact_model != "normal":
        raise RuntimeError(
            "This experiment is written for the thesis' scalar normal-force "
            f"model; current support_contact_model={support.contact_model!r}."
        )

    q = np.asarray(q, dtype=float)
    g = np.asarray(scene.robot.gravity(q), dtype=float).reshape(-1)
    a = np.asarray(support.support_direction_torque(q), dtype=float).reshape(-1)

    # TRUE derated actuator budget, i.e. rho * nominal limits.
    B = np.asarray(support.tau_max, dtype=float).reshape(-1)
    if not (g.shape == a.shape == B.shape):
        return {
            "feasible": False,
            "lambda": np.nan,
            "tau": g,
            "utilisation": np.full_like(g, np.inf),
            "reason": "shape_mismatch",
        }

    # g_j - a_j*lambda <= B_j*t
    # -g_j + a_j*lambda <= B_j*t
    # -> [-a_j, -B_j] [lambda,t]^T <= -g_j
    # -> [ a_j, -B_j] [lambda,t]^T <=  g_j
    A_ub = []
    b_ub = []
    for gj, aj, Bj in zip(g, a, B):
        A_ub.append([-aj, -Bj])
        b_ub.append(-gj)
        A_ub.append([+aj, -Bj])
        b_ub.append(+gj)

    lam_hi = None if not np.isfinite(support.max_lambda) else float(support.max_lambda)
    result = linprog(
        c=np.array([0.0, 1.0]),
        A_ub=np.asarray(A_ub, dtype=float),
        b_ub=np.asarray(b_ub, dtype=float),
        bounds=[(0.0, lam_hi), (0.0, None)],
        method="highs",
    )

    if not result.success:
        return {
            "feasible": False,
            "lambda": np.nan,
            "tau": g,
            "utilisation": np.full_like(g, np.inf),
            "reason": f"linprog_failed:{result.message}",
        }

    lam = float(result.x[0])
    tau = g - lam * a
    util = np.abs(tau) / B
    return {
        "feasible": bool(np.all(np.isfinite(util))),
        "lambda": lam,
        "tau": tau,
        "utilisation": util,
        "peak": float(np.max(util)),
        "reason": "ok",
    }


# ---------------------------------------------------------------------------
# 2) Score a complete route using the new support policy.
# ---------------------------------------------------------------------------

def score_route(scene: Scene, segments, support: ContactSupportFeasibility) -> dict:
    torque = TorqueFeasibility(scene.robot, scene.tau_max, derate=scene.derate)

    all_state_peaks = []
    support_lambdas = []
    worst = {
        "u": -np.inf,
        "joint": -1,
        "segment": "",
        "q": None,
        "lambda": 0.0,
    }

    free_length = 0.0
    total_length = 0.0

    rows = []
    global_index = 0

    for label, path in segments:
        path = np.asarray(path, dtype=float)
        total_length += transport._joint_length(path)
        if label == "free_flight":
            free_length += transport._joint_length(path)

        dense = transport._linear_densify_joint_path(path, max_step=EFFORT_SAMPLE_STEP)
        for q in dense:
            if label.startswith("slide"):
                sol = minimax_support(scene, support, q)
                if not sol["feasible"]:
                    return {
                        "valid": False,
                        "reason": sol["reason"],
                        "score": (np.inf, np.inf, np.inf, np.inf),
                    }
                util = np.asarray(sol["utilisation"], dtype=float)
                lam = float(sol["lambda"])
                support_lambdas.append(lam)
            else:
                util = np.asarray(torque.utilisation(q), dtype=float)
                lam = 0.0

            peak = float(np.max(util))
            j = int(np.argmax(util))
            all_state_peaks.append(peak)

            if peak > worst["u"]:
                worst = {
                    "u": peak,
                    "joint": j,
                    "segment": label,
                    "q": np.asarray(q, dtype=float).copy(),
                    "lambda": lam,
                }

            row = {
                "sample": global_index,
                "segment": label,
                "lambda_opt_N": lam,
                "peak_util": peak,
                "worst_joint": j + 1,
            }
            for k, uk in enumerate(util, start=1):
                row[f"u{k}"] = float(uk)
            rows.append(row)
            global_index += 1

    mean_peak = float(np.mean(all_state_peaks)) if all_state_peaks else np.inf
    mean_lambda = float(np.mean(support_lambdas)) if support_lambdas else 0.0
    max_lambda = float(np.max(support_lambdas)) if support_lambdas else 0.0

    # Lexicographic optimisation:
    #   1) minimise worst utilised joint anywhere on route
    #   2) minimise unsupported/free-flight joint-space motion
    #   3) minimise average worst-joint loading
    #   4) minimise total joint-space motion
    score = (float(worst["u"]), float(free_length), mean_peak, float(total_length))

    return {
        "valid": True,
        "score": score,
        "peak_u": float(worst["u"]),
        "mean_peak_u": mean_peak,
        "free_length": float(free_length),
        "total_length": float(total_length),
        "mean_lambda": mean_lambda,
        "max_lambda": max_lambda,
        "worst_joint": int(worst["joint"] + 1),
        "worst_segment": str(worst["segment"]),
        "worst_q": worst["q"],
        "worst_lambda": float(worst["lambda"]),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 3) Transport search copied from the thesis structure, but final selection is
#    by score_route() instead of total joint length.
# ---------------------------------------------------------------------------

def plan_surface_effort(
    scene,
    start,
    goal,
    planner="RRTConnect",
    solve_time=10.0,
    delta=0.1,
    n_tries=300,
    best_of_n=EFFORT_BEST_OF_N,
    budget=EFFORT_BUDGET,
    support_margin=None,
    constrained_space="atlas",
):
    r = scene.robot
    constrained_space = str(constrained_space).strip().lower()
    if constrained_space not in {"projected", "atlas"}:
        return transport.TransportResult(False, f"unknown constrained space {constrained_space!r}"), None

    if support_margin is None:
        support_margin = float(os.environ.get("PAYLOAD_SUPPORT_MARGIN", str(rp.SUPPORT_MARGIN)))

    rng = np.random.default_rng(int(os.environ.get("PAYLOAD_SEED", "20260612")))
    surface_a = scene.surface(start.surface)
    surface_b = scene.surface(goal.surface)
    mode_a = transport.on_surface_mode(scene, surface_a)
    mode_b = transport.on_surface_mode(scene, surface_b)
    mode_f = transport.free_flight_mode(scene)
    torque = TorqueFeasibility(r, scene.tau_max, derate=scene.derate)
    support = transport.support_torque(scene)

    if support.contact_model != "normal":
        raise RuntimeError(
            "Set PAYLOAD_SUPPORT_CONTACT_MODEL=normal for this experiment."
        )

    # Same start/goal placement logic as the thesis planner.
    start_lock_yaw = bool(scene.config.get("start_lock_yaw", False))
    start_yaw_deg = float(scene.config.get("start_yaw_deg", 0.0))
    R_start = transport.R_FLAT
    if start_yaw_deg != 0.0:
        a = np.radians(start_yaw_deg)
        Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                       [np.sin(a),  np.cos(a), 0.0],
                       [0.0,        0.0,       1.0]])
        R_start = Rz @ transport.R_FLAT

    q_start, ok_start = None, False
    if start_lock_yaw:
        q_start, ok_start = transport._place_feasible(
            scene, surface_a, start.x, start.y, mode_a, rng,
            lock_yaw=True, target_rotation=R_start,
        )
    if not ok_start or q_start is None:
        q_start, ok_start = transport._place_feasible(
            scene, surface_a, start.x, start.y, mode_a, rng
        )
    if not ok_start or q_start is None:
        return transport.TransportResult(False, "start placement infeasible"), None

    goal_lock_yaw = bool(scene.config.get("goal_lock_yaw", False))
    q_goal, ok_goal = None, False
    if goal_lock_yaw:
        q_goal, ok_goal = transport._place_feasible(
            scene, surface_b, goal.x, goal.y, mode_b, rng,
            lock_yaw=True, target_rotation=R_start,
        )
    if not ok_goal or q_goal is None:
        q_goal, ok_goal = transport._place_feasible(
            scene, surface_b, goal.x, goal.y, mode_b, rng
        )
    if not ok_goal or q_goal is None:
        return transport.TransportResult(False, "goal placement infeasible"), None

    lifts = transport.sample_transitions(
        scene, surface_a, n_tries,
        other_surface=surface_b, rng=rng, support_margin=support_margin,
    )
    sets = transport.sample_transitions(
        scene, surface_b, n_tries,
        other_surface=surface_a, rng=rng, support_margin=support_margin,
    )

    print(f"transition candidates: lift={len(lifts)}, set={len(sets)}")
    if not lifts or not sets:
        return transport.TransportResult(
            False, "no torque-feasible transitions",
            q_start=q_start, q_goal=q_goal,
            n_lift=len(lifts), n_set=len(sets),
        ), None

    # Keep the original cheap pair-ordering heuristic.  This is only the order
    # in which expensive planning calls are tried; it is NOT the final cost.
    pairs = []
    for i, q_lift in enumerate(lifts):
        for j, q_set in enumerate(sets):
            pairs.append((float(np.linalg.norm(q_lift - q_set)), i, j))
    pairs.sort(key=lambda item: item[0])

    accepted = []
    pairs_tried = 0

    for cheap_distance, i, j in pairs[: max(0, int(budget))]:
        pairs_tried += 1
        q_lift = lifts[i]
        q_set = sets[j]

        seg_a = transport._plan_segment(
            r, mode_a, q_start, q_lift,
            planner, solve_time, delta, constrained_space,
        )
        if not seg_a.success:
            continue

        if transport._foliation_enabled():
            seg_f = transport._plan_free_flight_foliated(
                scene, q_lift, q_set,
                planner, solve_time, delta, constrained_space,
            )
        else:
            seg_f = transport._plan_segment(
                r, mode_f, q_lift, q_set,
                planner, solve_time, delta, constrained_space,
            )
        if not seg_f.success:
            continue

        seg_b = transport._plan_segment(
            r, mode_b, q_set, q_goal,
            planner, solve_time, delta, constrained_space,
        )
        if not seg_b.success:
            continue

        segments = [
            ("slide_A", transport._drop_close_waypoints(seg_a.path)),
            ("free_flight", transport._drop_close_waypoints(seg_f.path)),
            ("slide_B", transport._drop_close_waypoints(seg_b.path)),
        ]

        metrics = score_route(scene, segments, support)
        if not metrics["valid"]:
            continue

        print(
            f"accepted {len(accepted)+1:02d}: pair_d={cheap_distance:.3f}, "
            f"peak_u={metrics['peak_u']:.4f}, "
            f"free_L={metrics['free_length']:.3f}, "
            f"mean_u={metrics['mean_peak_u']:.4f}, "
            f"total_L={metrics['total_length']:.3f}, "
            f"lambda_mean/max={metrics['mean_lambda']:.2f}/{metrics['max_lambda']:.2f} N"
        )

        accepted.append({
            "segments": segments,
            "q_lift": q_lift.copy(),
            "q_set": q_set.copy(),
            "seg_results": [seg_a, seg_f, seg_b],
            "metrics": metrics,
        })

        if len(accepted) >= max(1, int(best_of_n)):
            break

    if not accepted:
        return transport.TransportResult(
            False,
            f"no feasible transport triple among {pairs_tried} candidate pairs",
            q_start=q_start, q_goal=q_goal,
            n_lift=len(lifts), n_set=len(sets), pairs_tried=pairs_tried,
        ), None

    winner = min(accepted, key=lambda item: item["metrics"]["score"])
    segments = winner["segments"]
    metrics = winner["metrics"]
    q_lift = winner["q_lift"]
    q_set = winner["q_set"]

    raw_full = np.vstack([path for _, path in segments])
    dense_path = transport._linear_densify_joint_path(raw_full, max_step=0.01)

    print("=" * 70)
    print("SURFACE-EFFORT WINNER")
    print(f"  score tuple            : {metrics['score']}")
    print(f"  peak utilisation       : {metrics['peak_u']:.4f}")
    print(f"  worst joint            : J{metrics['worst_joint']}")
    print(f"  worst segment          : {metrics['worst_segment']}")
    print(f"  free-flight joint len  : {metrics['free_length']:.4f} rad")
    print(f"  total joint len        : {metrics['total_length']:.4f} rad")
    print(f"  mean peak utilisation  : {metrics['mean_peak_u']:.4f}")
    print(f"  support lambda mean    : {metrics['mean_lambda']:.3f} N")
    print(f"  support lambda max     : {metrics['max_lambda']:.3f} N")
    print(f"  lift free-hold util    : {float(np.max(torque.utilisation(q_lift))):.4f}")
    print(f"  set free-hold util     : {float(np.max(torque.utilisation(q_set))):.4f}")

    result = transport.TransportResult(
        True,
        f"surface-effort transport selected from {len(accepted)} successful routes",
        segments=segments,
        lift_off=q_lift,
        set_down=q_set,
        q_start=q_start,
        q_goal=q_goal,
        n_lift=len(lifts),
        n_set=len(sets),
        pairs_tried=pairs_tried,
        segment_results=list(zip(["slide_A", "free_flight", "slide_B"], winner["seg_results"])),
        dense_path=dense_path,
    )
    return result, metrics


# ---------------------------------------------------------------------------
# 4) Save in the same general NPZ format expected by animate_pyvista.py and
#    export_trajectory.py, plus the new optimisation metrics.
# ---------------------------------------------------------------------------

def save_outputs(scene: Scene, result: transport.TransportResult, metrics: dict) -> None:
    full = np.asarray(result.full_path(), dtype=float)
    dense = transport.uniform_resample_result(
        scene,
        result,
        n_points=300,
        support_margin=rp.SUPPORT_MARGIN,
        yaw_band=rp.SLIDE_YAW_BAND,
    )
    ee_xyz = np.asarray(
        [scene.robot.frame_pose(rp.FRAME, q)[:3, 3] for q in dense],
        dtype=float,
    )

    np.savez(
        PLAN_OUT,
        path=full,
        dense=dense,
        ee_xyz=ee_xyz,
        payload_case=np.asarray(rp.PAYLOAD_CASE),
        payload_mass=np.asarray(rp.TOTAL_MASS, dtype=float),
        payload_com=np.asarray(rp.COM, dtype=float),
        payload_inertia=np.asarray(rp.INERTIA, dtype=float),
        lift_off=np.asarray(result.lift_off, dtype=float),
        set_down=np.asarray(result.set_down, dtype=float),
        segments=np.array([p for _, p in result.segments], dtype=object),
        seg_labels=np.array([lbl for lbl, _ in result.segments]),
        seg_lengths=np.asarray([len(p) for _, p in result.segments], dtype=int),
        tableA=np.asarray(rp.TABLE_A["xy_bounds"], dtype=float),
        hA=np.asarray(float(rp.TABLE_A["height"])),
        tableB=np.asarray(rp.TABLE_B["xy_bounds"], dtype=float),
        hB=np.asarray(float(rp.TABLE_B["height"])),
        table_collision_margin=np.asarray(rp.TABLE_COLLISION_MARGIN, dtype=float),
        support_contact_local=np.asarray(rp.PAYLOAD_CONTACT_LOCAL, dtype=float),
        support_normal_world=np.asarray(rp.SUPPORT_NORMAL_WORLD, dtype=float),
        support_force_max=np.asarray(rp.SUPPORT_FORCE_MAX, dtype=float),
        support_contact_model=np.asarray(rp.SUPPORT_CONTACT_MODEL),
        support_mu=np.asarray(rp.SUPPORT_MU, dtype=float),
        support_mu_torsion=np.asarray(rp.SUPPORT_MU_TORSION, dtype=float),
        support_base_radius=np.asarray(rp.SUPPORT_BASE_RADIUS, dtype=float),
        support_derate=np.asarray(rp.SUPPORT_DERATE, dtype=float),
        gap_wall_enabled=np.asarray(bool(rp.WALL_ENABLED)),
        obstacle_center=np.asarray(rp._wall_T[:3, 3], dtype=float),
        obstacle_half_extents=np.asarray(
            [(rp._wx_hi-rp._wx_lo)/2.0, rp._wy_thick/2.0, rp.WALL_TOP/2.0],
            dtype=float,
        ),
        frame=np.asarray(rp.FRAME),
        urdf=np.asarray(rp.URDF),
        objective=np.asarray("lexicographic_peak_u_free_length_mean_u_total_length"),
        effort_peak_u=np.asarray(metrics["peak_u"], dtype=float),
        effort_mean_peak_u=np.asarray(metrics["mean_peak_u"], dtype=float),
        effort_free_length=np.asarray(metrics["free_length"], dtype=float),
        effort_total_length=np.asarray(metrics["total_length"], dtype=float),
        effort_worst_joint=np.asarray(metrics["worst_joint"], dtype=int),
        effort_worst_segment=np.asarray(metrics["worst_segment"]),
        effort_mean_lambda=np.asarray(metrics["mean_lambda"], dtype=float),
        effort_max_lambda=np.asarray(metrics["max_lambda"], dtype=float),
    )
    print(f"saved plan -> {PLAN_OUT}")

    fieldnames = [
        "sample", "segment", "lambda_opt_N", "peak_util", "worst_joint",
        "u1", "u2", "u3", "u4", "u5", "u6", "u7",
    ]
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics["rows"])
    print(f"saved per-sample effort metrics -> {CSV_OUT}")


# ---------------------------------------------------------------------------
# 5) Main: reuse the same scene/query discovery as run_panda.py.
# ---------------------------------------------------------------------------

def main() -> int:
    np.set_printoptions(precision=4, suppress=True)
    scene = Scene(rp.CFG)

    print("=" * 70)
    print("Surface-exploiting torque-aware transport experiment")
    print(f"payload case      : {rp.PAYLOAD_CASE}")
    print(f"payload mass      : {rp.TOTAL_MASS:.3f} kg")
    print(f"torque derate rho : {scene.derate:.3f}")
    print(f"planner           : {rp.PAYLOAD_PLANNER_NAME}")
    print(f"constrained space : {rp.PAYLOAD_CONSTRAINED_SPACE}")
    print(f"successful routes : up to {EFFORT_BEST_OF_N}")
    print(f"pair budget       : {EFFORT_BUDGET}")
    print("objective          : min peak_u -> min free length -> min mean_u -> min total length")
    print("support policy     : choose lambda that minimises worst joint utilisation")

    seeds = rp._seed_candidates(scene)
    rp._collision_self_check(scene, seeds)

    start, _ = rp._discover_start(scene, seeds)
    if start is None:
        return 1
    goal, _, goal_seed = rp._discover_goal(scene, seeds)
    if goal is None:
        return 1
    if goal_seed is not None and goal_seed != 0:
        seeds = [seeds[goal_seed], *seeds[:goal_seed], *seeds[goal_seed + 1:]]

    print(f"start: {start.surface} ({start.x:.3f}, {start.y:.3f})")
    print(f"goal : {goal.surface} ({goal.x:.3f}, {goal.y:.3f})")

    result, metrics = plan_surface_effort(
        scene,
        start,
        goal,
        planner=rp.PAYLOAD_PLANNER_NAME,
        solve_time=rp.PAYLOAD_SOLVE_TIME,
        delta=rp.PAYLOAD_DELTA,
        n_tries=300,
        best_of_n=EFFORT_BEST_OF_N,
        budget=EFFORT_BUDGET,
        support_margin=rp.SUPPORT_MARGIN,
        constrained_space=rp.PAYLOAD_CONSTRAINED_SPACE,
    )

    print("=" * 70)
    print(result.message)
    if not result.success or metrics is None:
        return 1

    save_outputs(scene, result, metrics)
    print()
    print("Visualise this plan with:")
    print(f"  python -B animate_pyvista.py --plan {PLAN_OUT.name} --once")
    print()
    print("NOTE: export_trajectory.py currently uses the thesis minimum-support")
    print("oracle during its supported dynamic check.  Therefore its dynamic")
    print("support allocation is not yet the same minimax-lambda policy used here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())