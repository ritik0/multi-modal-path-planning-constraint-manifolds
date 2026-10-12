"""Sweep compact payload mass and save per-mass plans plus diagnostics.

This is a batch driver only: it rebuilds the Pinocchio model from one generated
URDF per mass, reuses the existing multimodal planner, and writes results for
later inspection.  It does not animate anything.
"""

from __future__ import annotations

import csv
import argparse
import os
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from scene import Scene
import multimodal as mm
from ompl_planner import plan_in_mode
from run_panda import (TABLE_A, TABLE_B, FRAME, COLLISION_FRAMES, SUPPORT_MARGIN,
                       PLANNER_NAME, SOLVE_TIME, LOCK_SLIDE_YAW,
                       SLIDE_YAW_BAND, _seed_candidates)

best_utilisation = mm.best_utilisation


HERE = Path(__file__).resolve().parent
BASE_URDF = HERE / "fer_franka_hand_payload.urdf"
OUT_DIR = HERE / "sweep_results"
URDF_DIR = OUT_DIR / "urdf"
SUMMARY_CSV = OUT_DIR / "summary.csv"
SUMMARY_TXT = OUT_DIR / "summary.txt"

TORQUE_DERATE = 0.85
START = mm.Placement("A", 0.63, 0.34)
PREFERRED_GOAL = mm.Placement("B", 0.63, -0.34)
FALLBACK_GOAL = mm.Placement("B", 0.58, -0.30)
GOAL = PREFERRED_GOAL
MASSES = np.arange(4.0, 12.0 + 1.0e-9, 0.5)

CSV_FIELDS = [
    "mass",
    "best_posture_boundary_x",
    "start_best_util",
    "success",
    "attempt_lift_margin",
    "message",
    "liftoff_x",
    "liftoff_y",
    "liftoff_margin",
    "setdown_margin",
    "d0",
    "d1",
    "d2",
    "backward_flag",
    "slideA_cart_len",
    "flight_cart_len",
    "slideB_cart_len",
    "total_cart_len",
    "max_flight_z",
    "min_flight_z",
    "min_flight_margin",
    "n_full_configs",
    "plan_time_s",
]


def _mass_tag(mass: float) -> str:
    return f"{float(mass):04.1f}"


def _write_mass_urdf(mass: float) -> Path:
    tree = ET.parse(BASE_URDF)
    root = tree.getroot()
    payload = None
    for link in root.findall("link"):
        if link.attrib.get("name") == "payload":
            payload = link
            break
    if payload is None:
        raise RuntimeError(f"No link named 'payload' in {BASE_URDF}")
    mass_elem = payload.find("./inertial/mass")
    if mass_elem is None:
        raise RuntimeError(f"No payload inertial mass element in {BASE_URDF}")
    mass_elem.set("value", f"{float(mass):.6g}")

    URDF_DIR.mkdir(parents=True, exist_ok=True)
    out = URDF_DIR / f"payload_m{_mass_tag(mass)}.urdf"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def _scene_config(urdf: Path) -> dict:
    return {
        "urdf": str(urdf),
        "controlled_frame": FRAME,
        "torque_derate": TORQUE_DERATE,
        "collision_frames": COLLISION_FRAMES,
        "obstacles": [],
        "surfaces": [TABLE_A, TABLE_B],
    }


def _placement_reachable(scene: Scene, placement: mm.Placement,
                         seeds: list[np.ndarray]) -> bool:
    for seed_idx, seed in enumerate(seeds):
        q, ok = mm.place_config(
            scene,
            placement,
            seed,
            rng=np.random.default_rng(12000 + seed_idx),
        )
        if ok and q is not None:
            return True
    return False


def _resolve_goal(scene: Scene, seeds: list[np.ndarray]) -> mm.Placement:
    if _placement_reachable(scene, PREFERRED_GOAL, seeds):
        print(
            "using GOAL: "
            f"({PREFERRED_GOAL.surface}, {PREFERRED_GOAL.x:.3f}, {PREFERRED_GOAL.y:.3f})"
        )
        return PREFERRED_GOAL
    print(
        "preferred GOAL projection failed; falling back to "
        f"({FALLBACK_GOAL.surface}, {FALLBACK_GOAL.x:.3f}, {FALLBACK_GOAL.y:.3f})"
    )
    return FALLBACK_GOAL


def _plan_in_mode_compat(robot, mode, q_start, q_goal, **kw):
    return plan_in_mode(robot, mode, q_start, q_goal, **kw)


def _cartesian_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float).reshape((-1, 3))
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _empty_plan_arrays() -> dict:
    return {
        "path": np.empty((0, 7), dtype=float),
        "dense": np.empty((0, 7), dtype=float),
        "ee_xyz": np.empty((0, 3), dtype=float),
        "lift_off": np.full(7, np.nan, dtype=float),
        "set_down": np.full(7, np.nan, dtype=float),
        "seg_lengths": np.empty((0,), dtype=int),
        "segments": np.empty((0,), dtype="<U16"),
    }


def _concat_drop_shared(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.empty((0, 7), dtype=float)
    out = [chunks[0]]
    for chunk in chunks[1:]:
        out.append(chunk[1:] if len(chunk) else chunk)
    return np.vstack(out)


def _ee_path(scene: Scene, path: np.ndarray) -> np.ndarray:
    arr = np.asarray(path, dtype=float).reshape((-1, scene.robot.nq))
    return np.asarray(
        [scene.robot.frame_pose(FRAME, q)[:3, 3] for q in arr],
        dtype=float,
    )


def _transition_distances(scene: Scene, res) -> tuple[float, float, float, bool]:
    p_start = scene.robot.frame_pose(FRAME, res.q_start)[:3, 3]
    p_lift = scene.robot.frame_pose(FRAME, res.lift_off)[:3, 3]
    p_set = scene.robot.frame_pose(FRAME, res.set_down)[:3, 3]
    p_goal = scene.robot.frame_pose(FRAME, res.q_goal)[:3, 3]
    d0 = float(np.linalg.norm(p_start - p_goal))
    d1 = float(np.linalg.norm(p_lift - p_goal))
    d2 = float(np.linalg.norm(p_set - p_goal))
    return d0, d1, d2, bool(d1 > d0)


def _diagnose_boundary(scene: Scene, seed: np.ndarray, mass: float) -> tuple[str, float | None]:
    # TODO: keep this scenario-specific scan local unless a shared helper can be
    # introduced without changing the exact sampled x-grid or RNG sequence.
    rng = np.random.default_rng(31000 + int(round(float(mass) * 10.0)))
    boundary = None
    start_best = None
    carry_seed = np.asarray(seed, dtype=float)
    scan_xs = np.unique(np.r_[np.linspace(0.37, START.x, 14), START.x])
    for x in scan_xs:
        util, q_best = best_utilisation(
            scene,
            float(x),
            START.y,
            scene.surface(START.surface).height,
            n_restarts=24,
            seed=carry_seed,
            rng=rng,
        )
        if q_best is not None:
            carry_seed = q_best
        if np.isclose(x, START.x):
            start_best = util
        if util is not None and boundary is None and util > 1.0:
            boundary = float(x)
    return ("none" if boundary is None else f"{boundary:.3f}"), start_best


def _run_plan_once(scene: Scene, seeds: list[np.ndarray], mass: float, lift_margin: float,
                   max_pairs: int = 10):
    last = None
    for seed_idx, seed in enumerate(seeds):
        try:
            res = mm.plan(
                scene,
                START,
                GOAL,
                planner=_plan_in_mode_compat,
                seed_q=seed,
                n_transition_tries=200,
                perturb=0.3,
                max_pairs=max_pairs,
                support_margin=SUPPORT_MARGIN,
                rng=np.random.default_rng(9000 + int(round(mass * 10.0)) + seed_idx),
                space="atlas",
                delta=0.4,
                lift_margin=lift_margin,
                planner_name=PLANNER_NAME,
                solve_time=SOLVE_TIME,
                interpolate_points=300,
                lock_slide_yaw=LOCK_SLIDE_YAW,
                slide_yaw_band=SLIDE_YAW_BAND,
                verbose=False,
            )
        except Exception as exc:
            last = None
            message = f"exception in seed {seed_idx}: {exc}"
            continue
        last = res
        if res.success:
            return res
    if last is not None:
        return last
    return mm.MultiModalResult(False, message if "message" in locals() else "all seeds raised")


def _run_plan(scene: Scene, seeds: list[np.ndarray], mass: float, lift_margin: float = 0.0):
    res = _run_plan_once(scene, seeds, mass, lift_margin=lift_margin, max_pairs=10)
    if res.success:
        return res, lift_margin
    if lift_margin <= 0.0:
        return res, lift_margin
    res_retry = _run_plan_once(scene, seeds, mass, lift_margin=0.0, max_pairs=10)
    if res_retry.success:
        res_retry.message = f"{res_retry.message} (retry with lift_margin=0.0)"
        return res_retry, 0.0
    res_retry.message = (
        f"{res_retry.message} (retry with lift_margin=0.0 after: {res.message})"
    )
    return res_retry, 0.0


def _summarize_success(scene: Scene, res) -> tuple[dict, dict]:
    full = np.asarray(res.full_path(), dtype=float)
    segment_paths = [np.asarray(path, dtype=float) for _, path in res.segments]
    dense = mm.uniform_resample_result(
        scene,
        res,
        n_points=300,
        support_margin=SUPPORT_MARGIN,
    )
    ee_xyz = _ee_path(scene, dense)
    ee_segments = [_ee_path(scene, seg) for seg in segment_paths]

    lengths = {label: _cartesian_length(ee)
               for (label, _), ee in zip(res.segments, ee_segments)}
    slide_a = float(lengths.get("slide_A", 0.0))
    flight = float(lengths.get("free_flight", 0.0))
    slide_b = float(lengths.get("slide_B", 0.0))
    total = _cartesian_length(ee_xyz)
    flight_ee = ee_segments[1] if len(ee_segments) >= 2 else np.empty((0, 3))

    torque = mm.build_modes(scene)["torque"]
    flight_path = next((path for label, path in res.segments
                        if label == "free_flight"), np.empty((0, scene.robot.nq)))
    flight_margins = [float(torque.margin(q)) for q in flight_path]
    lift_pos = scene.robot.frame_pose(FRAME, res.lift_off)[:3, 3]
    d0, d1, d2, backward_flag = _transition_distances(scene, res)
    npz_data = {
        "path": full,
        "dense": dense,
        "ee_xyz": ee_xyz,
        "lift_off": np.asarray(res.lift_off, dtype=float),
        "set_down": np.asarray(res.set_down, dtype=float),
        "seg_lengths": np.asarray([len(path) for _, path in res.segments], dtype=int),
        "segments": np.asarray([label for label, _ in res.segments]),
    }
    metrics = {
        "liftoff_x": float(lift_pos[0]),
        "liftoff_y": float(lift_pos[1]),
        "liftoff_margin": float(torque.margin(res.lift_off)),
        "setdown_margin": float(torque.margin(res.set_down)),
        "d0": d0,
        "d1": d1,
        "d2": d2,
        "backward_flag": backward_flag,
        "slideA_cart_len": slide_a,
        "flight_cart_len": flight,
        "slideB_cart_len": slide_b,
        "total_cart_len": total,
        "max_flight_z": float(np.max(flight_ee[:, 2])) if len(flight_ee) else float("nan"),
        "min_flight_z": float(np.min(flight_ee[:, 2])) if len(flight_ee) else float("nan"),
        "min_flight_margin": float(np.min(flight_margins)) if flight_margins else float("nan"),
        "n_full_configs": int(len(full)),
    }
    return npz_data, metrics


def _segment_report(scene: Scene, res) -> str:
    result_by_label = {
        label: seg for label, seg in getattr(res, "segment_results", [])
    }
    parts = []
    for label, path in res.segments:
        seg = result_by_label.get(label)
        planner = getattr(seg, "planner", "unknown") if seg is not None else "unknown"
        ee = _ee_path(scene, np.asarray(path, dtype=float))
        parts.append(f"{label}:{planner}:{_cartesian_length(ee):.3f}m")
    return ", ".join(parts)


def _save_npz(path: Path, scene: Scene, urdf: Path, mass: float, success: bool,
              arrays: dict, message: str, metrics: dict | None = None) -> None:
    metrics = metrics or {}
    np.savez(
        path,
        path=arrays["path"],
        dense=arrays["dense"],
        ee_xyz=arrays["ee_xyz"],
        lift_off=arrays["lift_off"],
        set_down=arrays["set_down"],
        seg_lengths=arrays["seg_lengths"],
        segments=arrays["segments"],
        tableA=np.asarray(TABLE_A["xy_bounds"], dtype=float),
        hA=np.asarray(float(TABLE_A["height"])),
        tableB=np.asarray(TABLE_B["xy_bounds"], dtype=float),
        hB=np.asarray(float(TABLE_B["height"])),
        frame=np.asarray(FRAME),
        urdf=np.asarray(str(urdf)),
        mass=np.asarray(float(mass)),
        success=np.asarray(bool(success)),
        message=np.asarray(str(message)),
        metrics_names=np.asarray(list(metrics.keys())),
        metrics_values=np.asarray([float(v) for v in metrics.values()], dtype=float),
    )


def _blank_metrics() -> dict:
    return {
        "liftoff_x": float("nan"),
        "liftoff_y": float("nan"),
        "liftoff_margin": float("nan"),
        "setdown_margin": float("nan"),
        "d0": float("nan"),
        "d1": float("nan"),
        "d2": float("nan"),
        "backward_flag": float("nan"),
        "slideA_cart_len": float("nan"),
        "flight_cart_len": float("nan"),
        "slideB_cart_len": float("nan"),
        "total_cart_len": float("nan"),
        "max_flight_z": float("nan"),
        "min_flight_z": float("nan"),
        "min_flight_margin": float("nan"),
        "n_full_configs": 0,
    }


def _fmt(value) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return f"{value:.3f}"
    return str(value)


def _write_summary_txt(rows: list[dict], summary_txt: Path = SUMMARY_TXT) -> None:
    widths = {
        "mass": 5,
        "best_posture_boundary_x": 9,
        "start_best_util": 10,
        "success": 7,
        "liftoff_x": 8,
        "d0": 7,
        "d1": 7,
        "d2": 7,
        "backward_flag": 8,
        "slideA_cart_len": 8,
        "slideB_cart_len": 8,
        "flight_cart_len": 8,
        "max_flight_z": 8,
        "min_flight_margin": 8,
        "plan_time_s": 8,
    }
    cols = list(widths)
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("Payload mass sweep summary\n")
        f.write(f"START=({START.surface}, {START.x:.3f}, {START.y:.3f})  ")
        f.write(f"GOAL=({GOAL.surface}, {GOAL.x:.3f}, {GOAL.y:.3f})\n\n")
        f.write(" ".join(col[:widths[col]].ljust(widths[col]) for col in cols) + "\n")
        f.write(" ".join("-" * widths[col] for col in cols) + "\n")
        for row in rows:
            f.write(" ".join(_fmt(row[col])[:widths[col]].ljust(widths[col])
                             for col in cols) + "\n")


def _coarse_masses() -> np.ndarray:
    return np.asarray([4.0, 6.0, 8.0, 10.0, 12.0], dtype=float)


def run_sweep(masses: np.ndarray, out_dir: Path) -> list[dict]:
    urdf_dir = out_dir / "urdf"
    summary_csv = out_dir / "summary.csv"
    summary_txt = out_dir / "summary.txt"
    global OUT_DIR, URDF_DIR, GOAL
    OUT_DIR = out_dir
    URDF_DIR = urdf_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    urdf_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for mass in masses:
            tic = time.perf_counter()
            tag = _mass_tag(float(mass))
            plan_path = OUT_DIR / f"plan_m{tag}.npz"
            row = {field: "" for field in CSV_FIELDS}
            row["mass"] = f"{float(mass):.1f}"
            segment_report = ""
            try:
                urdf = _write_mass_urdf(float(mass))
                scene = Scene(_scene_config(urdf))
                seeds = _seed_candidates(scene)
                if len(rows) == 0:
                    GOAL = _resolve_goal(scene, seeds)
                boundary, start_best = _diagnose_boundary(scene, seeds[0], float(mass))
                row["best_posture_boundary_x"] = boundary
                row["start_best_util"] = start_best if start_best is not None else float("nan")

                plan_tic = time.perf_counter()
                res, attempt_lift_margin = _run_plan(scene, seeds, float(mass))
                plan_time = time.perf_counter() - plan_tic
                row["plan_time_s"] = plan_time
                row["success"] = bool(res.success)
                row["attempt_lift_margin"] = attempt_lift_margin
                row["message"] = res.message
                if res.success:
                    arrays, metrics = _summarize_success(scene, res)
                    segment_report = _segment_report(scene, res)
                else:
                    arrays, metrics = _empty_plan_arrays(), _blank_metrics()
                for key, value in metrics.items():
                    row[key] = value
                _save_npz(plan_path, scene, urdf, float(mass), bool(res.success),
                          arrays, res.message, metrics)
            except Exception as exc:
                row["best_posture_boundary_x"] = row.get("best_posture_boundary_x") or "error"
                row["start_best_util"] = row.get("start_best_util") or float("nan")
                row["success"] = False
                row["attempt_lift_margin"] = "error"
                row["message"] = f"{exc}\n{traceback.format_exc(limit=3)}"
                row["plan_time_s"] = time.perf_counter() - tic
                for key, value in _blank_metrics().items():
                    row[key] = value
                arrays = _empty_plan_arrays()
                urdf = URDF_DIR / f"payload_m{tag}.urdf"
                _save_npz(plan_path, None, urdf, float(mass), False, arrays, row["message"],
                          _blank_metrics())

            writer.writerow(row)
            csv_file.flush()
            rows.append(row.copy())
            print(
                f"m={float(mass):4.1f} kg  "
                f"boundary={row['best_posture_boundary_x']}  "
                f"success={row['success']}  "
                f"slideA={_fmt(row['slideA_cart_len'])} m  "
                f"slideB={_fmt(row['slideB_cart_len'])} m  "
                f"max_flight_z={_fmt(row['max_flight_z'])} m  "
                f"time={_fmt(row['plan_time_s'])} s"
                + (f"  segments=[{segment_report}]" if segment_report else "")
            )
            _write_summary_txt(rows, summary_txt)

    print(f"summary CSV -> {summary_csv}")
    print(f"summary TXT -> {summary_txt}")
    print(f"plans       -> {out_dir}")
    return rows


def _print_validation(rows: list[dict]) -> bool:
    utils = [float(row["start_best_util"]) for row in rows]
    monotonic = all(b >= a - 1.0e-9 for a, b in zip(utils, utils[1:]))
    light_success = any(bool(row["success"]) for row in rows if float(row["mass"]) <= 6.0)
    print("validation summary:")
    print(f"  start_best monotonic: {monotonic}")
    print(f"  at least one light mass succeeded: {light_success}")
    for row in rows:
        print(
            f"  m={float(row['mass']):4.1f}  "
            f"start_best={float(row['start_best_util']):.3f}  "
            f"success={row['success']}  boundary={row['best_posture_boundary_x']}"
        )
    return monotonic and light_success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", action="store_true",
                        help="Run only the validation masses 4,6,8,10,12.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory override.")
    parser.add_argument("--planner-name", default=PLANNER_NAME,
                        help="OMPL planner name, e.g. RRTConnect or BITstar.")
    parser.add_argument("--solve-time", type=float, default=SOLVE_TIME,
                        help="Per-segment OMPL solve/optimization time.")
    parser.add_argument("--no-lock-slide-yaw", action="store_true",
                        help="Disable endpoint-orientation yaw locks on slide modes.")
    args = parser.parse_args()

    globals()["PLANNER_NAME"] = str(args.planner_name)
    globals()["SOLVE_TIME"] = float(args.solve_time)
    globals()["LOCK_SLIDE_YAW"] = not bool(args.no_lock_slide_yaw)
    print(
        f"sweep planner={PLANNER_NAME}, solve_time={SOLVE_TIME:.1f}s, "
        f"lock_slide_yaw={LOCK_SLIDE_YAW}, slide_yaw_band={SLIDE_YAW_BAND:.3g}"
    )

    masses = _coarse_masses() if args.coarse else MASSES
    out_dir = Path(args.out_dir) if args.out_dir else (
        HERE / "sweep_results_coarse" if args.coarse else HERE / "sweep_results"
    )
    rows = run_sweep(masses, out_dir)
    if args.coarse:
        _print_validation(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
