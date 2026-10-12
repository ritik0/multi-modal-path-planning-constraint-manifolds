"""Plot payload trajectory position, velocity, and acceleration limits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from export_trajectory import SAFETY_FRACTION, FRANKA_QDOT_MAX, FRANKA_QDDOT_MAX


HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "traj_exec.csv"
DEFAULT_PNG = HERE / "trajectory_plot.png"


def _load_traj_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no trajectory rows")

    t = np.asarray([float(row["t"]) for row in rows], dtype=float)
    q = np.asarray(
        [[float(row[f"q{j}"]) for j in range(1, 8)] for row in rows],
        dtype=float,
    )
    dq = np.asarray(
        [[float(row[f"dq{j}"]) for j in range(1, 8)] for row in rows],
        dtype=float,
    )
    ddq = np.asarray(
        [[float(row[f"ddq{j}"]) for j in range(1, 8)] for row in rows],
        dtype=float,
    )
    return t, q, dq, ddq


def _peak_report(name: str, t: np.ndarray, values: np.ndarray, limits: np.ndarray):
    fractions = np.abs(values) / limits.reshape(1, -1)
    flat_idx = int(np.argmax(fractions))
    sample_idx, joint_idx = np.unravel_index(flat_idx, fractions.shape)
    peak_value = float(values[sample_idx, joint_idx])
    peak_fraction = float(fractions[sample_idx, joint_idx])
    limit = float(limits[joint_idx])
    peak_time = float(t[sample_idx])
    print(
        f"{name}: joint {joint_idx + 1} peaks at t={peak_time:.3f}s, "
        f"value={peak_value:.6g}, enforced_limit=+/-{limit:.6g}, "
        f"fraction={peak_fraction:.4f}"
    )
    return {
        "name": name,
        "sample_idx": sample_idx,
        "joint_idx": joint_idx,
        "time": peak_time,
        "value": peak_value,
        "limit": limit,
        "fraction": peak_fraction,
    }


def _binding_report(vel_peak: dict, acc_peak: dict) -> None:
    binding = vel_peak if vel_peak["fraction"] >= acc_peak["fraction"] else acc_peak
    print(
        "binding constraint: "
        f"{binding['name']} joint {binding['joint_idx'] + 1} at "
        f"t={binding['time']:.3f}s "
        f"({binding['fraction']:.4f} of enforced limit)"
    )


def _plot(t, q, dq, ddq, qdot_limits, qddot_limits, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [f"q{j}" for j in range(1, 8)]
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, constrained_layout=True)

    for j, label in enumerate(labels):
        color = colors[j % len(colors)] if colors else None
        axes[0].plot(t, q[:, j], linewidth=1.1, color=color, label=label)
        axes[1].plot(t, dq[:, j], linewidth=1.1, color=color, label=label)
        axes[2].plot(t, ddq[:, j], linewidth=1.1, color=color, label=label)
        axes[1].axhline(qdot_limits[j], linestyle="--", linewidth=0.8, color=color, alpha=0.55)
        axes[1].axhline(-qdot_limits[j], linestyle="--", linewidth=0.8, color=color, alpha=0.55)
        axes[2].axhline(qddot_limits[j], linestyle="--", linewidth=0.8, color=color, alpha=0.55)
        axes[2].axhline(-qddot_limits[j], linestyle="--", linewidth=0.8, color=color, alpha=0.55)

    axes[0].set_ylabel("position [rad]")
    axes[1].set_ylabel("velocity [rad/s]")
    axes[2].set_ylabel("acceleration [rad/s^2]")
    axes[2].set_xlabel("time [s]")
    axes[0].set_title("Joint trajectory")
    axes[1].set_title("Joint velocity with enforced limits")
    axes[2].set_title("Joint acceleration with enforced limits")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=7, fontsize=9, loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_PNG)
    args = parser.parse_args(argv)

    t, q, dq, ddq = _load_traj_csv(args.csv)
    qdot_limits = SAFETY_FRACTION * FRANKA_QDOT_MAX
    qddot_limits = SAFETY_FRACTION * FRANKA_QDDOT_MAX

    vel_peak = _peak_report("velocity", t, dq, qdot_limits)
    acc_peak = _peak_report("acceleration", t, ddq, qddot_limits)
    _binding_report(vel_peak, acc_peak)

    _plot(t, q, dq, ddq, qdot_limits, qddot_limits, args.out)
    print(f"saved plot: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
