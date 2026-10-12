"""Overlay two EXPORTED trajectories (thesis vs effort) for a clean comparison.

Reads two full-trajectory CSVs written by the exporter, with columns:
    time_s, q1..q7, dq1..dq7, ddq1..ddq7, torque1..torque7, ee_x, ee_y, ee_z

Utilisation is computed from the ACTUAL exported joint torques:
    u_j(t) = |torque_j(t)| / (rho * tau_max_j),   u_max(t) = max_j u_j(t)
so this reflects the true DYNAMIC load (inertial + gravity + contact) the
exporter validated -- not a quasi-static approximation. On the slide phases the
support force is already baked into the exported torques, so u_max there reflects
the real supported load.

Phases are inferred from end-effector height: near table height => slide,
clearly raised => free carry. (The full CSV carries no phase column, so we
segment on ee_z; override the threshold with --lift-eps if needed.)

Usage (from payload_planner/):
    python -B compare_traj.py \
        --thesis traj_4p1_thesis.csv \
        --effort traj_4p1_effort.csv \
        --derate 0.5945 \
        --out compare_4p1_traj.png
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAU_MAX = np.array([87., 87., 87., 87., 12., 12., 12.])
TORQUE_COLS = [f"torque{i}" for i in range(1, 8)]


def load(path, derate):
    t, umax, ee_z = [], [], []
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        have = rdr.fieldnames or []
        missing = [c for c in (["time_s", "ee_z"] + TORQUE_COLS) if c not in have]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}\n  header was: {have}")
        budget = derate * TAU_MAX
        for r in rdr:
            tau = np.array([float(r[c]) for c in TORQUE_COLS])
            u = np.abs(tau) / budget
            t.append(float(r["time_s"]))
            umax.append(float(np.max(u)))
            ee_z.append(float(r["ee_z"]))
    return np.array(t), np.array(umax), np.array(ee_z)


def phase_spans_from_z(t, z, lift_eps):
    """Segment into slide / free / slide by whether ee_z is above table+eps."""
    base = np.min(z)
    raised = z > (base + lift_eps)
    spans, start = [], 0
    for i in range(1, len(raised)):
        if raised[i] != raised[start]:
            spans.append((t[start], t[i], bool(raised[start]))); start = i
    spans.append((t[start], t[-1], bool(raised[start])))
    return spans


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thesis", required=True)
    ap.add_argument("--effort", required=True)
    ap.add_argument("--derate", type=float, default=0.5945,
                    help="rho used to normalise torque into utilisation")
    ap.add_argument("--lift-eps", type=float, default=0.02,
                    help="ee_z rise (m) above table that counts as 'in free carry'")
    ap.add_argument("--out", default="compare_traj.png")
    args = ap.parse_args()

    data = {}
    for tag, p in [("thesis", args.thesis), ("effort", args.effort)]:
        if not Path(p).exists():
            print(f"  MISSING {p}"); continue
        data[tag] = load(p, args.derate)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax2 = ax.twinx()
    ucol = {"thesis": "#0b3d5c", "effort": "#c04a2b"}
    zcol = {"thesis": "#8fb8cc", "effort": "#e0a08c"}

    ref = data.get("thesis") or data.get("effort")
    if ref is not None:
        tt, _, zz = ref
        for (ts, te, raised) in phase_spans_from_z(tt, zz, args.lift_eps):
            ax.axvspan(ts, te, color=("#ffffff" if raised else "#dfe7ee"),
                       alpha=0.5, zorder=0)

    summ = {}
    for tag in ("thesis", "effort"):
        if tag not in data:
            continue
        t, u, z = data[tag]
        ax.plot(t, u, color=ucol[tag], lw=1.9, label=f"{tag}: $u_{{max}}$", zorder=3)
        ax2.plot(t, z, color=zcol[tag], lw=1.4, ls="--",
                 label=f"{tag}: ee height", zorder=2)
        summ[tag] = (float(np.max(u)), float(np.mean(u)), float(t[-1]),
                     int(np.argmax(u)))

    ax.axhline(1.0, color="#e8890c", ls="--", lw=1.2, zorder=1)
    ax.text(0.01, 1.008, "budget $u=1$", transform=ax.get_yaxis_transform(),
            color="#e8890c", fontsize=9, va="bottom")
    ax.set_xlabel("time $t$ (s)")
    ax.set_ylabel(r"max torque utilisation $u_{max}$")
    ax2.set_ylabel("end-effector height (m)")
    top = max((s[0] for s in summ.values()), default=1.1)
    ax.set_ylim(0, max(1.1, top * 1.05))
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.2, lw=0.5)
    ax.set_title(f"Thesis planner vs. torque-aware effort selector "
                 f"(rho={args.derate:g})", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out, dpi=170)
    print(f"wrote {args.out}")

    print("\n=== summary (true dynamic supported utilisation) ===")
    for tag, (pk, mn, dur, ai) in summ.items():
        over = " OVER budget" if pk > 1.0 else ""
        print(f"  {tag:7s}: peak_u={pk:.4f}{over}  mean_u={mn:.4f}  dur={dur:.2f}s")
    if "thesis" in summ and "effort" in summ:
        dpk = summ["effort"][0] - summ["thesis"][0]
        dmn = summ["effort"][1] - summ["thesis"][1]
        print(f"\n  effort - thesis:  peak_u {dpk:+.4f}   mean_u {dmn:+.4f}")
        print("  (negative = effort selector loads the joints less)")


if __name__ == "__main__":
    raise SystemExit(main())