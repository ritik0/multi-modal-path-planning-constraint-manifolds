"""Overlay two saved plans (thesis vs effort) as a utilisation-vs-time comparison.

Reads two .npz plan files produced by run_panda.py / exp_surface_effort_transport.py
and plots, for each, the peak normalised joint utilisation and the end-effector
height across the slide-A / free-carry / slide-B phases -- the same style as the
thesis worked-transport figure, but with BOTH plans overlaid so the effort
selector can be compared against the thesis planner directly.

Utilisation is recomputed here from the saved joint path via the same free-hold
definition the planner uses (|g(q)| / (rho * tau_max)), so no CSV export is
needed -- the .npz alone is enough.

Usage (from payload_planner/):
    python -B compare_effort.py \
        --thesis plan_thesis_4p1.npz \
        --effort  panda_plan_effort.npz \
        --derate 0.5945 \
        --out compare_4p1.png

If --derate is omitted it is read from the plan's support_derate field if present,
else defaults to 1.0 (i.e. plots raw utilisation).
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_panda as rp
from scene import Scene
from constraints import TorqueFeasibility

PHASE_LABELS = {"slide_A": "slide A", "free_flight": "free carry", "slide_B": "slide B"}


def load_plan(path):
    d = np.load(path, allow_pickle=True)
    segs = list(d["segments"])           # object array: one joint path per segment
    labels = [str(x) for x in d["seg_labels"]]
    return segs, labels, d


def per_sample_util_and_z(scene, torque, segs, labels, frame):
    """Return concatenated (phase_id, umax, ee_z) over all segments, in order."""
    umax, ee_z, phase = [], [], []
    for pid, (seg, lab) in enumerate(zip(segs, labels)):
        seg = np.asarray(seg, dtype=float)
        for q in seg:
            u = float(np.max(torque.utilisation(q)))     # |g|/(rho*tau_max)
            z = float(scene.robot.frame_pose(frame, q)[2, 3])
            umax.append(u); ee_z.append(z); phase.append(pid)
    return np.array(umax), np.array(ee_z), np.array(phase), labels


def pseudo_time(n):
    """A normalised 0..1 progress axis (no dynamics timing in the geometric plan)."""
    return np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thesis", required=True)
    ap.add_argument("--effort", required=True)
    ap.add_argument("--derate", type=float, default=None)
    ap.add_argument("--out", default="compare_effort.png")
    ap.add_argument("--csv", default=None, help="optional: also dump the traces to CSV")
    args = ap.parse_args()

    scene = Scene(rp.CFG)
    frame = rp.FRAME
    # derate: CLI wins; else from plan; else 1.0
    derate = args.derate
    if derate is None:
        try:
            derate = float(np.load(args.thesis, allow_pickle=True)["support_derate"])
        except Exception:
            derate = 1.0
    torque = TorqueFeasibility(scene.robot, scene.tau_max, derate=derate)
    print(f"using derate rho = {derate:.4f}")

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax2 = ax.twinx()
    colors = {"thesis": "#0b3d5c", "effort": "#c04a2b"}
    zcolors = {"thesis": "#8fb8cc", "effort": "#e0a08c"}

    summary = {}
    for tag, path in [("thesis", args.thesis), ("effort", args.effort)]:
        if not Path(path).exists():
            print(f"  MISSING {path} -- skipping {tag}")
            continue
        segs, labels, _ = load_plan(path)
        u, z, phase, labs = per_sample_util_and_z(scene, torque, segs, labels, frame)
        t = pseudo_time(len(u))
        ax.plot(t, u, color=colors[tag], lw=1.8, label=f"{tag}: $u_{{max}}$", zorder=3)
        ax2.plot(t, z, color=zcolors[tag], lw=1.4, ls="--",
                 label=f"{tag}: ee height", zorder=2)
        summary[tag] = {"peak_u": float(np.max(u)), "mean_u": float(np.mean(u)),
                        "n": len(u), "labels": labs}

        # phase boundaries (from the FIRST plan drawn) as vertical guides
        if tag == "thesis":
            bounds = np.cumsum([len(np.asarray(s)) for s in segs])[:-1] / len(u)
            for b in bounds:
                ax.axvline(b, color="0.6", ls=":", lw=0.8, zorder=1)

    ax.axhline(1.0, color="#e8890c", ls="--", lw=1.2, zorder=1)
    ax.text(0.01, 1.005, "budget $u=1$", color="#e8890c", fontsize=9, va="bottom")
    ax.set_xlabel("normalised progress")
    ax.set_ylabel(r"max utilisation $u_{max}$")
    ax2.set_ylabel("end-effector height (m)")
    ax.set_ylim(0, 1.1); ax.set_xlim(0, 1)
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("Thesis planner vs. torque-aware effort selector", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out, dpi=170)
    print(f"wrote {args.out}")

    print("\n=== comparison summary ===")
    for tag, s in summary.items():
        print(f"  {tag:7s}: peak_u={s['peak_u']:.4f}  mean_u={s['mean_u']:.4f}  "
              f"samples={s['n']}  phases={s['labels']}")
    if "thesis" in summary and "effort" in summary:
        dp = summary["effort"]["peak_u"] - summary["thesis"]["peak_u"]
        dm = summary["effort"]["mean_u"] - summary["thesis"]["mean_u"]
        print(f"\n  effort - thesis:  peak_u {dp:+.4f}   mean_u {dm:+.4f}")
        print("  (negative = effort selector reduced utilisation)")

    if args.csv:
        import csv as _csv
        # re-emit both traces
        with open(args.csv, "w", newline="") as f:
            w = _csv.writer(f); w.writerow(["plan","progress","umax","ee_z","phase"])
            for tag, path in [("thesis", args.thesis), ("effort", args.effort)]:
                if not Path(path).exists(): continue
                segs, labels, _ = load_plan(path)
                u, z, ph, _ = per_sample_util_and_z(scene, torque, segs, labels, frame)
                t = pseudo_time(len(u))
                for i in range(len(u)):
                    w.writerow([tag, f"{t[i]:.5f}", f"{u[i]:.5f}", f"{z[i]:.5f}", ph[i]])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    raise SystemExit(main())