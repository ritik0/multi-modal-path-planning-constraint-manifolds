"""
visualize.py
============
A minimal picture of the whole idea, using only pinocchio + matplotlib.

It does NOT precompute a region map for the planner to use -- it just *draws*
the live lift-off inequality C(q) over the table so you can see it:

    * left  : top-down map of the table.  For each (x, y) we place the object
              there (flat, by projection) and colour it by the torque margin
              = 1 - max|g(q)|/tau_max.   green = arm can hold it alone (may fly),
              red = it cannot (must stay supported).  The black line is the
              lift-off boundary, margin = 0.
    * right : a single straight reach (y = 0) showing the margin crossing zero.

Run:  python -B visualize.py        (saves lift_off_map.png and shows it)
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")               # save to file; remove this line to pop a window
import matplotlib.pyplot as plt

from scene import Scene, FRANKA_TAU_MAX
from constraints import FrameConstraint, TorqueFeasibility, Mode

HERE = Path(__file__).resolve().parent


def place_and_margin(scene, torque, x, y, seed):
    """Project the object onto (x, y, table_h) held flat; return (margin, q)."""
    place = Mode("place", equalities=[FrameConstraint(
        scene.robot, scene.controlled_frame,
        target_position=[x, y, scene.table_height], pos_mask=(1, 1, 1),
        target_rotation=np.diag([1., -1., -1.]), rot_mask=(1, 1, 1))])
    q, ok = place.project(seed)
    if not ok:
        return np.nan, seed
    return torque.margin(q), q


def main():
    cfg = {
        "urdf": str(HERE / "urdf" / "fer_franka_hand_payload.urdf"),
        "controlled_frame": "payload",
        "tau_max": FRANKA_TAU_MAX,
        "torque_derate": 0.85,
        "table_height": 0.30,
    }
    scene = Scene(cfg)
    torque = TorqueFeasibility(scene.robot, scene.tau_max, derate=scene.derate)

    xs = np.linspace(0.20, 0.85, 45)
    ys = np.linspace(-0.50, 0.50, 45)
    margin = np.full((len(ys), len(xs)), np.nan)

    seed = np.array([0, -0.4, 0, -2.0, 0, 1.8, 0.0])
    for iy, y in enumerate(ys):
        row_seed = seed                       # warm-start each row from the base seed
        for ix, x in enumerate(xs):
            m, row_seed = place_and_margin(scene, torque, x, y, row_seed)
            margin[iy, ix] = m

    fig, (axmap, axline) = plt.subplots(1, 2, figsize=(12, 5),
                                        gridspec_kw={"width_ratios": [1.3, 1]})

    # --- left: the map ---
    vmax = np.nanmax(np.abs(margin))
    im = axmap.pcolormesh(xs, ys, margin, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                          shading="auto")
    cs = axmap.contour(xs, ys, margin, levels=[0.0], colors="k", linewidths=2)
    axmap.clabel(cs, fmt={0.0: "lift-off boundary"})
    axmap.plot(0, 0, "ks", ms=8)
    axmap.annotate("robot base", (0, 0), textcoords="offset points", xytext=(6, 6))
    axmap.set_xlabel("x  [m]  (reach)")
    axmap.set_ylabel("y  [m]  (side)")
    axmap.set_title("Lift-off map: green = arm holds it (fly),\nred = must stay on table")
    axmap.set_aspect("equal")
    fig.colorbar(im, ax=axmap, label="torque margin  (1 - max util)")

    # --- right: the reach slice at y = 0 ---
    iy0 = int(np.argmin(np.abs(ys)))
    C = -margin[iy0]                          # C(q) = util - 1 = -margin
    axline.axhline(0, color="gray", lw=1)
    axline.plot(xs, C, "o-", color="tab:blue")
    axline.fill_between(xs, C, 0, where=(C <= 0), color="tab:green", alpha=0.25)
    axline.fill_between(xs, C, 0, where=(C > 0), color="tab:red", alpha=0.25)
    axline.set_xlabel("x  [m]  (reach, at y = 0)")
    axline.set_ylabel("C(q) = max util - 1")
    axline.set_title("Reach slice: C(q) crosses 0 at the boundary\n(<=0 may fly,  >0 must slide)")

    fig.tight_layout()
    out = HERE / "lift_off_map.png"
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()