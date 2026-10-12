#!/usr/bin/env python3
"""
probe_where_peak.py -- WHERE along the carry does peak torque occur?

Uses the existing (reliable) evaluate_leaf at the fixed-height optimum, then
reports the peak torque utilisation SEPARATELY for the three sub-parts:
  rise (lift->leaf), cross (on the leaf), descend (leaf->set).

This tells us whether a variable-height profile could even help:
  - if the peak is on the CROSS (constant height) -> lowering the ends won't
    help; the fixed height is where the torque is spent; foliation is right.
  - if the peak is on the RISE/DESCEND -> the transitions dominate, and a
    smarter height profile (or transition) is where the gain is.
Also reports the height (z) at the peak, so we see if high carry = high torque.
"""
import os
import numpy as np
import transport as T
import transfer_foliation as fol
from constraints import TorqueFeasibility


def util_profile(scene, path):
    tq = TorqueFeasibility(scene.robot, scene.tau_max, derate=scene.derate)
    out = []
    for q in np.asarray(path, float):
        try:
            u = float(np.max(tq.utilisation(q)))
            z = float(scene.robot.frame_pose(scene.controlled_frame, q)[2, 3])
            out.append((u, z))
        except Exception as e:
            print("  [warn] sample skipped:", repr(e))
    return out


def main():
    import run_panda
    from scene import Scene
    scene = Scene(run_panda.CFG)
    st = float(os.environ.get("PAYLOAD_SOLVE_TIME", "15"))
    delta = float(os.environ.get("PAYLOAD_DELTA", "0.08"))
    space = os.environ.get("PAYLOAD_CONSTRAINED_SPACE", "atlas")

    sa, sb = scene.surface("A"), scene.surface("B")
    rng = np.random.default_rng(20260612)
    lifts = T.sample_transitions(scene, sa, 300, other_surface=sb, rng=rng, support_margin=0.05)
    sets = T.sample_transitions(scene, sb, 300, other_surface=sa, rng=rng, support_margin=0.05)
    q_lift = sorted(lifts, key=lambda q: float(np.linalg.norm(q)))[0]
    q_set = sorted(sets, key=lambda q: float(np.linalg.norm(q)))[0]

    h, info = fol._select_continuous(scene, q_lift, q_set, "RRTConnect",
                                     st, delta, space, 10.0, 0.1, False)
    if h is None:
        raise SystemExit("fixed-height selection failed")
    print(f"fixed-height optimum h*={h:.3f}, reported cross util={info['util_peak']:.3f}")
    print("=" * 60)
    for tag, path in info.get("segments", []):
        prof = util_profile(scene, path)
        if not prof:
            print(f"{tag:14s}: (no samples)"); continue
        us = [u for u, _ in prof]
        i = int(np.argmax(us))
        peak_u, peak_z = prof[i]
        zr = [z for _, z in prof]
        print(f"{tag:14s}: peak util={peak_u:.3f} at z={peak_z:.3f}  "
              f"(z range [{min(zr):.3f},{max(zr):.3f}], {len(prof)} pts)")
    print("=" * 60)
    print("If peak is on 'cross' -> torque is spent at the carry height itself.")
    print("If peak is on 'rise'/'leaf_to_set' -> the up/down transitions dominate.")


if __name__ == "__main__":
    main()