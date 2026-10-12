import numpy as np, run_panda, transport as T, transfer_foliation as fol
from scene import Scene
sc = Scene(run_panda.CFG)
rng = np.random.default_rng(20260612)
lifts = T.sample_transitions(sc, sc.surface('A'), 200, other_surface=sc.surface('B'), rng=rng, support_margin=0.05)
sets  = T.sample_transitions(sc, sc.surface('B'), 200, other_surface=sc.surface('A'), rng=rng, support_margin=0.05)
q_lift = sorted(lifts, key=lambda q: float(np.linalg.norm(q)))[0]
q_set  = sorted(sets,  key=lambda q: float(np.linalg.norm(q)))[0]
h, psi, info = fol.select_transfer_2d(sc, q_lift, q_set, solve_time=15, delta=0.08, space="atlas")
print("CHOSEN (h, psi):", h, psi)