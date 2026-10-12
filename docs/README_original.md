# payload_planner — constraint-based payload planning (v2)

A clean restart of the payload work around the mentor's reframing. **Separate
from the old code** — nothing here imports the old project; the old pipeline is
untouched. No PyBullet: dynamics come from **Pinocchio**, collision from
analytic primitives.

## The one idea

Describe every situation as **equality + inequality constraints**, bake the
**payload into the URDF**, and let a single live number decide lift-off.

```
EQUALITY    F(q) = 0     stay exactly ON a surface     -> defines a manifold
INEQUALITY  C(q) <= 0    stay on the allowed SIDE      -> carves the feasible region
```

The key inequality is the **lift-off test**:

```
C(q) = max_j |g_j(q)| / tau_max_j  -  1
       \_______________________/
        gravity torque incl. payload (Pinocchio g(q))

C(q) <= 0  ->  arm can hold the load by itself here  ->  MAY fly free
C(q) >  0  ->  it cannot  ->  object must stay supported on the table
```

No precomputed green/red region map — `C(q)` is evaluated wherever the planner
looks. Lift-off / set-down happen wherever `C(q)` crosses 0; there are **many**
such points, not one hand-placed waypoint.

## Files

| file | role |
|---|---|
| `urdf/fer_franka_hand_payload.urdf` | Franka FER hand **+ a 4 kg `payload` link** fixed at `fer_hand_tcp`. The load now lives in `g(q)`. |
| `robot_model.py` | Pinocchio wrapper (mentor's `RobotModel`). Gives `frame_pose`, `frame_jacobian`, `gravity(q)`, `update_dynamics` → `M, c, g`. |
| `collision.py` | Analytic obstacles (ellipsoid/cylinder/box), mentor's code. "Is this point inside me?" |
| `constraints.py` | Equality (`FrameConstraint`, `PlaneConstraint`, `FlatCarryConstraint`), inequality (`TorqueFeasibility`, `WorkspaceBoundary`, `CollisionConstraint`), `Mode` (a bundle), `Mode.project` (IK-free retraction), `sample_transition`. |
| `scene.py` | Data-driven scene (URDF, controlled frame, `tau_max`, table height, obstacles). No start/goal baked in. |
| `verify.py` | Pinocchio-only end-to-end check: payload in `g(q)`, projection, lift-off boundary, transition sampling. |

## How a plan flows (and what we reuse)

```
scene (data) ─► RobotModel (Pinocchio, payload in URDF) ─► g(q), M, c
                       │
        ┌──────────────┴───────────────┐
   EQUALITIES                      INEQUALITIES
   PlaneConstraint                 TorqueFeasibility   (the lift-off test)
   FlatCarryConstraint             WorkspaceBoundary
                                   CollisionConstraint
        │                               │
        └────────►  Mode = {equalities, inequalities}  ◄────┐
                       │  on_manifold(q)  feasible(q)        │
                       │  project(q)  ← IK-free retraction   │
                       ▼                                     │
            sample_transition(modeA, modeB)  = configs valid in BOTH
            (lift-off / set-down / regrasp, projected, MANY)
                       │
                       ▼
   per-segment motion = REUSE the existing pipeline:
        constrained plan (project onto manifold) → CHOMP → time-param → audit
```

We only rebuilt the **constraint + payload + feasibility** layer. The motion
core (constrained planner, CHOMP smoother, time-parameterisation, audit) is
unchanged and gets called per segment exactly as before.

## Modes in this setup

* **on_table** — `PlaneConstraint` (object on `z = table_h`) + `FlatCarryConstraint`
  (held flat). Torque is *not* required: the table carries the load.
* **free_flight** — `FlatCarryConstraint` + `TorqueFeasibility` (arm must hold it
  alone everywhere along the arc).
* **lift-off transition** — a config on the `on_table` manifold that also passes
  the `free_flight` torque test. The thin intersection, found by projection.

## Grasping is IK-free

A grasp is written as a `FrameConstraint` (a pose **region**, Task-Space-Region
style) and reached with `Mode.project` — the same Jacobian retraction used to
stay on a manifold. No explicit IK; works in joint space; general across objects
(CBiRRT / Berenson).

## Run

```bash
pip install pin
python -B verify.py
```

## Next

* add `WorkspaceBoundary` to `free_flight` so transitions stay in front of the base
* wire the existing constrained planner to plan **within** a mode and to plan the
  free-flight arc between two sampled transitions
* discrete search over (placement, grasp) nodes once regrasp modes are added