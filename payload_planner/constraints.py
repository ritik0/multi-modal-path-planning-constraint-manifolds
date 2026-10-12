"""
constraints.py
==============
The whole method expressed as constraints.  Two flavours only:

    EQUALITY    F(q) = 0     "stay exactly ON this surface"   (defines a manifold)
    INEQUALITY  C(q) <= 0    "stay on the allowed SIDE"        (carves the feasible region)

A Mode is just a bundle of equalities + inequalities.  Moving = keep equalities
satisfied (project onto the surface) while keeping inequalities satisfied.  A
transition (lift-off, set-down, regrasp) is a configuration that satisfies two
modes at once -- found by projection, not by hand-placed waypoints.

v2 additions (this file):
  * FrameConstraint now supports a TOLERANCE BAND per axis (pos_tol / rot_tol).
    tol = 0  -> hard equality "be exactly here"  (old behaviour, unchanged).
    tol > 0  -> a Task-Space Region (Berenson TSR): "be ANYWHERE in this box".
    Residual is 0 inside the band and the signed overflow outside it, so the
    SAME projection turns a pinned pose into a pose REGION the planner can pick
    within.  This is how "let the planner decide the orientation/grasp" is
    expressed: relax the band, add a cost (e.g. torque margin) to choose.
  * Mode exposes residual(q)/jacobian(q)/ambient_dim/codim/within_bounds so it
    can be dropped straight into the existing OMPL ConstrainedStateSpace setup.

No PyBullet.  Geometry/Jacobians come from RobotModel (Pinocchio); collision
from collision.py.
"""

import numpy as np
import pinocchio


def _band(e, tol):
    """
    TSR residual for raw errors `e` against a symmetric band of half-width `tol`.
    Returns 0 where |e| <= tol (inside the region) and the signed amount by which
    e leaves the band otherwise.  Vectorised, elementwise.

        e > +tol  ->  e - tol   (> 0)
        e < -tol  ->  e + tol   (< 0)
        else      ->  0
    """
    e = np.asarray(e, float)
    tol = np.asarray(tol, float)
    return np.maximum(e - tol, 0.0) + np.minimum(e + tol, 0.0)


# --------------------------------------------------------------------------- #
#  EQUALITY CONSTRAINTS  ->  F(q) = 0   (these define manifolds)
# --------------------------------------------------------------------------- #
class FrameConstraint:
    """
    General task-space equality (or TSR) on one robot frame.  Select which
    position axes and/or orientation axes are pinned via masks; give each axis
    a tolerance to turn a pin into an allowed band.  This single class covers:

        * "object stays on the table plane"   -> pos_mask=(0,0,1), target z
        * "object held flat"                  -> rot_mask=(1,1,0)  (roll,pitch)
        * "place object at exactly (x,y,z)"   -> pos_mask=(1,1,1)
        * "grasp pose REGION" / "carry tilt allowed within +-15deg"
                                               -> rot_mask=(1,1,0), rot_tol=0.26

    value(q)    -> stacked residual (=0 on the surface / inside the region)
    jacobian(q) -> d(residual)/dq, used for projection (rows gate to 0 inside
                   the band, so a satisfied region exerts no pull).
    """

    def __init__(self, robot, frame_name,
                 target_position=None, pos_mask=(0, 0, 0), pos_tol=0.0,
                 target_rotation=None, rot_mask=(0, 0, 0), rot_tol=0.0,
                 name="frame"):
        self.robot = robot
        self.frame = frame_name
        self.target_position = None if target_position is None else np.asarray(target_position, float)
        self.target_rotation = None if target_rotation is None else np.asarray(target_rotation, float)
        self.pos_mask = np.asarray(pos_mask, bool)
        self.rot_mask = np.asarray(rot_mask, bool)
        # half-widths; broadcast a scalar to all three axes
        self.pos_tol = np.broadcast_to(np.asarray(pos_tol, float), (3,)).copy()
        self.rot_tol = np.broadcast_to(np.asarray(rot_tol, float), (3,)).copy()
        self.name = name

    def value(self, q):
        T = self.robot.frame_pose(self.frame, q)
        res = []
        if self.pos_mask.any():
            err = T[:3, 3] - self.target_position
            res.append(_band(err, self.pos_tol)[self.pos_mask])
        if self.rot_mask.any():
            # rotation error as a 3-vector in the world frame.
            # Sign chosen so d(Rerr)/dq == angular Jacobian rows (Gauss-Newton).
            Rerr = pinocchio.log3(T[:3, :3] @ self.target_rotation.T)
            res.append(_band(Rerr, self.rot_tol)[self.rot_mask])
        return np.concatenate(res) if res else np.zeros(0)

    def jacobian(self, q):
        T = self.robot.frame_pose(self.frame, q)
        J = self.robot.frame_jacobian(self.frame, q)   # 6 x nq, rows 0:3 lin, 3:6 ang
        rows = []
        if self.pos_mask.any():
            err = T[:3, 3] - self.target_position
            active = (_band(err, self.pos_tol) != 0.0)          # 0 inside the band
            Jp = J[0:3] * active[:, None]
            rows.append(Jp[self.pos_mask])
        if self.rot_mask.any():
            Rerr = pinocchio.log3(T[:3, :3] @ self.target_rotation.T)
            active = (_band(Rerr, self.rot_tol) != 0.0)
            Jr = J[3:6] * active[:, None]
            rows.append(Jr[self.rot_mask])
        return np.vstack(rows) if rows else np.zeros((0, self.robot.nq))


def PlaneConstraint(robot, frame_name, height, tol=0.0, name="on_plane"):
    """Object frame stays on (or within tol of) the horizontal plane z = height."""
    return FrameConstraint(robot, frame_name,
                           target_position=[0, 0, height], pos_mask=(0, 0, 1),
                           pos_tol=[0, 0, tol], name=name)


def FlatCarryConstraint(robot, frame_name, R_flat=np.eye(3), tol=0.0, name="flat_carry"):
    """
    Object held flat (roll & pitch pinned, yaw free).  tol > 0 allows a tilt
    band of +-tol rad on roll & pitch -- i.e. "approximately flat", which is the
    knob you relax to let the planner choose a carry orientation.
    """
    return FrameConstraint(robot, frame_name,
                           target_rotation=R_flat, rot_mask=(1, 1, 0),
                           rot_tol=[tol, tol, 0.0], name=name)


def TSR(robot, frame_name, target_position=None, pos_mask=(0, 0, 0), pos_tol=0.0,
        target_rotation=None, rot_mask=(0, 0, 0), rot_tol=0.0, name="tsr"):
    """
    Explicit Task-Space-Region constructor (Berenson).  Identical machinery to
    FrameConstraint -- this name just makes intent obvious at the call site:
    a grasp/placement expressed as a BOX of allowed poses, reached by the same
    Jacobian projection (IK-free), choosable within by a cost.
    """
    return FrameConstraint(robot, frame_name,
                           target_position=target_position, pos_mask=pos_mask, pos_tol=pos_tol,
                           target_rotation=target_rotation, rot_mask=rot_mask, rot_tol=rot_tol,
                           name=name)


# --------------------------------------------------------------------------- #
#  INEQUALITY CONSTRAINTS  ->  C(q) <= 0   (these carve the feasible region)
# --------------------------------------------------------------------------- #
def _skew(v): # cross product calculation 
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


class TorqueFeasibility:
    """
    THE lift-off test.  C(q) = max_j |g_j(q)| / tau_max_j - 1.

        C(q) <= 0  ->  the arm can hold the (URDF) payload by itself
                       at this pose  ->  it MAY fly free here.
        C(q) >  0  ->  it cannot  ->  the object must stay supported on the
                       table, which carries the excess load.
    """

    def __init__(self, robot, tau_max, derate=1.0, name="torque_feasible"):
        self.robot = robot
        self.tau_max = np.asarray(tau_max, float) * derate
        self.name = name

    def utilisation(self, q):
        return np.abs(self.robot.gravity(q)) / self.tau_max

    def value(self, q):
        return float(np.max(self.utilisation(q)) - 1.0)

    def margin(self, q):
        """1 - max utilisation.  Positive = arm can hold it alone."""
        return 1.0 - float(np.max(self.utilisation(q)))


class ContactSupportFeasibility:
    r"""
    Load-sharing torque feasibility with a UNILATERAL surface contact.

    Static equilibrium with the surface pushing on the controlled frame:

        tau_arm = g(q) - J_c(q)^T  W

    g(q) is the gravity-holding joint torque (payload in the URDF); J_c is the
    contact Jacobian at the contact point; W is the contact wrench the surface
    applies.  We minimise the contact effort and let the arm carry the rest,
    pushed to its torque budget:

        minimise   || W ||
        s.t.       tau_arm = g(q) - J_c^T W
                   |tau_arm| <= tau_budget
                   <contact cone on W>

        W == 0 at the optimum  ->  MAY-LIFT   (arm holds it alone)
        W != 0 at the optimum  ->  MUST-SLIDE (surface shares the load)

    Three contact models of increasing richness (`contact_model`):

      "normal"  W = lambda n,   lambda >= 0                  (scalar, analytic)
                a single vertical push -- the original model.

      "force"   W = (fx,fy,fz),  fz >= 0,  ||f_t|| <= mu fz   (R^3, the board)
                normal push + Coulomb friction (the mentor's tau = D^T lam + g,
                lam in R^3, lam = (D D^T)^-1 D (tau - g)).

      "wrench"  W = (f; m) in R^6,  + tipping ||m_t|| <= r fz
                                    + twist   |m_n|  <= mu_t fz   (the drawing)
                the full contact wrench: the rotational moments that resist the
                cylinder TIPPING (base radius r) and TWISTING.

    The board's closed form  lambda = (J_c J_c^T)^-1 J_c (g - tau)  is
    `contact_force_for_torque` (least-squares force for a chosen arm torque).

    `support_derate` (<= 1) shrinks the budget the load-sharing AIMS for, so the
    slide leaves torque headroom for the motion terms; utilisation is reported
    against the TRUE derated budget, so it tops out near support_derate.
    """

    def __init__(
        self,
        robot,
        tau_max,
        frame_name="payload",
        local_contact_point=(0.0, 0.0, -0.173),
        normal_world=(0.0, 0.0, 1.0),
        derate=1.0,
        max_lambda=np.inf,
        contact_model="wrench",
        mu=0.0,
        mu_torsion=0.0,
        base_radius=0.0,
        support_derate=1.0,
        name="support_load_sharing",
    ):
        self.robot = robot
        self.tau_max = np.asarray(tau_max, dtype=float).reshape(-1) * float(derate)
        self.tau_budget = self.tau_max * float(support_derate)
        self.frame = str(frame_name)
        self.local_contact_point = np.asarray(local_contact_point, dtype=float).reshape(3)
        self.normal_world = np.asarray(normal_world, dtype=float).reshape(3)
        n_norm = float(np.linalg.norm(self.normal_world))
        if n_norm <= 1.0e-12:
            raise ValueError("normal_world must be non-zero")
        self.normal_world = self.normal_world / n_norm
        self.max_lambda = float(max_lambda)
        self.contact_model = str(contact_model).strip().lower()
        if self.contact_model not in ("normal", "force", "wrench"):
            raise ValueError("contact_model must be 'normal', 'force' or 'wrench'")
        self.mu = float(mu)
        self.mu_torsion = float(mu_torsion)
        self.base_radius = float(base_radius)
        self.support_derate = float(support_derate)
        self.name = name

    @classmethod
    def from_scene(cls, scene, **overrides):
        """Single construction point for every caller -- reads scene.config knobs."""
        cfg = getattr(scene, "config", {}) or {}
        kwargs = dict(
            frame_name=getattr(scene, "controlled_frame", "payload"),
            local_contact_point=cfg.get("support_contact_local", [0.0, 0.0, -0.173]),
            normal_world=cfg.get("support_normal_world", [0.0, 0.0, 1.0]),
            derate=scene.derate,
            max_lambda=float(cfg.get("support_force_max", np.inf)),
            contact_model=cfg.get("support_contact_model", "normal"),
            mu=float(cfg.get("support_mu", 0.0)),
            mu_torsion=float(cfg.get("support_mu_torsion", 0.0)),
            base_radius=float(cfg.get("support_base_radius", 0.0)),
            support_derate=float(cfg.get("support_derate", 1.0)),
        )
        kwargs.update(overrides)
        return cls(scene.robot, scene.tau_max, **kwargs)

    # -- contact Jacobian at the contact point ----------------------------
    def contact_jacobian(self, q, full=False):
        q = np.asarray(q, dtype=float)
        T = self.robot.frame_pose(self.frame, q)
        J = self.robot.frame_jacobian(self.frame, q)        # 6 x nq
        Jv, Jw = J[0:3, :], J[3:6, :]
        r_world = T[:3, :3] @ self.local_contact_point      # offset to the contact point
        Jc_v = Jv - _skew(r_world) @ Jw                     # linear at contact point
        if full:
            return np.vstack([Jc_v, Jw])                    # 6 x nq (forces + moments)
        return Jc_v                                         # 3 x nq

    def support_direction_torque(self, q):
        """Joint torque per unit normal force (the analytic scalar path)."""
        return self.contact_jacobian(q).T @ self.normal_world

    # -- dispatch ----------------------------------------------------------
    def solve(self, q):
        q = np.asarray(q, dtype=float)
        g = np.asarray(self.robot.gravity(q), dtype=float).reshape(-1)
        if self.contact_model == "normal":
            return self._solve_normal(q, g)
        return self._solve_cone(q, g, dim=(6 if self.contact_model == "wrench" else 3))

    def _pack(self, feasible, lam_normal, contact, tau, reason):
        tau = np.asarray(tau, dtype=float)
        return {
            "feasible": bool(feasible),
            "lambda": float(lam_normal),
            "contact": np.asarray(contact, dtype=float),
            "tau": tau,
            "utilisation": np.abs(tau) / self.tau_max,
            "reason": str(reason),
        }

    # analytic scalar normal force (fast path; aims at tau_budget) ---------
    def _solve_normal(self, q, g):
        a = np.asarray(self.support_direction_torque(q), dtype=float).reshape(-1)
        budget = self.tau_budget.reshape(-1)
        if not (g.shape == a.shape == budget.shape):
            return self._pack(False, np.nan, [0.0, 0.0, np.nan], g, "shape_mismatch")
        lo, hi = 0.0, self.max_lambda
        eps = 1.0e-12
        for gi, ai, ti in zip(g, a, budget):
            if abs(ai) <= eps:
                if abs(gi) > ti + 1.0e-10:
                    return self._pack(False, np.nan, [0.0, 0.0, np.nan], g,
                                      "torque_violation_uncontrollable_by_support")
                continue
            if ai > 0.0:
                lo_i, hi_i = (gi - ti) / ai, (gi + ti) / ai
            else:
                lo_i, hi_i = (gi + ti) / ai, (gi - ti) / ai
            lo, hi = max(lo, float(lo_i)), min(hi, float(hi_i))
            if lo > hi + 1.0e-10:
                return self._pack(False, np.nan, [0.0, 0.0, np.nan], g, "empty_lambda_interval")
        lam = max(0.0, lo)
        if lam > hi + 1.0e-10:
            return self._pack(False, np.nan, [0.0, 0.0, np.nan], g, "lambda_outside_interval")
        contact = lam * self.normal_world
        tau = g - a * lam
        ok = bool(np.max(np.abs(tau) / self.tau_max) <= 1.0 + 1.0e-9)
        return self._pack(ok, lam, contact, tau, "ok")

    # 3-force / 6-wrench min-norm cone solve (scipy SLSQP) -----------------
    def _solve_cone(self, q, g, dim):
        try:
            from scipy.optimize import minimize
        except Exception as exc:
            return self._pack(False, np.nan, np.zeros(dim), g, f"scipy_unavailable:{exc}")

        Jc = self.contact_jacobian(q, full=(dim == 6))      # 3xn or 6xn
        budget = self.tau_budget.reshape(-1)
        n = self.normal_world
        mu, mut, r = self.mu, self.mu_torsion, self.base_radius

        def tau_of(W):
            return g - Jc.T @ W

        def fn_of(W):
            return float(W[0:3] @ n)

        cons = [
            {"type": "ineq", "fun": lambda W: budget - tau_of(W)},      # upper torque
            {"type": "ineq", "fun": lambda W: budget + tau_of(W)},      # lower torque
            {"type": "ineq", "fun": lambda W: np.array([fn_of(W)])},    # normal >= 0
        ]
        if mu > 0.0:
            def friction(W):
                f = W[0:3]; fn = float(f @ n)
                return np.array([(mu * fn) ** 2 - (float(f @ f) - fn * fn)])
            cons.append({"type": "ineq", "fun": friction})
        if dim == 6:
            if r > 0.0:
                def tipping(W):
                    m = W[3:6]; fn = float(W[0:3] @ n); mn = float(m @ n)
                    return np.array([(r * fn) ** 2 - (float(m @ m) - mn * mn)])
                cons.append({"type": "ineq", "fun": tipping})
            if mut > 0.0:
                def twist(W):
                    fn = float(W[0:3] @ n); mn = float(W[3:6] @ n)
                    return np.array([(mut * fn) ** 2 - mn * mn])
                cons.append({"type": "ineq", "fun": twist})

        res = minimize(lambda W: float(W @ W), np.zeros(dim), jac=lambda W: 2.0 * W,
                       constraints=cons, method="SLSQP",
                       options={"ftol": 1.0e-9, "maxiter": 300})
        if not res.success:
            return self._pack(False, np.nan, np.zeros(dim), g, f"cone_solve_failed:{res.message}")
        W = res.x
        contact = W[0:3] if dim == 3 else W
        tau = g - Jc.T @ W
        ok = bool(np.max(np.abs(tau) / self.tau_max) <= 1.0 + 1.0e-9)
        return self._pack(ok, fn_of(W), contact, tau, "ok")

    # -- max's closed form: lambda = (Jc Jc^T)^-1 Jc (g - tau) ----------
    def contact_force_for_torque(self, q, tau):
        q = np.asarray(q, dtype=float)
        g = np.asarray(self.robot.gravity(q), dtype=float).reshape(-1)
        Jc = self.contact_jacobian(q)                       # 3 x nq
        tau = np.asarray(tau, dtype=float).reshape(-1)
        # lambda = (Jc Jc^T)^-1 Jc (g - tau).  Use pinv, not inv: when the arm
        # is at a singular contact configuration Jc Jc^T drops rank and a plain
        # inverse blows up.  pinv((Jc Jc^T)) gives the minimum-norm least-squares
        # lambda instead -- the "not always invertible, so transpose-and-invert
        # the pseudo-inverse"
        return np.linalg.pinv(Jc @ Jc.T) @ (Jc @ (g - tau))

    # -- API (unchanged for existing callers) -----------------------------
    def utilisation(self, q):
        return self.solve(q)["utilisation"]

    def lambda_value(self, q):
        return float(self.solve(q)["lambda"])

    def supported_torque(self, q):
        return np.asarray(self.solve(q)["tau"], dtype=float)

    def value(self, q):
        sol = self.solve(q)
        if not sol["feasible"]:
            return 1.0
        return float(np.max(sol["utilisation"]) - 1.0)

    def margin(self, q):
        return -self.value(q)


class JointLimit:
    """Joint-box inequality.  C(q) <= 0 inside the model limits."""

    def __init__(self, robot, tol=1e-6, name="joint_limits"):
        self.robot = robot
        qlo, qhi = robot.joint_limits
        self.qlo = np.asarray(qlo, float)
        self.qhi = np.asarray(qhi, float)
        self.tol = float(tol)
        self.name = name

    def value(self, q):
        q = np.asarray(q, float)
        return float(np.max(np.concatenate([self.qlo - q, q - self.qhi])) - self.tol)


class WorkspaceBoundary:
    """Keep a frame's position inside an axis-aligned box.  C(q) <= 0 inside."""

    def __init__(self, robot, frame_name, lo, hi, name="boundary"):
        self.robot = robot
        self.frame = frame_name
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.name = name

    def value(self, q):
        p = self.robot.frame_pose(self.frame, q)[:3, 3]
        return float(np.max(np.concatenate([self.lo - p, p - self.hi])))


class PayloadFloorConstraint:
    """
    Keep the lowest point of the carried payload at or above a floor height.
    C(q) = (floor + clearance) - z_min(q)  <=  0   when the payload clears the floor.

    During free flight the object is no longer supported by any surface, so
    without this the planner is free to route the carry THROUGH the tables (the
    object dips below table-top mid-flight).  This inequality forbids that: it
    samples the payload's local collision points, takes the lowest world-z, and
    requires it to stay at/above the floor plus a clearance margin.  Cheap,
    robust, and independent of solid-box collision wiring.
    """

    def __init__(self, robot, frame_name, floor_z=0.0, clearance=0.0,
                 local_points=None, name="payload_floor"):
        self.robot = robot
        self.frame = str(frame_name)
        self.floor_z = float(floor_z)
        self.clearance = float(clearance)
        # Local sample points on the payload (its frame).  Defaults to a single
        # point at the frame origin; pass the payload's collision points for a
        # tight bound on the true lowest corner.
        if local_points is None:
            local_points = [(0.0, 0.0, 0.0)]
        self.local_points = np.asarray(local_points, float).reshape(-1, 3)
        self.name = name

    def _z_min(self, q):
        T = self.robot.frame_pose(self.frame, q)      # 4x4
        R, p = T[:3, :3], T[:3, 3]
        world = (self.local_points @ R.T) + p          # (N,3)
        return float(np.min(world[:, 2]))

    def value(self, q):
        # <= 0 when the lowest payload point is above floor + clearance
        return float((self.floor_z + self.clearance) - self._z_min(q))


class CollisionConstraint:
    """
    Not inside any obstacle.  C(q) <= 0 when collision-free.  Checks a set of
    frames (e.g. the payload corners / the hand) against collision.py objects.
    """

    def __init__(
        self,
        robot,
        frame_names,
        collision_objects,
        name="collision_free",
        frame_points=None,
        self_collision_frames=(),
        self_collision_min_distance=0.0,
    ):
        self.robot = robot
        self.frames = list(frame_names)
        self.objects = collision_objects
        self.name = name
        self.frame_points = {
            str(frame): np.asarray(points, dtype=float).reshape((-1, 3))
            for frame, points in (frame_points or {}).items()
        }
        self.self_collision_frames = list(self_collision_frames)
        self.self_collision_min_distance = float(self_collision_min_distance)

    def _points_for_frame(self, frame_name, q):
        T = self.robot.frame_pose(frame_name, q)
        points = self.frame_points.get(frame_name)
        if points is None or len(points) == 0:
            return [T[:3, 3]]
        R = T[:3, :3]
        p = T[:3, 3]
        return [p + R @ local for local in points]

    def hit_info(self, q):
        """Return a raw collision hit tuple from the exact query feasible() uses."""
        positions = {}
        for f in self.frames:
            points = self._points_for_frame(f, q)
            positions[f] = points[0]
            for point_index, p in enumerate(points):
                for obj in self.objects:
                    if obj.is_in_collision(p):
                        return {
                            "kind": "environment",
                            "frame": f,
                            "point_index": int(point_index),
                            "point": np.asarray(p, dtype=float),
                            "object": obj.__class__.__module__ + "." + obj.__class__.__name__,
                            "object_name": str(getattr(obj, "obstacle", {}).get("name", "<unnamed>")),
                        }
        if self.self_collision_min_distance > 0.0 and self.self_collision_frames:
            for f in self.self_collision_frames:
                if f not in positions:
                    positions[f] = self.robot.frame_pose(f, q)[:3, 3]
            frames = self.self_collision_frames
            for i, a in enumerate(frames):
                pa = positions[a]
                for j in range(i + 3, len(frames)):
                    b = frames[j]
                    if np.linalg.norm(pa - positions[b]) < self.self_collision_min_distance:
                        return {
                            "kind": "self",
                            "frame_a": a,
                            "frame_b": b,
                            "distance": float(np.linalg.norm(pa - positions[b])),
                        }
        return None

    def value(self, q):
        return 1.0 if self.hit_info(q) is not None else -1.0


# --------------------------------------------------------------------------- #
#  MODE = a bundle of equalities + inequalities
# --------------------------------------------------------------------------- #
class Mode:
    """
    A mode of operation, e.g. "object supported on the table" or "object in
    free flight".  Defined purely by which constraints are active.
    """

    def __init__(self, name, equalities=(), inequalities=()):
        self.name = name
        self.equalities = list(equalities)
        self.inequalities = list(inequalities)

    # --- equality side (the manifold) ---
    def equality_residual(self, q):
        parts = [c.value(q) for c in self.equalities]
        return np.concatenate(parts) if parts else np.zeros(0)

    def equality_jacobian(self, q):
        parts = [c.jacobian(q) for c in self.equalities]
        return np.vstack(parts) if parts else np.zeros((0, len(q)))

    def on_manifold(self, q, tol=1e-4):
        r = self.equality_residual(q)
        return r.size == 0 or np.linalg.norm(r) <= tol

    # --- inequality side (the feasible region) ---
    def inequality_values(self, q):
        return {c.name: c.value(q) for c in self.inequalities}

    def feasible(self, q, tol=1e-9):
        return all(v <= tol for v in self.inequality_values(q).values())

    def valid(self, q, tol=1e-4):
        return self.on_manifold(q, tol) and self.feasible(q)

    # --- the IK-free retraction: project q onto this mode's manifold ---
    def project(self, q0, max_iter=100, tol=1e-6, step=1.0, max_step=0.4,
                qlo=None, qhi=None):
        """
        Newton / Jacobian-pseudoinverse projection onto F(q)=0 (CBiRRT-style
        retraction; replaces explicit IK).  The per-iteration step norm is
        capped at `max_step` rad so projection stays near the seed instead of
        drifting to a kinematically-equivalent but out-of-range branch.  When
        bounds are supplied, the iterate is clipped to the joint box after each
        Newton step.  Returns (q_projected, ok).
        """
        q = np.array(q0, float)
        qlo = None if qlo is None else np.asarray(qlo, float)
        qhi = None if qhi is None else np.asarray(qhi, float)
        for _ in range(max_iter):
            r = self.equality_residual(q)
            if r.size == 0 or np.linalg.norm(r) <= tol:
                return q, True
            J = self.equality_jacobian(q)
            dq = step * np.linalg.lstsq(J, r, rcond=None)[0]
            n = np.linalg.norm(dq)
            if n > max_step:                 # cap the step to stay local
                dq *= max_step / n
            q = q - dq
            if qlo is not None and qhi is not None:
                q = np.minimum(np.maximum(q, qlo), qhi)
        return q, np.linalg.norm(self.equality_residual(q)) <= tol

    # --- OMPL ConstrainedStateSpace hooks --------------------------------- #
    # These let a Mode be wrapped directly as an ompl Constraint: the equality
    # set IS the implicit manifold OMPL projects onto.  (Inequalities are NOT
    # here -- they belong in the state-validity checker; see ompl_planner.py.)
    def residual(self, q):
        return self.equality_residual(np.asarray(q, float))

    def jacobian(self, q):
        return self.equality_jacobian(np.asarray(q, float))

    def codim(self, q_ref):
        """Number of equality rows (the manifold codimension) at a reference q."""
        return int(self.equality_residual(np.asarray(q_ref, float)).size)


# --------------------------------------------------------------------------- #
#  TRANSITION between two modes = config valid in BOTH (their intersection)
# --------------------------------------------------------------------------- #
def sample_transition(mode_a, mode_b, seed, n_tries=20, perturb=0.3, rng=None,
                      q_lo=None, q_hi=None, global_frac=0.5):
    """
    Find configurations on BOTH modes' manifolds that satisfy BOTH modes'
    inequalities -- the lift-off / set-down / regrasp set.  The thin
    intersection is reached by projecting candidate seeds onto the stacked
    equality system (never found by luck).  Returns a LIST: in general there
    are MANY such configs, and the discrete layer chooses among them.

    Seeding is deliberately GLOBAL when joint limits are given: a transition is
    typically NOT near any single configuration (e.g. the lift-off region sits
    well inside the must-slide start), so a fraction `global_frac` of the tries
    draw seeds uniformly across the joint box and project onto the intersection,
    while the rest perturb `seed` locally.  Each accepted config is feasible in
    both modes.
    """
    rng = rng or np.random.default_rng(0)
    seed = np.asarray(seed, float)
    combined = Mode("transition",
                    equalities=mode_a.equalities + mode_b.equalities)
    use_global = q_lo is not None and q_hi is not None
    q_lo = None if q_lo is None else np.asarray(q_lo, float)
    q_hi = None if q_hi is None else np.asarray(q_hi, float)
    found = []
    for k in range(n_tries):
        if use_global and rng.random() < global_frac:
            q0 = rng.uniform(q_lo, q_hi)                       # explore globally
        else:
            q0 = seed + perturb * rng.standard_normal(len(seed))
        q, ok = combined.project(q0, qlo=q_lo, qhi=q_hi)
        if ok and mode_a.feasible(q) and mode_b.feasible(q):
            found.append(q)
    return found