"""
ompl_planner.py
===============
Plan WITHIN a single Mode using OMPL on a constrained (projected) state space,
with RRT-Connect by default.  Same split the existing constrained-OMPL setup
uses, driven entirely by a v2 `Mode`:

    Mode.equalities  ->  the implicit MANIFOLD OMPL projects every sample onto
    Mode.inequalities + joint limits  ->  the STATE-VALIDITY checker

RRT-Connect grows two trees through joint space, projecting each extension back
onto the manifold; the path shape is WHATEVER clears the validity check -- never
prescribed.  A path-length objective is installed for planners that use an
optimization objective; for BIT*/RRT* ``solve_time`` becomes its optimization
budget.  No PyBullet, no old-pipeline imports.  Import-safe without OMPL.

State handling note (OMPL version-robust):
  OMPL 1.x Boost.Python exposed ob.State(space) / ScopedState.  OMPL 2.x
  Nanobind removes that constructor and expects allocState(), with constrained
  states usually filled by state.copy(vectorDouble).  This file supports both
  APIs.  Reading also supports both copyToReals signatures:
      old: values = space.copyToReals(state)
      new: space.copyToReals(values, state)
  and falls back to element-wise state[i] reads.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

try:
    from ompl import base as ob
    from ompl import geometric as og
except Exception:
    ob = None
    og = None


@dataclass
class PlanResult:
    success: bool
    path: "np.ndarray | None"
    message: str
    solve_time: float = 0.0
    raw_points: int = 0
    path_points: int = 0
    max_equality_residual: float = float("nan")
    feasible_along_path: bool = False
    planner: str = "RRTConnect"
    constrained_space: str = "projected"


def _q_raw(state, nq):
    """Read a state's ambient values element-wise."""
    return np.asarray([float(state[i]) for i in range(int(nq))], dtype=float)


def _vector_double(values):
    """Build ompl.util.vectorDouble when the binding exposes it."""
    values = [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]
    try:
        from ompl import util as ou

        v = ou.vectorDouble()
        for x in values:
            v.append(float(x))
        return v
    except Exception:
        return values


def _as_array_from_container(container, nq):
    vals = [float(container[i]) for i in range(int(nq))]
    return np.asarray(vals, dtype=float)


def _q(space, state, nq):
    """Read a state into a length-nq numpy vector, robustly across OMPL builds."""
    nq = int(nq)

    # OMPL 1.x Boost.Python: copyToReals(state) -> list-like
    try:
        out = np.asarray(space.copyToReals(state), dtype=float).reshape(-1)
        if out.shape[0] >= nq:
            return out[:nq]
    except Exception:
        pass

    # OMPL 2.x Nanobind: copyToReals(out, state) -> None, mutates vectorDouble.
    for make_out in (
        lambda: _vector_double([0.0] * nq),
        lambda: [0.0] * nq,
    ):
        try:
            out = make_out()
            ret = space.copyToReals(out, state)
            if ret is not None:
                arr = np.asarray(ret, dtype=float).reshape(-1)
            else:
                arr = _as_array_from_container(out, nq).reshape(-1)
            if arr.shape[0] >= nq:
                return arr[:nq]
        except Exception:
            pass

    return _q_raw(state, nq)


def _raw_state(state):
    """Return the raw OMPL state for old ScopedState or new raw-state objects."""
    try:
        return state()
    except Exception:
        return state


def _make_state(space, q, nq):
    """Create and fill an OMPL state for both Boost.Python and Nanobind APIs."""
    q = np.asarray(q, dtype=float).reshape(int(nq))
    q_list = [float(v) for v in q]
    errors = []

    # OMPL 1.x / Boost.Python path: ScopedState constructor and item write.
    try:
        s = ob.State(space)
        for i, value in enumerate(q_list):
            s[i] = float(value)
        return s
    except Exception as exc:
        errors.append(f"ob.State(space): {exc}")

    # OMPL 2.x / Nanobind path: raw allocated state.  For constrained states,
    # item assignment usually fails, but state.copy(vectorDouble) works.
    try:
        s = space.allocState()
    except Exception as exc:
        errors.append(f"space.allocState(): {exc}")
        raise RuntimeError("Could not allocate OMPL state; " + "; ".join(errors))

    writers = []
    if hasattr(s, "copy"):
        writers.extend([
            ("state.copy(list)", lambda: s.copy(q_list)),
            ("state.copy(tuple)", lambda: s.copy(tuple(q_list))),
            ("state.copy(vectorDouble)", lambda: s.copy(_vector_double(q_list))),
        ])
    writers.extend([
        ("space.copyFromReals(list)", lambda: space.copyFromReals(s, q_list)),
        ("space.copyFromReals(tuple)", lambda: space.copyFromReals(s, tuple(q_list))),
        ("space.copyFromReals(vectorDouble)", lambda: space.copyFromReals(s, _vector_double(q_list))),
        ("state[i] assignment", lambda: [s.__setitem__(i, float(v)) for i, v in enumerate(q_list)]),
    ])

    for label, writer in writers:
        try:
            writer()
            # Validate that the filled state is readable.  For constrained states
            # this should also return ambient coordinates.
            readback = _q(space, s, nq)
            if readback.shape[0] == int(nq) and np.all(np.isfinite(readback)):
                return s
            errors.append(f"{label}: unreadable readback {readback}")
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    raise RuntimeError("Could not fill OMPL state; " + "; ".join(errors))


def _path_to_numpy(space, path, nq):
    return np.asarray([_q(space, path.getState(i), nq) for i in range(path.getStateCount())],
                      dtype=float)


def _copy_path(path):
    """Copy an OMPL geometric path when the Python binding exposes a copy ctor."""
    try:
        return og.PathGeometric(path)
    except Exception:
        return None


_PY_OMPL_KEEPALIVE = []


def _raw_state(state):
    """Return raw OMPL state for old ScopedState or new nanobind state."""
    try:
        return state()
    except Exception:
        return state


def _vector_double(values):
    """Build ompl.util.vectorDouble when OMPL 2.x wants a std::vector<double>."""
    try:
        from ompl import util as ou
        v = ou.vectorDouble()
        for x in values:
            v.append(float(x))
        return v
    except Exception:
        return list(float(x) for x in values)


def _set_state_values(state, values):
    """Write q values into old/new OMPL state objects."""
    vals = [float(x) for x in values]

    # OMPL 2.x constrained states usually support state.copy([...])
    if hasattr(state, "copy"):
        try:
            state.copy(vals)
            return state
        except Exception:
            try:
                state.copy(_vector_double(vals))
                return state
            except Exception:
                pass

    # Plain RealVector states support index assignment.
    for i, v in enumerate(vals):
        state[i] = float(v)
    return state


def _set_validity_checker(ss, si, fn):
    """Install validity checker across Boost.Python OMPL and nanobind OMPL 2.x."""
    if hasattr(ob, "StateValidityCheckerFn"):
        try:
            ss.setStateValidityChecker(ob.StateValidityCheckerFn(fn))
            return
        except Exception:
            pass

    # OMPL 2.x may accept a Python callable directly.
    try:
        ss.setStateValidityChecker(fn)
        return
    except Exception as direct_exc:
        last_exc = direct_exc

    # Fallback: subclass StateValidityChecker.
    if hasattr(ob, "StateValidityChecker"):
        class _PyStateValidityChecker(ob.StateValidityChecker):
            def __init__(self, si_):
                super().__init__(si_)

            def isValid(self, state):
                return bool(fn(state))

        checker = _PyStateValidityChecker(si)
        _PY_OMPL_KEEPALIVE.append(checker)
        ss.setStateValidityChecker(checker)
        return

    raise last_exc


class _ModeConstraint(ob.Constraint if ob is not None else object):
    """Wrap a Mode's EQUALITY set as an OMPL implicit-manifold constraint."""

    def __init__(self, mode, nq, codim):
        self.mode = mode
        self.nq = int(nq)
        super().__init__(int(nq), int(codim))

    def function(self, x, out):
        r = np.asarray(self.mode.residual(_q_raw(x, self.nq)), float).reshape(-1)
        for i in range(r.shape[0]):
            out[i] = float(r[i])

    def jacobian(self, x, out):
        J = np.asarray(self.mode.jacobian(_q_raw(x, self.nq)), float)
        for i in range(J.shape[0]):
            for j in range(J.shape[1]):
                out[i, j] = float(J[i, j])


def _certify(mode, path):
    if path is None or len(path) == 0:
        return float("nan"), False
    res = max(float(np.linalg.norm(mode.equality_residual(q))) for q in path)
    feas = all(mode.feasible(q) for q in path)
    return res, bool(feas)


def _certifies(mode, path, residual_tol):
    max_res, feasible = _certify(mode, path)
    ok = feasible and max_res <= max(float(residual_tol) * 10.0, 1e-4)
    return bool(ok), max_res, feasible


def _run_shortcut(ps, path):
    if hasattr(ps, "shortcutPath"):
        ps.shortcutPath(path)
    elif hasattr(ps, "partialShortcutPath"):
        ps.partialShortcutPath(path)


def _path_joint_length(space, path, nq):
    """Total joint-space length of an OMPL path (lower = cleaner/shorter)."""
    try:
        arr = _path_to_numpy(space, path, nq)
    except Exception:
        return float("inf")
    if arr is None or len(arr) < 2:
        return float("inf")
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def _solution_path_candidates(space, ss, mode, nq, residual_tol, simplify):
    path = ss.getSolutionPath()
    base = _copy_path(path) or path
    candidates = [base]

    if not simplify:
        return candidates

    try:
        ps = ss.getPathSimplifier()
    except Exception:
        return candidates

    # Simplify in ONE bulk pass, then certify ONCE.  Certifying after every
    # shortcut round densifies and residual-checks the whole path each time,
    # which is pathologically slow on large atlas paths (it can hang for
    # minutes).  Doing the cleanup first and verifying once is fast and gives
    # the same shortened result; if the simplified path fails certification we
    # simply keep the raw planner path.
    work = _copy_path(base)
    if work is None:
        return candidates
    try:
        ps.reduceVertices(work)
        for _ in range(5):
            _run_shortcut(ps, work)
        ps.smoothBSpline(work)
    except Exception:
        return candidates

    try:
        ok, _, _ = _certifies(mode, _path_to_numpy(space, work, nq), residual_tol)
    except Exception:
        ok = False

    if ok:
        candidates.insert(0, work)
    return candidates


def plan_in_mode(robot, mode, q_start, q_goal,
                 planner_name="RRTConnect",
                 solve_time=5.0,
                 range_value=None,
                 delta=0.2,
                 residual_tol=1e-5,
                 max_constraint_iters=50,
                 simplify=False,
                 interpolate_points=200,
                 project_endpoints=True,
                 space="projected"):
    """Plan q_start -> q_goal staying on `mode`'s manifold and feasible region."""
    nq = int(robot.nq)
    q_start = np.asarray(q_start, float).reshape(nq)
    q_goal = np.asarray(q_goal, float).reshape(nq)
    res = PlanResult(success=False, path=None, message="", planner=planner_name)

    if ob is None or og is None:
        res.message = "OMPL Python bindings not available."
        return res

    if project_endpoints:
        q_start, ok_s = mode.project(q_start)
        q_goal, ok_g = mode.project(q_goal)
        if not (ok_s and ok_g):
            res.message = "Endpoint projection onto the mode manifold failed."
            return res
    if not (mode.feasible(q_start) and mode.feasible(q_goal)):
        res.message = "Endpoint(s) violate the mode's inequality constraints."
        return res

    if space not in ("projected", "atlas"):
        res.message = f"Unknown constrained space '{space}'. Use 'projected' or 'atlas'."
        return res

    def solve_with(requested_space):
        out = PlanResult(success=False, path=None, message="", planner=planner_name,
                         constrained_space=requested_space)
        qlo, qhi = robot.joint_limits
        ambient = ob.RealVectorStateSpace(nq)
        bounds = ob.RealVectorBounds(nq)
        for i in range(nq):
            bounds.setLow(i, float(qlo[i]))
            bounds.setHigh(i, float(qhi[i]))
        ambient.setBounds(bounds)

        codim = mode.codim(q_start)

        if codim == 0:
            ss = og.SimpleSetup(ambient)
            si = ss.getSpaceInformation()
            alloc = ambient
            out.constrained_space = requested_space
            css = None
        else:
            constraint = _ModeConstraint(mode, nq, codim)
            if hasattr(constraint, "setTolerance"):
                constraint.setTolerance(max(float(residual_tol), 1e-8))
            if hasattr(constraint, "setMaxIterations"):
                constraint.setMaxIterations(int(max_constraint_iters))
            if requested_space == "atlas":
                css = ob.AtlasStateSpace(ambient, constraint)
            else:
                css = ob.ProjectedStateSpace(ambient, constraint)
            csi = ob.ConstrainedSpaceInformation(css)
            css.setSpaceInformation(csi)
            if hasattr(css, "setDelta"):
                css.setDelta(float(delta))
            css.setup()
            ss = og.SimpleSetup(csi)
            si = csi
            alloc = css

        def is_valid(state):
            return bool(mode.feasible(_q(alloc, state, nq)))

        _set_validity_checker(ss, si, is_valid)

        # Build start/goal states.  Old OMPL returns ScopedState; OMPL 2.x
        # returns raw states from allocState().  SimpleSetup accepts both in
        # supported bindings.
        start = _make_state(alloc, q_start, nq)
        goal = _make_state(alloc, q_goal, nq)
        if requested_space == "atlas" and css is not None:
            try:
                css.anchorChart(_raw_state(start))
                css.anchorChart(_raw_state(goal))
            except Exception:
                pass
        ss.setStartAndGoalStates(start, goal)

        planner_cls = getattr(og, planner_name, None)
        if planner_cls is None:
            out.message = f"OMPL planner '{planner_name}' unavailable."
            return out
        planner = planner_cls(si)
        if range_value is not None and hasattr(planner, "setRange"):
            planner.setRange(float(range_value))
        ss.setPlanner(planner)
        try:
            ss.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))
        except Exception:
            try:
                ss.getProblemDefinition().setOptimizationObjective(
                    ob.PathLengthOptimizationObjective(si)
                )
            except Exception:
                pass
        ss.setup()
        try:
            si.setStateValidityCheckingResolution(0.002)
        except Exception:
            pass

        tic = time.perf_counter()
        status = ss.solve(float(solve_time))
        out.solve_time = time.perf_counter() - tic
        if not ss.haveExactSolutionPath():
            out.message = f"{planner_name} found no exact path ({status})."
            return out

        path = ss.getSolutionPath()
        out.raw_points = path.getStateCount()
        candidates = _solution_path_candidates(alloc, ss, mode, nq, residual_tol, simplify)
        q_path = None
        ok = False
        max_res = float("nan")
        feasible = False
        for candidate in candidates:
            path = _copy_path(candidate) or candidate
            if int(interpolate_points) > 1:
                try:
                    path.interpolate(int(interpolate_points))
                except TypeError:
                    path.interpolate()
            q_candidate = _path_to_numpy(alloc, path, nq)
            ok, max_res, feasible = _certifies(mode, q_candidate, residual_tol)
            q_path = q_candidate
            if ok:
                break
        out.path_points = 0 if q_path is None else len(q_path)
        out.max_equality_residual = max_res
        out.feasible_along_path = feasible
        out.success = bool(ok)
        out.path = q_path if ok else None
        out.message = (f"{planner_name} solved (max residual {max_res:.2e}, "
                       f"feasible={feasible})." if ok else
                       f"{planner_name} path failed certification "
                       f"(max residual {max_res:.2e}, feasible={feasible}).")
        return out

    try:
        return solve_with(space)
    except Exception as exc:
        res.constrained_space = space
        res.message = f"OMPL {space} backend failed: {exc}"
        return res


__all__ = ["PlanResult", "plan_in_mode"]