"""PyBullet playback for the saved Panda payload transfer.

The renderer loads the mesh-only Franka hand URDF and the already planned
``panda_plan.npz`` trajectory.  It does not load or draw a payload box; the
payload dynamics remain only in the planner's Pinocchio URDF.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

try:
    import pybullet as p
    import pybullet_data
except Exception as exc:  # import-safe diagnostics
    p = None
    pybullet_data = None
    _PYBULLET_IMPORT_ERROR = exc
else:
    _PYBULLET_IMPORT_ERROR = None


HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / "panda_plan.npz"
TOPPFAU_RENDER_URDF = HERE / "TOPPFAU_GOFA5" / "robot-assets" / "model" / "urdf" / "fer_franka_hand.urdf"
LOCAL_RENDER_URDF = HERE / "fer_franka_hand_render.urdf"
ASSET_ROOT = HERE / "franka_description"
EE_LINK = "fer_hand_tcp"


def _connect():
    if p is None:
        print(f"PyBullet is not available: {_PYBULLET_IMPORT_ERROR}")
        return None, False
    try:
        cid = p.connect(p.GUI)
        if cid >= 0:
            return cid, True
    except Exception as exc:
        print(f"PyBullet GUI connection failed: {exc}")
    cid = p.connect(p.DIRECT)
    if cid >= 0:
        print("PyBullet GUI unavailable; running DIRECT load/step diagnostics.")
        return cid, False
    print("PyBullet connection failed.")
    return None, False


def _setup_search_paths(client_id, urdf_path: Path):
    if pybullet_data is not None:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
    model_root = urdf_path.parent.parent
    paths = [model_root, ASSET_ROOT, urdf_path.parent]
    for env in ("ROBOT_ASSETS_ROOT", "TOPPFAU_GOFA5"):
        value = os.environ.get(env)
        if value:
            root = Path(value)
            paths.extend([root, root / "robot-assets" / "model"])
    primary = None
    for path in paths:
        if path.exists():
            if primary is None and (path / "meshes").exists():
                primary = path
        else:
            print(f"asset search path not found: {path}")
    if primary is None:
        for path in paths:
            if path.exists():
                primary = path
                break
    if primary is not None:
        p.setAdditionalSearchPath(str(primary), physicsClientId=client_id)
        print(f"PyBullet asset search path: {primary}")


def _load_plan(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing plan file: {path}")
    plan = np.load(path, allow_pickle=False)
    required = ["dense", "ee_xyz", "tableA", "tableB", "hA", "hB", "lift_off", "set_down"]
    missing = [key for key in required if key not in plan.files]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    return {
        "dense": np.asarray(plan["dense"], dtype=float).reshape((-1, 7)),
        "ee_xyz": np.asarray(plan["ee_xyz"], dtype=float).reshape((-1, 3)),
        "tableA": np.asarray(plan["tableA"], dtype=float).reshape((2, 2)),
        "tableB": np.asarray(plan["tableB"], dtype=float).reshape((2, 2)),
        "hA": float(np.asarray(plan["hA"]).item()),
        "hB": float(np.asarray(plan["hB"]).item()),
        "lift_off": np.asarray(plan["lift_off"], dtype=float).reshape(7),
        "set_down": np.asarray(plan["set_down"], dtype=float).reshape(7),
    }


def _render_urdf() -> Path:
    env_path = os.environ.get("PYBULLET_RENDER_URDF")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([TOPPFAU_RENDER_URDF, LOCAL_RENDER_URDF])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    print("No render URDF found. Paths checked:")
    for candidate in candidates:
        print(f"  {candidate}")
    raise FileNotFoundError("render URDF not found")


def _movable_revolute_joints(body_id, client_id):
    joints = []
    for i in range(p.getNumJoints(body_id, physicsClientId=client_id)):
        info = p.getJointInfo(body_id, i, physicsClientId=client_id)
        if int(info[2]) == p.JOINT_REVOLUTE:
            joints.append(i)
    if len(joints) < 7:
        raise RuntimeError(f"Expected at least 7 revolute joints, found {len(joints)}")
    if len(joints) > 7:
        print(f"Found {len(joints)} revolute joints; using the first 7.")
    return joints[:7]


def _link_index(body_id, link_name: str, client_id):
    wanted = link_name.encode("utf-8")
    for i in range(p.getNumJoints(body_id, physicsClientId=client_id)):
        info = p.getJointInfo(body_id, i, physicsClientId=client_id)
        if info[12] == wanted:
            return i
    print(f"link '{link_name}' not found; transition markers will use saved ee_xyz endpoints only.")
    return None


def _reset_q(body_id, joints, q, client_id):
    q = np.asarray(q, dtype=float).reshape(7)
    for joint, value in zip(joints, q):
        p.resetJointState(body_id, joint, float(value), physicsClientId=client_id)


def _frame_position(body_id, link_idx, client_id):
    if link_idx is None:
        return None
    state = p.getLinkState(body_id, link_idx, computeForwardKinematics=True,
                           physicsClientId=client_id)
    return np.asarray(state[4], dtype=float)


def _make_table(bounds, height, color, client_id):
    (xmin, ymin), (xmax, ymax) = np.asarray(bounds, dtype=float).reshape(2, 2)
    thickness = 0.02
    half = [0.5 * (xmax - xmin), 0.5 * (ymax - ymin), 0.5 * thickness]
    center = [0.5 * (xmin + xmax), 0.5 * (ymin + ymax), float(height) - 0.5 * thickness]
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half,
        rgbaColor=color,
        physicsClientId=client_id,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual,
        basePosition=center,
        physicsClientId=client_id,
    )


def _make_marker(position, color, radius, client_id):
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=float(radius),
        rgbaColor=color,
        physicsClientId=client_id,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual,
        basePosition=np.asarray(position, dtype=float).reshape(3).tolist(),
        physicsClientId=client_id,
    )


def main() -> int:
    print("Run commands:")
    print(f"  python -B {HERE / 'run_panda.py'}")
    print(f"  python -B {HERE / 'animate_pybullet.py'}")
    print("Requires an environment with pybullet and a display (native Windows or WSLg).")

    client_id, has_gui = _connect()
    if client_id is None:
        return 1

    try:
        plan = _load_plan(PLAN_PATH)
        urdf_path = _render_urdf().resolve()
        _setup_search_paths(client_id, urdf_path)
        p.resetSimulation(physicsClientId=client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=client_id)

        body = p.loadURDF(
            str(urdf_path),
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
            physicsClientId=client_id,
        )
        joints = _movable_revolute_joints(body, client_id)
        print(f"loaded render URDF: {urdf_path}")
        print(f"using revolute joints: {joints}")

        _make_table(plan["tableA"], plan["hA"], [0.18, 0.62, 0.32, 1.0], client_id)
        _make_table(plan["tableB"], plan["hB"], [0.20, 0.38, 0.75, 1.0], client_id)

        ee_link = _link_index(body, EE_LINK, client_id)
        _reset_q(body, joints, plan["lift_off"], client_id)
        lift_pos = _frame_position(body, ee_link, client_id)
        _reset_q(body, joints, plan["set_down"], client_id)
        set_pos = _frame_position(body, ee_link, client_id)
        _reset_q(body, joints, plan["dense"][0], client_id)

        _make_marker(plan["ee_xyz"][0], [0.0, 0.90, 1.0, 1.0], 0.018, client_id)
        _make_marker(plan["ee_xyz"][-1], [1.0, 0.0, 0.0, 1.0], 0.018, client_id)
        _make_marker(lift_pos if lift_pos is not None else plan["ee_xyz"][0],
                     [1.0, 0.85, 0.0, 1.0], 0.018, client_id)
        _make_marker(set_pos if set_pos is not None else plan["ee_xyz"][-1],
                     [1.0, 0.0, 1.0, 1.0], 0.018, client_id)

        if has_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.15,
                cameraYaw=-45,
                cameraPitch=-28,
                cameraTargetPosition=[0.48, 0.0, 0.42],
                physicsClientId=client_id,
            )

        print("Animating. Close the PyBullet window or press Ctrl+C to stop.")
        dt = 1.0 / 30.0
        while True:
            p.removeAllUserDebugItems(physicsClientId=client_id)
            prev = None
            for q, ee in zip(plan["dense"], plan["ee_xyz"]):
                if has_gui and not p.isConnected(client_id):
                    return 0
                _reset_q(body, joints, q, client_id)
                ee = np.asarray(ee, dtype=float)
                if prev is not None:
                    p.addUserDebugLine(
                        prev.tolist(),
                        ee.tolist(),
                        lineColorRGB=[1.0, 0.45, 0.0],
                        lineWidth=4,
                        lifeTime=0.0,
                        physicsClientId=client_id,
                    )
                prev = ee
                p.stepSimulation(physicsClientId=client_id)
                time.sleep(dt)
    except KeyboardInterrupt:
        print("Animation stopped.")
        return 0
    except Exception as exc:
        print(f"PyBullet animation failed: {exc}")
        return 1
    finally:
        if p is not None and p.isConnected(client_id):
            p.disconnect(physicsClientId=client_id)


if __name__ == "__main__":
    raise SystemExit(main())
