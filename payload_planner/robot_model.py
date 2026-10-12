"""
robot_model.py
==============
Thin wrapper around Pinocchio that gives us, for any joint configuration q:

    * frame pose       T(q)            (where the grasped object / tool is)
    * frame Jacobian   J(q)            (how that pose moves with the joints)
    * dynamics         M(q), c(q,dq), g(q)

The single most important quantity for the payload work is g(q): the
gravity-holding torque.  Because the payload is baked into the URDF as a rigid
link at the gripper, g(q) ALREADY includes the load -- we never compute a
separate payload torque by hand.  Nothing here depends on PyBullet.

The RobotModel class itself is the max's code, kept as it is; the helper
methods at the bottom are the only additions.
"""

import numpy as np
import pinocchio


def skew(v):
    """Skew-symmetric matrix of a 3-vector."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


class RobotModel:
    def __init__(self, urdf_file,
                 p0=np.zeros(3),
                 R0=np.eye(3),
                 Tgp=np.eye(4)):
        self.pin_model = pinocchio.buildModelFromUrdf(urdf_file)
        self.pin_data = self.pin_model.createData()

        self.nq = self.pin_model.nq

        self.p0 = p0
        self.R0 = R0
        self.T0 = pinocchio.SE3(R0, p0)
        self.Tgp = pinocchio.SE3(Tgp[:3, :3], Tgp[:3, 3])
        self.Ggp = np.eye(6)

    def update_kinematics(self, frame_name, q, dq) -> tuple:
        pinocchio.forwardKinematics(self.pin_model, self.pin_data, q, dq)
        pinocchio.updateFramePlacements(self.pin_model, self.pin_data)

        frame_id = self.pin_model.getFrameId(frame_name)
        T = self.pin_data.oMf[frame_id]
        T = self.T0 * T                                   # apply base transform

        J = pinocchio.computeFrameJacobian(
            self.pin_model, self.pin_data, q, frame_id,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = np.vstack((J[0:3, :], self.R0 @ J[3:6, :]))   # base rot on angular part

        # transform to the grasp frame
        self.Ggp[0:3, 3:6] = -skew(T.rotation @ self.Tgp.translation)
        J = self.Ggp @ J
        T = T * self.Tgp
        return T, J

    def update_frame_kinematics(self, frame_name, q, dq, ddq) -> tuple:
        pinocchio.forwardKinematics(self.pin_model, self.pin_data, q, dq, ddq)
        pinocchio.updateFramePlacements(self.pin_model, self.pin_data)

        frame_id = self.pin_model.getFrameId(frame_name)
        T = self.pin_data.oMf[frame_id]
        T = self.T0 * T

        v = pinocchio.getFrameVelocity(self.pin_model, self.pin_data, frame_id,
                                       pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        a = pinocchio.getFrameAcceleration(self.pin_model, self.pin_data, frame_id,
                                           pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        return T, v, a

    def update_dynamics(self, q, dq) -> tuple:
        # NOTE gravity is assumed [0,0,-9.81] in the base frame.
        M = pinocchio.crba(self.pin_model, self.pin_data, q)
        M = (M + M.T) / 2.0
        pinocchio.nonLinearEffects(self.pin_model, self.pin_data, q, dq)
        n = np.copy(self.pin_data.nle)
        g = pinocchio.nonLinearEffects(self.pin_model, self.pin_data, q,
                                       np.zeros(self.pin_model.nv))
        c = n - g
        return M, c, g

    # ------------------------------------------------------------------ #
    # convenience helpers 
    # ------------------------------------------------------------------ #
    def gravity(self, q) -> np.ndarray:
        """Gravity-holding joint torque g(q) -- includes the payload link."""
        return pinocchio.computeGeneralizedGravity(self.pin_model, self.pin_data,
                                                    np.asarray(q, float))

    def frame_pose(self, frame_name, q) -> np.ndarray:
        """4x4 homogeneous pose of a frame at configuration q."""
        T, _ = self.update_kinematics(frame_name, np.asarray(q, float),
                                      np.zeros(self.nq))
        return T.homogeneous

    def frame_jacobian(self, frame_name, q) -> np.ndarray:
        """6 x nq frame Jacobian (rows 0:3 linear, 3:6 angular)."""
        _, J = self.update_kinematics(frame_name, np.asarray(q, float),
                                      np.zeros(self.nq))
        return J

    @property
    def joint_limits(self):
        return self.pin_model.lowerPositionLimit, self.pin_model.upperPositionLimit