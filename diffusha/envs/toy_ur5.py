import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy import linalg
import math as m
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

class UR5Env(gym.Env):
    def __init__(self, control_frequency=70.0):
        super(UR5Env, self).__init__()
        
        self.control_frequency = control_frequency  # Hz
        self.num_joints = 6

        # UR5 link parameters
        self.home_position, self.screw_axes = self.ur5_params()
        self.joint_limits = np.array([
            [-2 * np.pi, 2 * np.pi],  # Joint 1 limits
            [-2 * np.pi, 2 * np.pi],  # Joint 2 limits
            [-2 * np.pi, 2 * np.pi],  # Joint 3 limits
            [-2 * np.pi, 2 * np.pi],  # Joint 4 limits
            [-2 * np.pi, 2 * np.pi],  # Joint 5 limits
            [-2 * np.pi, 2 * np.pi],  # Joint 6 limits
        ])

        # Joint state and end effector state
        self.joint_state = np.zeros(self.num_joints)
        self.end_effector_position = np.zeros(3)
        self.end_effector_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion

        # Action and observation spaces
        self.action_space = spaces.Box(low=np.array([-0.1] * 6), high=np.array([0.1] * 6), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-np.inf] * 13), high=np.array([np.inf] * 13), dtype=np.float32)

        # Update end effector pose
        self._update_end_effector()

    def ur5_params(self):
        """
        Returns
        -------
        home_position : Home position of the UR5 with DH parameters
        screw_axes : Screw axes for UR5 with PoE parameters
        """
        # UR5 link parameters (in meters)
        l1 = 0.425
        l2 = 0.392
        h1 = 0.089159
        h2 = 0.09465
        w1 = 0.10915
        w2 = 0.0823

        # Home position (M matrix)
        home_position = np.array([
            [0, -1, 0, l1 + l2],
            [0, 0, -1, w1 + w2],
            [1, 0, 0, h1 - h2],
            [0, 0, 0, 1]
        ])

        # Screw axes (in the space frame when at home position)
        screw_axes = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, -h1, 0, 0],
            [0, 1, 0, -h1, 0, l1],
            [0, 1, 0, -h1, 0, l1 + l2],
            [0, 0, -1, 0, l1 + l2, 0],
            [0, 1, 0, h2 - h1, 0, l1 + l2]
        ]).T  # Transpose to get each column as a screw axis

        return home_position, screw_axes

    def skew(self, vector):
        """
        Returns the skew-symmetric matrix of a vector.
        """
        w1, w2, w3 = vector
        return np.array([
            [0, -w3, w2],
            [w3, 0, -w1],
            [-w2, w1, 0]
        ])

    def matrix_exp6(self, screw_axis, theta):
        """
        Computes the matrix exponential for a screw axis.
        """
        omega = screw_axis[:3]
        v = screw_axis[3:]

        omega_norm = np.linalg.norm(omega)
        if omega_norm < 1e-8:
            # Pure translation
            return np.vstack((
                np.hstack((np.eye(3), (v * theta).reshape((3,1)))),
                np.array([0, 0, 0, 1])
            ))
        else:
            omega_hat = self.skew(omega / omega_norm)
            theta_omega = omega_norm * theta
            R = np.eye(3) + np.sin(theta_omega) * omega_hat + (1 - np.cos(theta_omega)) * omega_hat @ omega_hat
            G = (np.eye(3) * theta) + (1 - np.cos(theta_omega)) * omega_hat + (theta_omega - np.sin(theta_omega)) * omega_hat @ omega_hat
            p = G @ (v / omega_norm)
            return np.vstack((
                np.hstack((R, p.reshape((3,1)))),
                np.array([0, 0, 0, 1])
            ))

    def forward_kinematics_poe(self, joint_angles):
        """
        Compute forward kinematics using the Product of Exponentials formula.
        """
        T = np.eye(4)
        for i in range(self.num_joints):
            screw_axis = self.screw_axes[:, i]
            theta = joint_angles[i]
            exp6 = self.matrix_exp6(screw_axis, theta)
            T = T @ exp6
        T = T @ self.home_position
        position = T[:3, 3]
        orientation = R.from_matrix(T[:3, :3]).as_quat()
        return position, orientation

    def compute_jacobian_body(self, joint_angles):
        """
        Compute the body Jacobian for the UR5 robot.
        """
        T = self.home_position.copy()
        Jb = np.zeros((6, self.num_joints))
        for i in reversed(range(self.num_joints)):
            screw_axis = self.screw_axes[:, i]
            T_i = np.eye(4)
            for j in range(i + 1, self.num_joints):
                T_i = T_i @ self.matrix_exp6(-self.screw_axes[:, j], joint_angles[j])
            Ad_T_i = Adjoint(T_i)
            Jb[:, i] = Ad_T_i @ screw_axis
        return Jb

    def inverse_kinematics_poe(self, desired_pos, desired_rot, initial_guess=None, max_iterations=1000, tolerance=1e-5):
        """
        Compute inverse kinematics using the Newton-Raphson method with Damped Least Squares.
        """
        if initial_guess is None:
            joint_angles = np.zeros(self.num_joints)
        else:
            joint_angles = initial_guess.copy()

        desired_T = np.eye(4)
        desired_T[:3, :3] = desired_rot
        desired_T[:3, 3] = desired_pos

        for i in range(max_iterations):
            current_T = self.get_transform_poe(joint_angles)
            error_T = np.linalg.inv(current_T) @ desired_T
            error_se3 = MatrixLog6(error_T)
            error_twist = se3ToVec(error_se3)
            error_norm = np.linalg.norm(error_twist)
            if error_norm < tolerance:
                return joint_angles
            jacobian = self.compute_jacobian_body(joint_angles)
            # Damped Least Squares
            damping_factor = 1e-6
            JTJ = jacobian.T @ jacobian + damping_factor * np.eye(self.num_joints)
            delta_theta = np.linalg.solve(JTJ, jacobian.T @ error_twist)
            joint_angles = joint_angles + delta_theta
            # Enforce joint limits
            joint_angles = np.clip(joint_angles, self.joint_limits[:, 0], self.joint_limits[:, 1])
        raise ValueError("Inverse kinematics did not converge")

    def get_transform_poe(self, joint_angles):
        """
        Compute the transformation matrix using PoE.
        """
        T = np.eye(4)
        for i in range(self.num_joints):
            screw_axis = self.screw_axes[:, i]
            theta = joint_angles[i]
            exp6 = self.matrix_exp6(screw_axis, theta)
            T = T @ exp6
        T = T @ self.home_position
        return T

    def reset(self):
        """Reset the environment."""
        self.joint_state = np.array([0, -np.pi / 2, 0, -np.pi / 2, 0, 0])  # Home position
        self._update_end_effector()
        return self._get_observation()

    def step(self, action):
        """
        Perform one step in the environment.
        The action specifies joint angle increments.
        """
        # Update joint angles
        self.joint_state += action
        # Enforce joint limits
        self.joint_state = np.clip(self.joint_state, self.joint_limits[:, 0], self.joint_limits[:, 1])

        self._update_end_effector()

        # Example reward: minimize distance to a target (e.g., origin)
        target_position = np.array([0.5, 0, 0.5])
        distance = np.linalg.norm(self.end_effector_position - target_position)
        reward = -distance
        done = False
        return self._get_observation(), reward, done, {}

    def _update_end_effector(self):
        """Update the end effector state."""
        self.end_effector_position, self.end_effector_orientation = self.forward_kinematics_poe(self.joint_state)

    def _get_observation(self):
        """Return the current state as an observation."""
        return np.concatenate([self.joint_state, self.end_effector_position, self.end_effector_orientation])

    def debug_fk_vs_ik(self):
        """
        Verify FK and IK consistency. FK -> IK -> FK should return the same result.
        """
        # Compute FK for the current joint state
        position, orientation = self.forward_kinematics_poe(self.joint_state)

        # Compute IK for the FK output
        try:
            desired_rot = R.from_quat(orientation).as_matrix()
            ik_joint_angles = self.inverse_kinematics_poe(position, desired_rot, initial_guess=self.joint_state)
            recovered_position, recovered_orientation = self.forward_kinematics_poe(ik_joint_angles)

            # Debug outputs
            print("FK Debug - Original Position:", position)
            print("FK Debug - Original Orientation:", orientation)
            print("FK Debug - Recovered Position:", recovered_position)
            print("FK Debug - Recovered Orientation:", recovered_orientation)

            # Assertions to verify consistency
            assert np.allclose(position, recovered_position, atol=1e-4), "FK and IK positions do not match!"
            assert np.allclose(orientation, recovered_orientation, atol=1e-4), "FK and IK orientations do not match!"

            print("Debug successful: FK and IK are consistent.")
        except ValueError as e:
            print("IK failed:", str(e))

    def render(self):
        """Simple 3D visualization using matplotlib."""
        positions = [np.zeros(3)]  # Base position
        T = np.eye(4)
        for i in range(self.num_joints):
            screw_axis = self.screw_axes[:, i]
            theta = self.joint_state[i]
            exp6 = self.matrix_exp6(screw_axis, theta)
            T = T @ exp6
            positions.append(T[:3, 3])
        positions = np.array(positions)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker="o")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    def close(self):
        pass

# Helper functions for PoE method

def skew(vector):
    """
    Returns the skew-symmetric matrix of a vector.
    """
    w1, w2, w3 = vector
    return np.array([
        [0, -w3, w2],
        [w3, 0, -w1],
        [-w2, w1, 0]
    ])

def MatrixLog3(R):
    """
    Compute the matrix logarithm of a rotation matrix R.
    """
    cos_theta = (np.trace(R) - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Clamp to handle numerical errors
    theta = np.arccos(cos_theta)
    if np.isclose(theta, 0):
        return np.zeros((3,3))
    else:
        omega_hat = (R - R.T) / (2 * np.sin(theta)) * theta
        return omega_hat

def MatrixLog6(T):
    """
    Compute the matrix logarithm of a homogeneous transformation matrix T.
    """
    R = T[:3, :3]
    p = T[:3, 3]
    omega_hat = MatrixLog3(R)
    if np.allclose(omega_hat, np.zeros((3,3)), atol=1e-6):
        # Pure translation
        v = p
        se3mat = np.zeros((4,4))
        se3mat[:3, 3] = v
        return se3mat
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        theta = np.clip(theta, -np.pi, np.pi)
        omega = np.array([omega_hat[2,1], omega_hat[0,2], omega_hat[1,0]]) / theta
        omega_hat_normalized = omega_hat / theta
        G_inv = (1 / theta) * np.eye(3) - 0.5 * omega_hat_normalized + \
                (1 / theta - 0.5 / np.tan(theta / 2.0)) * (omega_hat_normalized @ omega_hat_normalized)
        v = G_inv @ p
        se3mat = np.zeros((4,4))
        se3mat[:3, :3] = omega_hat
        se3mat[:3, 3] = v
        return se3mat

def se3ToVec(se3mat):
    """
    Convert an se3 matrix to a 6D vector (twist).
    """
    omega_hat = se3mat[:3, :3]
    v = se3mat[:3, 3]
    omega = np.array([omega_hat[2,1], omega_hat[0,2], omega_hat[1,0]])
    return np.concatenate((omega, v))

def Adjoint(T):
    """
    Compute the adjoint representation of transformation matrix T.
    """
    R = T[:3, :3]
    p = T[:3, 3]
    p_hat = skew(p)
    Ad_T = np.zeros((6,6))
    Ad_T[:3, :3] = R
    Ad_T[3:, :3] = p_hat @ R
    Ad_T[3:, 3:] = R
    return Ad_T