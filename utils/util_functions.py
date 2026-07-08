
import pybullet as p
import numpy as np
from math import pi
from scipy.spatial.transform import Rotation as R

def quaternion_slerp(q1, q2, t):
    """Spherical linear interpolation for quaternions (PyBullet [x,y,z,w] format)"""
    q1 = np.array(q1)
    q2 = np.array(q2)
    # Normalize quaternions
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    # Convert to w-first format for calculations
    q1 = np.array([q1[3], q1[0], q1[1], q1[2]])
    q2 = np.array([q2[3], q2[0], q2[1], q2[2]])

    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        result /= np.linalg.norm(result)
    else:
        theta_0 = np.arccos(dot)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0
        result = (s1 * q1 + s2 * q2)
    # Convert back to PyBullet format
    result = np.array([result[1], result[2], result[3], result[0]])
    return result

# def get_poc_pose(robot, elbow_index=9, wrist_index=12, theta=-np.pi/2):
#     """
#     Calculate the point of contact (PoC) and its orientation between two links.

#     Args:
#         robot: The robot object.
#         elbow_index: Index of the elbow link.
#         wrist_index: Index of the wrist link.
#         theta: Rotation angle around the y-axis (default: -pi/2).

#     Returns:
#         tuple: (poc_pos, poc_quat), where poc_pos is the position of the PoC and
#                poc_quat is the quaternion representing its orientation.
#     """
#     # Get the states of the elbow and wrist links
#     elbow_state = p.getLinkState(robot, elbow_index)  # elbow link
#     wrist_state = p.getLinkState(robot, wrist_index)  # wrist link

#     # Calculate the point of contact (midpoint between elbow and wrist)
#     poc_pos = tuple((e + w) / 2 for e, w in zip(elbow_state[0], wrist_state[0]))

#     # Get the rotation matrix from the wrist quaternion
#     pos_rot_matrix = p.getMatrixFromQuaternion(wrist_state[1])

#     # Rotate theta around the y-axis
#     rot_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
#     poc_rot_matrix = np.array(pos_rot_matrix).reshape(3, 3) @ rot_y

#     poc_quat = R.from_matrix(poc_rot_matrix).as_quat()

#     return poc_pos, poc_quat

def get_poc_pose(robot, forearm_index=10, dist_from_forearm=0.0, forearm_dim=0.0):
    """
    Calculate the point of contact (PoC) and its orientation between two links.

    Args:
        robot: The robot object.
        forearm_index: Index of the forearm link.
        dist_from_forearm: Distance from the forearm to the point of contact (default: 0.12).
        forearm_dim: Dimension of the forearm (default: 0.05).
        theta: Rotation angle around the y-axis (default: -pi/2).

    Returns:
        tuple: (poc_pos, poc_quat), where poc_pos is the position of the PoC and
               poc_quat is the quaternion representing its orientation.
    """
    # Get the states of the forearm and wrist links
    forearm_state = p.getLinkState(robot, forearm_index)  # Link=RightForeArm
    forearm_pos, forearm_quat = forearm_state[0], forearm_state[1]

    # Calculate the point of contact pose (above forearm: in forearm y-axis direction)
    forearm_poc_trans = np.array([0, forearm_dim, dist_from_forearm])
    # wrist_poc_quat = [0, 0, 0, 1] # Identity quaternion
    forearm_poc_quat = p.getQuaternionFromEuler([0, -pi/2, pi/2])
    poc_pos, poc_quat = p.multiplyTransforms(forearm_pos, forearm_quat, forearm_poc_trans, forearm_poc_quat)

    return np.concatenate([poc_pos, poc_quat])   #list(poc_pos) + list(poc_quat)
