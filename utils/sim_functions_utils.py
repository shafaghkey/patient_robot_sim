"""sim_functions_utils

PyBullet simulation helper functions shared across robot and human simulation modules.

Key utilities:
  quat_diff / add_pose_offset_xyz_quat — quaternion and pose arithmetic
  quaternion_slerp                     — spherical linear interpolation
  DS_linear_direction / DS_angular_direction — dynamical-system velocity directions
  update_damping_matrix_task/role/joint — passive DS damping matrix constructors
  get_poc_pose                         — point-of-contact pose on a forearm body
  disable_default_motors / apply_cmd_torque — PyBullet joint control helpers
"""
import pybullet as pb
import numpy as np
from math import pi
from scipy.spatial.transform import Rotation as R

#############################################################
######################  Helper functions ####################
#############################################################
def quat_diff(q_target, q_current):
    """Return small-angle 3D error vector between two quaternions (x,y,z,w)."""
    # Convert to numpy
    qt = np.array(q_target, dtype=np.float64)
    qc = np.array(q_current, dtype=np.float64)
    # Normalize to avoid drift
    qt /= np.linalg.norm(qt) + 1e-12
    qc /= np.linalg.norm(qc) + 1e-12
    # Relative rotation r = qt * qc^{-1}
    qc_conj = np.array([-qc[0], -qc[1], -qc[2], qc[3]])
    r = np.array([
        qt[3]*qc_conj[0] + qt[0]*qc_conj[3] + qt[1]*qc_conj[2] - qt[2]*qc_conj[1],
        qt[3]*qc_conj[1] - qt[0]*qc_conj[2] + qt[1]*qc_conj[3] + qt[2]*qc_conj[0],
        qt[3]*qc_conj[2] + qt[0]*qc_conj[1] - qt[1]*qc_conj[0] + qt[2]*qc_conj[3],
        qt[3]*qc_conj[3] - qt[0]*qc_conj[0] - qt[1]*qc_conj[1] - qt[2]*qc_conj[2],
    ])
    r /= np.linalg.norm(r) + 1e-12
    angle = 2 * np.arccos(np.clip(r[3], -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = r[:3] / (np.sin(angle/2.0) + 1e-12)
    return angle * axis

def add_pose_offset_xyz_quat(pose7, offset_xyz):
    """pose7: [x,y,z,qx,qy,qz,qw]; offset_xyz additive in world frame."""
    out = pose7.copy()
    out[:3] = (np.array(out[:3]) + np.array(offset_xyz)).tolist()
    return out

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

#############################################################
#######################  DS functions #######################
#############################################################

def DS_linear_direction(pos1, pos2):
    '''function to compute desired linear velocity direction from pos1 to pos2'''
    delta_pos = np.array(pos2) - np.array(pos1)
    if np.linalg.norm(delta_pos) < 1e-6:
        return np.zeros(3)
    ds_vel_lin = delta_pos #/ np.linalg.norm(delta_pos)
    return ds_vel_lin

def DS_angular_direction(q1, q2):
    '''deltaQ ⨂ q_conj'''
    dqd = quaternion_slerp(q1, q2, t=0.5)
    # Convert to w-first format for calculations
    ee_quat_wxyz = np.array([q1[3], q1[0], q1[1], q1[2]])
    dqd_wxyz = np.array([dqd[3], dqd[0], dqd[1], dqd[2]])
    deltaQ = dqd_wxyz - ee_quat_wxyz
    q_conj = np.array([ee_quat_wxyz[0], -ee_quat_wxyz[1], -ee_quat_wxyz[2], -ee_quat_wxyz[3]])
    # Quaternion product (deltaQ ⨂ q_conj)
    w = deltaQ[0]*q_conj[0] - np.dot(deltaQ[1:], q_conj[1:])
    xyz = deltaQ[0]*q_conj[1:] + q_conj[0]*deltaQ[1:] + np.cross(deltaQ[1:], q_conj[1:])
    temp_angVel = np.concatenate(([w], xyz))
    tmp_angular_vel = temp_angVel[1:]  # Use vector part of quaternion
    # Velocity limiting
    maxDq = 0.2
    if np.linalg.norm(tmp_angular_vel) > maxDq:
        tmp_angular_vel = (tmp_angular_vel / np.linalg.norm(tmp_angular_vel)) * maxDq
    # Nonlinear scaling factor
    theta_gq = (-0.5 / (4 * maxDq**2)) * np.dot(tmp_angular_vel, tmp_angular_vel)
    # Final desired angular velocity (ds)
    ds_vel_ang = 2 * (1 + np.exp(theta_gq)) * tmp_angular_vel
    return ds_vel_ang

def update_damping_matrix_task(ds_vel, damping_eigval):
        '''function to update the damping matrix
           first column toward ds_vel direction'''
        if np.linalg.norm(ds_vel) < 1e-6:
            return np.eye(3)
        damping_eigval = np.diag(damping_eigval)
        base_mat = np.zeros((3, 3))
        base_mat[:,0] = ds_vel / np.linalg.norm(ds_vel)
        temp_vec = np.array([0,1,0]) if abs(base_mat[0,0]) > 0.9 else np.array([1,0,0])
        base_mat[:,1] = temp_vec - np.dot(temp_vec, base_mat[:,0]) * base_mat[:,0]
        base_mat[:,1] /= np.linalg.norm(base_mat[:,1]) + 1e-6  # Prevent division by zero
        base_mat[:,2] = np.cross(base_mat[:,0], base_mat[:,1])
        D_ds = base_mat @ damping_eigval @ base_mat.T
        return D_ds

def update_damping_matrix_role(ds_vel, damping_eigval, role=1.0):
        '''function to update the damping matrix
           change damping_eigval based on role [0, 1]
           role = 0.0 (human follower) to 1.0 (human leader)'''
        if np.linalg.norm(ds_vel) < 1e-6:
            return np.eye(3)
        damping_eigval = np.diag(damping_eigval) * role
        base_mat = np.zeros((3, 3))
        base_mat[:,0] = ds_vel / np.linalg.norm(ds_vel)
        temp_vec = np.array([0,1,0]) if abs(base_mat[0,0]) > 0.9 else np.array([1,0,0])
        base_mat[:,1] = temp_vec - np.dot(temp_vec, base_mat[:,0]) * base_mat[:,0]
        base_mat[:,1] /= np.linalg.norm(base_mat[:,1]) + 1e-6  # Prevent division by zero
        base_mat[:,2] = np.cross(base_mat[:,0], base_mat[:,1])
        D_ds = base_mat @ damping_eigval @ base_mat.T
        return D_ds

def update_damping_matrix_joint(jnt_vel, lambda0=20, lambda1=5):
    '''Update joint-space damping matrix.
       First eigenvector aligned with jnt_vel, eigenvalues [lambda0, lambda1, ...].'''
    j = np.asarray(jnt_vel, dtype=float).flatten()
    n_dof = j.size
    if n_dof == 0:
        return np.eye(0)

    if np.linalg.norm(j) <= 1e-6:
        # No preferred direction; isotropic damping with lambda1
        return np.eye(n_dof) * float(lambda1)

    # Orthonormal basis with first column aligned to normalized jnt_vel
    j_hat = j / np.linalg.norm(j)
    A = np.random.randn(n_dof, n_dof)
    A[:, 0] = j_hat
    Q, _ = np.linalg.qr(A)  # Q is orthonormal, Q[:,0] ~ j_hat

    lambdas = np.array([lambda0] + [lambda1] * (n_dof - 1), dtype=float)
    D_ds = Q @ np.diag(lambdas) @ Q.T
    return D_ds

#############################################################
#####################  HRI sim functions ####################
#############################################################
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
    forearm_state = pb.getLinkState(robot, forearm_index)  # Link=RightForeArm
    forearm_pos, forearm_quat = forearm_state[0], forearm_state[1]

    # Calculate the point of contact pose (above forearm: in forearm y-axis direction)
    forearm_poc_trans = np.array([0, forearm_dim, dist_from_forearm])
    # wrist_poc_quat = [0, 0, 0, 1] # Identity quaternion
    forearm_poc_quat = pb.getQuaternionFromEuler([0, -pi/2, pi/2])
    poc_pos, poc_quat = pb.multiplyTransforms(forearm_pos, forearm_quat, forearm_poc_trans, forearm_poc_quat)

    return np.concatenate([poc_pos, poc_quat])   #list(poc_pos) + list(poc_quat)

# --- Control helpers ---
def disable_default_motors(body_id, joint_indices=None):
    if joint_indices is None:
        joint_indices = [j for j in range(pb.getNumJoints(body_id))
                            if pb.getJointInfo(body_id, j)[2] != pb.JOINT_FIXED]
    pb.setJointMotorControlArray(body_id, joint_indices,
                                    controlMode=pb.VELOCITY_CONTROL,
                                    forces=[0.0]*len(joint_indices))
    return joint_indices

def apply_cmd_torque(body_id, joint_indices, torques):
    torques = np.array(torques, dtype=float).flatten().tolist()
    pb.setJointMotorControlArray(body_id, joint_indices, 
                                    pb.TORQUE_CONTROL, 
                                    forces=torques,
                                    positionGains=[0.0]*len(joint_indices),    # disable pos ctrl
                                    velocityGains=[0.0]*len(joint_indices),    # disable vel ctrl
                                )
