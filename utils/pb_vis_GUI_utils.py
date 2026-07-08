"""pb_vis_GUI_utils

PyBullet visualization and GUI helpers used during simulation.

Key utilities:
  draw_pose_frame        — draw XYZ axes for a 6-D or 7-D pose in the PyBullet scene
  draw_wrench_arrows     — draw force/torque arrows from a wrench vector
  draw_contact_point     — draw a sphere at a contact location
  draw_contact_Force     — draw a force arrow at a contact point
  GUIcontrol             — slider-based GUI control panel for PyBullet debug UI
"""
import pybullet as pb
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy


def draw_pose_frame(pose, length=0.1, lineWidth=2, life_time=0):
    '''function to draw pose frame in pybullet
    Args:
        pose: 6D or 7D vector [x, y, z, qx, qy, qz, qw] or [x, y, z, roll, pitch, yaw]
        length: length of the axes
        lineWidth: width of the lines
        life_time: time for which the lines will be visible
    '''
    if len(pose) == 7:
        rot_matrix = R_scipy.from_quat(np.asarray(pose[3:7])).as_matrix()
    elif len(pose) == 6:
        rot_matrix = R_scipy.from_euler('xyz', np.asarray(pose[3:6])).as_matrix()
    else:
        raise ValueError("Draw Frame Err: pose must be a 6D or 7D vector.")
    origin = pose[0:3]
    x_axis = origin + length * rot_matrix[:, 0]
    y_axis = origin + length * rot_matrix[:, 1]
    z_axis = origin + length * rot_matrix[:, 2]
    pb.addUserDebugLine(origin, x_axis, lineColorRGB=[1, 0, 0], lineWidth=lineWidth, lifeTime=life_time) # x-axis in red
    pb.addUserDebugLine(origin, y_axis, lineColorRGB=[0, 1, 0], lineWidth=lineWidth, lifeTime=life_time) # y-axis in green
    pb.addUserDebugLine(origin, z_axis, lineColorRGB=[0, 0, 1], lineWidth=lineWidth, lifeTime=life_time) # z-axis in blue

def draw_wrench_arrows(origin, wrench, length=0.1, lineWidth=2, life_time=0):
    '''function to draw wrench in pybullet'''
    # wrench = [fx, fy, fz, mx, my, mz]
    force_end  = origin + length * np.array(wrench[:3])  # Scale the force vector
    torque_end = origin + length * np.array(wrench[3:])  # Scale the torque vector
    pb.addUserDebugLine(origin, force_end, lineColorRGB=[0, 0, 0], lineWidth=lineWidth, lifeTime=life_time)
    pb.addUserDebugLine(origin, torque_end, lineColorRGB=[.7, .7, .7], lineWidth=lineWidth, lifeTime=life_time)

def draw_contact_point(contact_point, radius=0.01, color=[1, 0, 0], life_time=0):
    """Draw a contact point in PyBullet."""
    # contact_point = [x, y, z]
    # Optionally, draw a small cross to indicate the contact point
    pb.addUserDebugLine([contact_point[0] - radius, contact_point[1], contact_point[2]],
                       [contact_point[0] + radius, contact_point[1], contact_point[2]], color, lineWidth=2, lifeTime=life_time)
    pb.addUserDebugLine([contact_point[0], contact_point[1] - radius, contact_point[2]],
                       [contact_point[0], contact_point[1] + radius, contact_point[2]], color, lineWidth=2, lifeTime=life_time)

def draw_contact_Force(contact_force, contact_point, length=0.01, color=[0, 0, 0], life_time=0):
    """Draw a contact force vector in PyBullet."""
    # contact_force = [fx, fy, fz]
    # contact_point = [x, y, z]
    force_end = np.array(contact_point) + np.array(contact_force) * length  # Scale the force vector
    pb.addUserDebugLine(contact_point, force_end, color, lineWidth=2, lifeTime=life_time)


class GUIcontrol:
    def __init__(self):
        pass

    def readGUIparams(self, ids):
        '''function to read GUI parameters'''
        return [pb.readUserDebugParameter(id) for id in ids]
    
    def clearGUIparams(self, ids):
        '''function to remove GUI parameters'''
        for id in ids:
            pb.removeUserDebugItem(id)

    def addGUIparams(self, params):
        '''function to add GUI parameters
        Args:
            params: list of tuples (name, min, max, initial value)
        Returns:
            ids: list of ids of the sliders
        '''
        ids = []
        for p in params:
            id = pb.addUserDebugParameter(p[0], p[1], p[2], p[3])
            ids.append(id)
        return ids

    def create_target_pose_sliders(self, pose):
        '''function to create GUI sliders for target pose
           input: initial value (6D or 7D)       '''
        ids = []
        ids.append(pb.addUserDebugParameter("target_x", -1, 1, pose[0]))
        ids.append(pb.addUserDebugParameter("target_y", -1, 1, pose[1]))
        ids.append(pb.addUserDebugParameter("target_z", -1, 1, pose[2]))
        if len(pose) == 7:
            ids.append(pb.addUserDebugParameter("target_qx", -1, 1, pose[3]))
            ids.append(pb.addUserDebugParameter("target_qy", -1, 1, pose[4]))
            ids.append(pb.addUserDebugParameter("target_qz", -1, 1, pose[5]))
            ids.append(pb.addUserDebugParameter("target_qw", -1, 1, pose[6]))   
        elif len(pose) == 6:
            ids.append(pb.addUserDebugParameter("target_roll", -np.pi, np.pi, pose[3]))
            ids.append(pb.addUserDebugParameter("target_pitch", -np.pi, np.pi, pose[4]))
            ids.append(pb.addUserDebugParameter("target_yaw", -np.pi, np.pi, pose[5]))
        else:
            raise ValueError("Target Pose Slider Err: pose must be a 6D or 7D vector.")
        return ids
    
    def read_target_pose_sliders(self, ids):
        '''function to read GUI sliders for target pose
           output: 6D or 7D vector'''
        tx = pb.readUserDebugParameter(ids[0])
        ty = pb.readUserDebugParameter(ids[1])
        tz = pb.readUserDebugParameter(ids[2])
        if len(ids) == 7:
            qx = pb.readUserDebugParameter(ids[3])
            qy = pb.readUserDebugParameter(ids[4])
            qz = pb.readUserDebugParameter(ids[5])
            qw = pb.readUserDebugParameter(ids[6])
            q = np.array([qx, qy, qz, qw], dtype=np.float64)
            q = q/np.linalg.norm(q) if np.linalg.norm(q) >= 1e-9 else np.array([0, 0, 0, 1], dtype=np.float64)
            return np.array([tx, ty, tz, *q], dtype=np.float64)
        elif len(ids) == 6:
            roll = pb.readUserDebugParameter(ids[3])
            pitch = pb.readUserDebugParameter(ids[4])
            yaw = pb.readUserDebugParameter(ids[5])
            q = R_scipy.from_euler('xyz', [roll, pitch, yaw]).as_quat()
            return np.array([tx, ty, tz, *q], dtype=np.float64)
        else:
            raise ValueError("Read Target Pose Slider Err: ids must correspond to a 6D or 7D vector.")

    def create_joint_angle_sliders(self, joint_angles, lower_limit=None, upper_limit=None):
        '''function to create GUI sliders for joint angles
           (name of the parameter,range,initial value)        '''
        if lower_limit is None:
            lower_limit = -np.pi * np.ones_like(joint_angles)
        if upper_limit is None:
            upper_limit = np.pi * np.ones_like(joint_angles)
        ids = []
        for i, (joint, low, high) in enumerate(zip(joint_angles, lower_limit, upper_limit)):
            id = pb.addUserDebugParameter(f'joint_angle_{i+1}', low, high, joint)
            ids.append(id)
        return ids
    
    def read_joint_angle_sliders(self, ids):
        '''function to read GUI sliders for joint angles'''
        joint_angles = []
        for id in ids:
            joint_angle = pb.readUserDebugParameter(id)
            joint_angles.append(joint_angle)
        return joint_angles
    
    def force(self, forces, max_limit = 1.0, min_limit = -1.0):
        fxId = pb.addUserDebugParameter("fx", min_limit, max_limit, forces[0]) #force along x
        fyId = pb.addUserDebugParameter("fy", min_limit, max_limit, forces[1]) #force along y
        fzId = pb.addUserDebugParameter("fz", min_limit, max_limit, forces[2]) #force along z
        mxId = pb.addUserDebugParameter("mx", min_limit, max_limit, forces[3]) #moment along x
        myId = pb.addUserDebugParameter("my", min_limit, max_limit, forces[4]) #moment along y
        mzId = pb.addUserDebugParameter("mz", min_limit, max_limit, forces[5]) #moment along z
        return [fxId, fyId, fzId, mxId, myId, mzId]
    
    def damping_eigval(self, damping_eigval):
        '''function to create GUI sliders for damping eigenvalues
           (name of the parameter,range,initial value)        '''
        ids = []
        for i in range(len(damping_eigval)):
            id = pb.addUserDebugParameter(f'damping_eigval_{i}', 0.0, 200.0, damping_eigval[i])
            ids.append(id)
        return ids
