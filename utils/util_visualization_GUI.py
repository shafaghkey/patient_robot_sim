
import pybullet as p
import numpy as np


def draw_pose_frame(pose, length=0.1, lineWidth=2, life_time=0):
    '''function to draw pose frame in pybullet
    Args:
        pose: 6D or 7D vector [x, y, z, qx, qy, qz, qw] or [x, y, z, roll, pitch, yaw]
        length: length of the axes
        lineWidth: width of the lines
        life_time: time for which the lines will be visible
    '''
    if len(pose) == 7:
        rot_matrix = np.array(p.getMatrixFromQuaternion(pose[3:7])).reshape(3, 3)
    elif len(pose) == 6:
        rot_matrix = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(pose[3:6]))).reshape(3, 3)
    else:
        raise ValueError("Draw Frame Err: pose must be a 6D or 7D vector.")
    origin = pose[0:3]
    x_axis = origin + length * rot_matrix[:, 0]
    y_axis = origin + length * rot_matrix[:, 1]
    z_axis = origin + length * rot_matrix[:, 2]
    p.addUserDebugLine(origin, x_axis, lineColorRGB=[1, 0, 0], lineWidth=lineWidth, lifeTime=life_time) # x-axis in red
    p.addUserDebugLine(origin, y_axis, lineColorRGB=[0, 1, 0], lineWidth=lineWidth, lifeTime=life_time) # y-axis in green
    p.addUserDebugLine(origin, z_axis, lineColorRGB=[0, 0, 1], lineWidth=lineWidth, lifeTime=life_time) # z-axis in blue

def draw_wrench_arrows(origin, wrench, length=0.1, lineWidth=2, life_time=0):
    '''function to draw wrench in pybullet'''
    # wrench = [fx, fy, fz, mx, my, mz]
    force_end  = origin + .1*length * np.array(wrench[:3])  # Scale the force vector
    torque_end = origin +    length * np.array(wrench[3:])  # Scale the torque vector
    p.addUserDebugLine(origin, force_end, lineColorRGB=[0, 0, 0], lineWidth=lineWidth, lifeTime=life_time)
    p.addUserDebugLine(origin, torque_end, lineColorRGB=[.7, .7, .7], lineWidth=lineWidth, lifeTime=life_time)

def draw_contact_point(contact_point, radius=0.01, color=[1, 0, 0], life_time=0):
    """Draw a contact point in PyBullet."""
    # contact_point = [x, y, z]
    # Optionally, draw a small cross to indicate the contact point
    p.addUserDebugLine([contact_point[0] - radius, contact_point[1], contact_point[2]],
                       [contact_point[0] + radius, contact_point[1], contact_point[2]], color, lineWidth=2, lifeTime=life_time)
    p.addUserDebugLine([contact_point[0], contact_point[1] - radius, contact_point[2]],
                       [contact_point[0], contact_point[1] + radius, contact_point[2]], color, lineWidth=2, lifeTime=life_time)

def draw_contact_Force(contact_force, contact_point, length=0.01, color=[0, 0, 0], life_time=0):
    """Draw a contact force vector in PyBullet."""
    # contact_force = [fx, fy, fz]
    # contact_point = [x, y, z]
    force_end = np.array(contact_point) + np.array(contact_force) * length  # Scale the force vector
    p.addUserDebugLine(contact_point, force_end, color, lineWidth=2, lifeTime=life_time)

class GUIcontrol:
    def __init__(self):
        pass

    def readGUIparams(self, ids):
        '''function to read GUI parameters'''
        return [p.readUserDebugParameter(id) for id in ids]

    def task_space(self, goal, max_limit = 3.14, min_limit = -3.14):
        '''function to create GUI sliders for task space
           (name of the parameter,range,initial value)
        Args:
            goal: 6D vector [x, y, z, roll, pitch, yaw]
            max_limit: maximum limit for the sliders
            min_limit: minimum limit for the sliders
        Returns:
            ids: list of ids of the sliders
        '''
        xId = p.addUserDebugParameter("x", min_limit, max_limit, goal[0]) #x
        yId = p.addUserDebugParameter("y", min_limit, max_limit, goal[1]) #y
        zId = p.addUserDebugParameter("z", min_limit, max_limit, goal[2]) #z
        rollId = p.addUserDebugParameter("roll", min_limit, max_limit, goal[3]) #roll
        pitchId = p.addUserDebugParameter("pitch", min_limit, max_limit, goal[4]) #pitch
        yawId = p.addUserDebugParameter("yaw", min_limit, max_limit, goal[5]) # yaw
        return [xId, yId, zId, rollId, pitchId, yawId]
    
    def joint_space(self, joints=None):
        '''function to create GUI sliders for joint angles
           (name of the parameter,range,initial value)        '''
        ids = []
        for j in self.controllable_joints:
            joint_info = p.getJointInfo(self.robot, j)
            joint_state = joints[j-self.controllable_joints[0]] #p.getJointState(self.robot, j)[0]
            id = p.addUserDebugParameter(str(joint_info[0]), -3.14, +3.14, joint_state)
            ids.append(id)
        return ids
    
    def force(self, forces, max_limit = 1.0, min_limit = -1.0):
        fxId = p.addUserDebugParameter("fx", min_limit, max_limit, forces[0]) #force along x
        fyId = p.addUserDebugParameter("fy", min_limit, max_limit, forces[1]) #force along y
        fzId = p.addUserDebugParameter("fz", min_limit, max_limit, forces[2]) #force along z
        mxId = p.addUserDebugParameter("mx", min_limit, max_limit, forces[3]) #moment along x
        myId = p.addUserDebugParameter("my", min_limit, max_limit, forces[4]) #moment along y
        mzId = p.addUserDebugParameter("mz", min_limit, max_limit, forces[5]) #moment along z
        return [fxId, fyId, fzId, mxId, myId, mzId]
