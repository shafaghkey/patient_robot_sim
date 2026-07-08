######################################################

import pybullet as p
import pybullet_data
import numpy as np
from math import pi
import time
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from utils.util_functions import *
from utils.util_human_rom import *
from utils.util_visualization_GUI import *

import cvxpy as cp
import ctypes

file_path = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(file_path, 'libcvxgensolver_joint_lim.so')
print("Loading shared library from:", file_path)
cbvgenlib = ctypes.CDLL('/home/shkey/catkins/patient_robot_ws/src/patient_robot_pybullet/libcvxgensolver_joint_lim.so')
# cbvgenlib = ctypes.CDLL('libcvxgensolver_joint_lim.so')
cbvgenlib.CVXsolver.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
cbvgenlib.CVXsolver.restype = None

def CVXsolver_in_python(J_in, fc_in, M_in, b_in, j_l_in, j_u_in):
    c_J_in = (ctypes.c_double * len(J_in))(*J_in)
    c_fc_in = (ctypes.c_double * len(fc_in))(*fc_in)
    c_M_in = (ctypes.c_double * len(M_in))(*M_in)
    c_b_in = (ctypes.c_double * len(b_in))(*b_in)

    c_j_l_in = (ctypes.c_double * len(j_l_in))(*j_l_in)
    c_j_u_in = (ctypes.c_double * len(j_u_in))(*j_u_in)
    opt_x1 = ctypes.c_double()
    opt_x2 = ctypes.c_double()
    opt_x3 = ctypes.c_double()
    opt_x4 = ctypes.c_double()
    opt_x5 = ctypes.c_double()
    opt_x6 = ctypes.c_double()
    opt_x7 = ctypes.c_double()

    cbvgenlib.CVXsolver(c_J_in, c_fc_in, c_M_in, c_b_in, c_j_l_in, c_j_u_in, ctypes.byref(opt_x1), ctypes.byref(opt_x2), ctypes.byref(opt_x3), ctypes.byref(opt_x4), ctypes.byref(opt_x5), ctypes.byref(opt_x6), ctypes.byref(opt_x7))

    result_x = [opt_x1.value, opt_x2.value, opt_x3.value, opt_x4.value, opt_x5.value, opt_x6.value, opt_x7.value]
    return result_x

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

class Robot:
    def __init__(self, robot_name = 'iiwa14', controllable_joints = None, EE_index = None):
        self.robot_name = robot_name
        self.robot = None
        self.controllable_joints = controllable_joints
        self.ee_index = EE_index
        self.time_step = 0.01

        if self.robot_name == 'iiwa14':
            # self.damping_eigval = [100.0, 100.0, 250.0, 10., 10., 1.0]   # [lin ang] damping eigenvalues
            self.damping_eigval = [100.0, 100.0, 250.0, 10., 10., 7.]
            # self.damping_eigval = [100, 50, 50, 10., 5., 5.]
        elif self.robot_name == 'human':
            self.damping_eigval = [10.0, 10.0, 25.0, 10., 10., 1.0]

    def load(self, base_pose=[0, 0, 0, 0, 0, 0, 1]):
        '''load the robot in pybullet'''
        
        urdf_file = 'agents/' + self.robot_name + '.urdf'
        # urdf_file = 'agents/urdf/' + self.robot_name + '.xacro'
        base_pose, base_quat = base_pose[:3], base_pose[3:7]
        self.robot = p.loadURDF(urdf_file, basePosition=base_pose, baseOrientation=base_quat, useFixedBase=True)

        self.create_fixed_constraints()

        if self.controllable_joints is None:
            self.controllable_joints = list(range(1, p.getNumJoints(self.robot)-1))
        if self.ee_index is None:
            self.ee_index = self.controllable_joints[-1]

        print('------------------------ \n ...Robot Initiated... robot =', self.robot_name, ' \n------------------------')
        print('# All Joints:', p.getNumJoints(self.robot))
        print('# Controllable Joints:', self.controllable_joints.__len__(), self.controllable_joints)
        print('# End-effector:', self.ee_index, p.getJointInfo(self.robot, self.ee_index)[1].decode('utf-8'))

        # for j in range(p.getNumJoints(self.robot)):
            # p.resetJointState(self.robot, j, 0)
        for j in self.controllable_joints:
            joint_info = p.getJointInfo(self.robot, j)
            print(f"Joint Index={joint_info[0]}: Name={joint_info[1]}, Type={joint_info[2]}, " 
                f"Link={joint_info[12].decode('utf-8')}, Lower Limit={joint_info[8]}, Upper Limit={joint_info[9]} ")
    
    def create_fixed_constraints(self):
        if self.controllable_joints is None:
            print('All joints are controllable. No fixed constraints created.')
            return
        for j in range(p.getNumJoints(self.robot)):
            if j not in self.controllable_joints:
                p.createConstraint(
                    parentBodyUniqueId=self.robot, parentLinkIndex=j,
                    childBodyUniqueId=-1, childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0,0,0], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
                
    def get_joint_states(self, joints='controllable'):
        '''get joint states (position, velocity, torque)
           joints = 'all' or 'controllable' (default)'''
        joints_to_use = self.controllable_joints if joints == 'controllable' else range(p.getNumJoints(self.robot))
        joint_states = p.getJointStates(self.robot, joints_to_use)
        if joints == 'all':
            joint_infos = [p.getJointInfo(self.robot, i) for i in joints_to_use]
            joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        # print("Debug - Retrieved jointStates positions, velocities, torques:", len(joint_positions), len(joint_velocities), len(joint_torques))
        return joint_positions, joint_velocities, joint_torques
            
    def set_joint_positions(self, joint_positions):
        '''set the joint positions of the robot'''
        print("Setting joint positions:", joint_positions)
        while True:
            p.setJointMotorControlArray(self.robot, self.controllable_joints,
                                    p.POSITION_CONTROL,
                                    targetPositions = joint_positions,
                                    targetVelocities = [0]*len(joint_positions),
                                    positionGains = [1]*len(joint_positions),
                                    velocityGains = [1]*len(joint_positions))
            joint_positions_current, _, _ = self.get_joint_states()
            if np.allclose(joint_positions_current, joint_positions, atol=1e-3):
                break
            p.stepSimulation()

    def get_jacobian(self, output='full'):
        # jointpos = [0, 0, 0, 0, 0, 0, 0]
        # jointpos[0] = p.getJointState(self.robot, 0)[0]
        # jointpos[1] = p.getJointState(self.robot, 1)[0]
        # jointpos[2] = p.getJointState(self.robot, 2)[0]
        # jointpos[3] = p.getJointState(self.robot, 3)[0]
        # jointpos[4] = p.getJointState(self.robot, 4)[0]
        # jointpos[5] = p.getJointState(self.robot, 5)[0]
        # jointpos[6] = p.getJointState(self.robot, 6)[0]
        # jac_t, jac_r = p.calculateJacobian(self.robot, self.ee_index, [0, 0, 0], 
        #                                   jointpos, [0.0]*len(joint_pos), [0.0]*len(joint_pos))
        # print("Debug - Joint positions:", jointpos)
        joint_pos, join_vel , _ = self.get_joint_states(joints='all')
        # print("Debug - Joint positions:", joint_pos)
        ee_state = p.getLinkState(self.robot, self.ee_index, )
        jac_t, jac_r = p.calculateJacobian(self.robot, self.ee_index, ee_state[2],
                                           list(joint_pos), [0.0]*len(joint_pos), [0.0]*len(joint_pos))
        # jac_t, jac_r = p.calculateJacobian(self.robot, self.ee_index, ee_state[2],
        #                                    list(joint_pos), list(join_vel), [0.0]*len(joint_pos))
        J_t, J_r = np.asarray(jac_t), np.asarray(jac_r)
        if len(self.controllable_joints) != len(joint_pos):
            J_t = J_t[:, self.controllable_joints]  # only rows for controllable joints
            J_r = J_r[:, self.controllable_joints]  # only rows for controllable joints
        if output == 'linrot':
            return J_t, J_r
        J = np.concatenate((J_t, J_r), axis=0)
        # debug
        if np.any(np.isnan(J)) or np.any(np.isinf(J)):
            raise ValueError("Jacobian matrix contains invalid values (NaN or inf).")
        #singularity check
        if np.linalg.cond(J) > 1e10:
            print('Jacobian is singular')
        return J
    
    def joint_velocity_control(self, joint_velocities, sim_time=2, max_force=200):
        '''set the joint velocities of the robot'''
        t=0
        while t < sim_time:
            p.setJointMotorControlArray(self.robot, self.controllable_joints,
                                        p.VELOCITY_CONTROL,
                                        targetVelocities = joint_velocities,
                                        forces = [1]*len(joint_velocities))
        p.stepSimulation()
        time.sleep(self.time_step)
        t += self.time_step
        print("Debug - Joint velocity control finished.")

    def get_dyn_matrices(self):
        '''get the dynamic matrices (inertia, coriolis and gravity) of the robot'''
        joint_pos, joint_vel, joint_torques = self.get_joint_states(joints='all')
        M_q = p.calculateMassMatrix(self.robot, joint_pos)
        C_q = p.calculateInverseDynamics(self.robot, joint_pos, joint_vel, joint_torques) #, [0.0]*n_dof)) - G_q
        G_q = p.calculateInverseDynamics(self.robot, joint_pos, [0]*len(joint_pos), [0]*len(joint_pos))
        if len(self.controllable_joints) != len(joint_pos):
            M_q = M_q[self.controllable_joints][:, self.controllable_joints]
            C_q = C_q[self.controllable_joints]
            G_q = G_q[self.controllable_joints]
        return M_q, C_q, G_q
    
    def update_damping_matrix(self, ds_vel, damping_eigval):
        '''function to update the damping matrix'''
        if np.linalg.norm(ds_vel) < 1e-6:
            return np.eye(3)
        
        damping_eigval = np.diag(damping_eigval)
        base_mat = np.zeros((3, 3))
        base_mat[:,0] = ds_vel / np.linalg.norm(ds_vel)
        # # Orthonormalize remaining columns
        # base_mat[:,1] = np.random.rand(3)
        # base_mat[:,1] -= base_mat[:,1].dot(base_mat[:,0]) * base_mat[:,0]
        # base_mat[:,1] /= np.linalg.norm(base_mat[:,1])
        # Revised orthogonalization
        temp_vec = np.array([0,1,0]) if abs(base_mat[0,0]) > 0.9 else np.array([1,0,0])
        base_mat[:,1] = temp_vec - np.dot(temp_vec, base_mat[:,0]) * base_mat[:,0]
        base_mat[:,1] /= np.linalg.norm(base_mat[:,1]) + 1e-6  # Prevent division by zero

        base_mat[:,2] = np.cross(base_mat[:,0], base_mat[:,1])
        D_ds = base_mat @ damping_eigval @ base_mat.T
        return D_ds
    
    def getJointLimit(self):
        joint_lower = [0, 0, 0, 0, 0, 0, 0]
        joint_upper = [0, 0, 0, 0, 0, 0, 0]

        for i in range(p.getNumJoints(self.robot) - 1):
            joint_lower[i] = p.getJointInfo(self.robot, i)[8]
            joint_upper[i] = p.getJointInfo(self.robot, i)[9]

        return joint_lower, joint_upper
    
    def passive_DS(self, desired_pose, null_pos=None, null_gain=None):
        '''Function to do impedence control in task space'''
        p.setRealTimeSimulation(False)

        n_dof = len(self.controllable_joints)

        self.prev_quat = None
        self.prev_time = 0

        # forward dynamics simulation loop
        # for turning off link and joint damping
        for link_idx in range(p.getNumJoints(self.robot)+1):
            p.changeDynamics(self.robot, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
            p.changeDynamics(self.robot, link_idx, maxJointVelocity=200)

        for j in self.controllable_joints:
            p.changeDynamics(self.robot, j, jointLowerLimit=-2*np.pi, jointUpperLimit=2*np.pi)

        # Enable torque control
        p.setJointMotorControlArray(self.robot, self.controllable_joints,
                                    p.VELOCITY_CONTROL, 
                                    forces=np.zeros(n_dof))

        # define GUI sliders
        gui_sliders = GUIcontrol()
        goalGUIids = gui_sliders.task_space(goal=desired_pose, max_limit=3.14, min_limit=-3.14)
        ForceGUIids = gui_sliders.force(forces=np.zeros(n_dof), max_limit=10, min_limit=-10)        # Initial forces

        lin_damping_eigenvalues = self.damping_eigval[:3]  # linear damping eigenvalues
        ang_damping_eigenvalues = self.damping_eigval[3:]  # angular damping eigenvalues

        while True:
            # read GUI values
            des_pose = gui_sliders.readGUIparams(goalGUIids) # task space goal
            F_ext = gui_sliders.readGUIparams(ForceGUIids) # applied external forces
            
            des_pos = des_pose[:3]
            des_quat = p.getQuaternionFromEuler(des_pose[3:6])

            q, dq, _ = self.get_joint_states()
            J = self.get_jacobian()
            J_t, J_r = self.get_jacobian(output='linrot')

            v = J @ dq
            ee_vel_t = v[:3]
            ee_vel_r = v[3:]
            
            
            # Get current end-effector state
            ee_state = p.getLinkState(self.robot, self.ee_index, 
                                      computeLinkVelocity=1,         # Enables velocity computation
                                      computeForwardKinematics=1)   # Ensures world frame coordinates
            # link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot = ee_state
            ee_pos = ee_state[4]    # [x,y,z] # or [4]
            ee_quat = ee_state[5]   # [qx,qy,qz,qw]  # or [5]
            ee_vel_lin = ee_state[6]  # linear velocity from physics engine
            ee_vel_ang = ee_state[7]  # angular velocity from physics engine

            # visualize target & ee pose
            ee_pose = list(ee_state[0]) + list(ee_state[1])
            draw_pose_frame(ee_pose, length=0.1, lineWidth=3, life_time=0.1)
            draw_pose_frame(des_pose, length=0.1, lineWidth=2, life_time=0.1)

            # # Transform to robot base frame
            # Get robot base frame in world coordinates
            base_pos, base_quat = p.getBasePositionAndOrientation(self.robot)
            # Compute inverse transform (world -> base)
            inv_base_pos, inv_base_quat = p.invertTransform(base_pos, base_quat)
            # Transform desired and end-effector poses to base frame
            des_pos_base, des_quat_base = p.multiplyTransforms(inv_base_pos, inv_base_quat, des_pos, des_quat)  # Returns (pos, quat)
            ee_pos_base, ee_quat_base = p.multiplyTransforms(inv_base_pos, inv_base_quat, ee_pos, ee_quat)  # Returns (pos, quat)


            # # # # Linear DS control
            lin_error = np.array(des_pos_base) - np.array(ee_pos_base)
            if np.linalg.norm(lin_error) > 0.01:
                ds_vel_lin = 10 * lin_error 
            else:
                ds_vel_lin = np.zeros(3)

            # lambda_lin = 0.5
            D_lin = self.update_damping_matrix(ds_vel_lin, lin_damping_eigenvalues)  # damiping matrix (linear)
            # wrench_lin = -D_lin @ (ee_vel_lin_base - ds_vel_lin)   # control output
            wrench_lin = -D_lin @ (ee_vel_t - ds_vel_lin)   # control output
            # wrench_lin = 100 * lin_error

            # # # # Angular DS control
            # # delta Euler does not work well
            # # Quaternion product (deltaQ ⨂ q_conj)
            # Slerp interpolation
            ee_quat, des_quat = ee_quat_base, des_quat_base
            dqd = quaternion_slerp(ee_quat, des_quat, t=0.5)
            # Convert to w-first format for calculations
            ee_quat_w0 = np.array([ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])
            dqd_w0 = np.array([dqd[3], dqd[0], dqd[1], dqd[2]])
            deltaQ = dqd_w0 - ee_quat_w0
            q_conj = np.array([ee_quat_w0[0], -ee_quat_w0[1], -ee_quat_w0[2], -ee_quat_w0[3]])
            # Quaternion product (deltaQ ⨂ q_conj)
            w = deltaQ[0]*q_conj[0] - np.dot(deltaQ[1:], q_conj[1:])
            xyz = deltaQ[0]*q_conj[1:] + q_conj[0]*deltaQ[1:] + np.cross(deltaQ[1:], q_conj[1:])
            temp_angVel = np.concatenate(([w], xyz))
            # Extract angular velocity components
            tmp_angular_vel = temp_angVel[1:]  # Use vector part of quaternion

            ang_error = np.array(tmp_angular_vel) 
            # if np.linalg.norm(ang_error) < 0.01:
            #     ds_vel_ang = np.zeros(3)
            # else:
            # Velocity limiting
            maxDq = 0.2
            if np.linalg.norm(tmp_angular_vel) > maxDq:
                tmp_angular_vel = (tmp_angular_vel / np.linalg.norm(tmp_angular_vel)) * maxDq
            # Nonlinear scaling factor
            theta_gq = (-0.5 / (4 * maxDq**2)) * np.dot(tmp_angular_vel, tmp_angular_vel)
            # Final desired angular velocity (ds)
            ds_vel_ang = 2 * 2.50 * (1 + np.exp(theta_gq)) * tmp_angular_vel
            # ds_vel_ang = 5 * tmp_angular_vel

            
            ee_vel_r = p.getQuaternionFromEuler(ee_vel_ang)[:3]  # Extract angular velocity components
            D_ang = self.update_damping_matrix(ds_vel_ang, ang_damping_eigenvalues)  # damiping matrix (angular)
            # wrench_ang = - D_ang @ (ee_vel_ang_base - ds_vel_ang)
            wrench_ang = -D_ang @ (ee_vel_r - ds_vel_ang)
            # wrench_ang = np.zeros(3) # 10 * ds_vel_ang 
            
            # Combine linear and angular DS control
            wrench = np.concatenate((wrench_lin, wrench_ang), axis=0) 
            F_ext_base = F_ext
            wrench += F_ext_base

            tau_task = J.T @ wrench
            # tau_lin = J_t.T @ wrench[:3]
            # tau_ang = J_r.T @ wrench[3:]
            # tau_task = tau_lin + tau_ang

            # Gravity compensation
            Mq, Gq, Cq = self.get_dyn_matrices()
            # print("Debug - Gq:", Gq)
            tau_total = tau_task + Gq #+ np.array(Cq) @ np.array(dq)
            
            # if null_pos is not None and null_gain is not None:
            #     # null-space control
            #     tau_null = - np.array(null_gain) * (np.array(q) - np.array(null_pos)) - 2.0 * np.array(dq)
            #     # Project null space torques
            #     J_pinv = np.linalg.pinv(J)
            #     null_projector = np.eye(n_dof) - J.T @ J_pinv.T
            #     tau_total = tau_task + null_projector @ tau_null
            # # max_torque = 50
            # # tau_total = np.clip(tau_total, -max_torque, max_torque)

            end_pos = ee_state[0]
            end_ori = ee_state[1] #xyz w
            kp = -2.0
            target = np.array(des_pose[:3])
            fx = [kp * (end_pos[0] - target[0]), kp * (end_pos[1] - target[1]), kp * (end_pos[2] - target[2])]

            ori = end_ori
            ori_des = np.array(des_quat)
            s1 = ori[3]
            s2 = ori_des[3]
            u1 = np.array([ori[0], ori[1], ori[2]])
            u2 = np.array([ori_des[0], ori_des[1], ori_des[2]])
            Su1 = np.array([[0, -u1[2], u1[1]], [u1[2], 0, -u1[0]], [-u1[1], u1[0], 0]])
            temp = -s1*u2 + s2*u1 - Su1@u2
            dori = np.array([s1*s2+u1@u2.T, temp[0], temp[1], temp[2]])
            logdq = np.arccos(dori[0])*(np.array([dori[1], dori[2], dori[3]]) / np.linalg.norm(np.array([dori[1], dori[2], dori[3]])))
            f_ori = -10*logdq

            xdot = ee_state[6]
            omegadot = ee_state[7]

            D_pos = 5.*np.eye(3)
            fc = -D_pos @ (np.transpose(np.array(xdot)) - np.transpose(fx))

            D_qua = 2.*np.eye(3)
            fq = -D_qua @ (np.transpose(np.array(omegadot)) - np.transpose(f_ori))

            f_all = np.concatenate((fc, fq))

            # #### Launch QP to get target torque ####
            # x = cp.Variable(7) # Torque
            # y = cp.Variable(7) # q ddot

            q = self.get_joint_states()[0]
            q_dot = self.get_joint_states()[1]
            q_ddot_test = [0, 0, 0, 0, 0, 0, 0]
            tau = list(p.calculateInverseDynamics(self.robot, q, q_dot, q_ddot_test))
            b = np.array(p.calculateMassMatrix(self.robot, q)) @ np.transpose(np.array(q_ddot_test)) - np.transpose(np.array(tau))

            ## Secondary task (nullspace)
            second_q = np.array([0.0, 1.309, 0.0, -1.3963, 0.0, -1.35192, 0.0])

            joint_lim_low, joint_lim_high = robot.getJointLimit()
            joint_lim_low = np.array(joint_lim_low)
            joint_lim_high = np.array(joint_lim_high)

            alpha1 = 100.0
            alpha2 = 100.0

            joint_low_barrier = -alpha1 * (np.array(q) - joint_lim_low - 0.1) - alpha2 * np.array(q_dot)
            joint_high_barrier = -alpha1 * (np.array(q) - joint_lim_high + 0.1) - alpha2 * np.array(q_dot)

            J_in = list(J.flatten('F'))
            fc_in = list(f_all)
            M_in = list(np.array(robot.getMassMatrix(q)).flatten('F'))
            b_in = list(b)
            jll_in = list(joint_low_barrier)
            jlu_in = list(joint_high_barrier)

            tau_total = CVXsolver_in_python(J_in, fc_in, M_in, b_in, jll_in, jlu_in)

            # Activate torque control  
            p.setJointMotorControlArray(self.robot, self.controllable_joints,
                                        controlMode = p.TORQUE_CONTROL, 
                                        forces = tau_total,
                                        # positionGains=[0]*len(self.controllable_joints),   # Critical for torque mode
                                        # velocityGains=[0]*len(self.controllable_joints)    # Disable implicit PD
                                        )
            
            # visualize the wrench
            wrench_world = wrench
            wrench_lin_world = p.rotateVector(base_quat, wrench_lin)  # Rotate linear wrench to world frame
            wrench_ang_world = p.rotateVector(base_quat, wrench_ang)  # Rotate angular wrench to world frame
            wrench_world = np.concatenate((wrench_lin_world, wrench_ang_world), axis=0)
            draw_wrench_arrows(ee_pose, wrench_world, length=0.1, lineWidth=2, life_time=0.1)

            applied_torques = self.get_joint_states()[2]
            # print(f"Forces: {F_ext.round(2)}")
            print(f"-------- Error: {np.array(lin_error).round(2)}, {np.array(ang_error).round(2)}")
            print( "--------ds_vel:", ds_vel_lin, ds_vel_ang)
            print( "--------ee_vel:", ee_vel_t, ee_vel_r)
            # print(f"Command wrench: {wrench.round(2)}")
            # print(f"-wrench_linear: {wrench[:3].round(2)}")
            # print(f"----tau_linear: {tau_lin.round(2)}")
            # print(f"wrench_angular: {wrench[3:].round(2)}")
            # print(f"---tau_angular: {tau_ang.round(2)}")
            # print(f"Command wrench: {wrench.round(2)}")
            # print(f"--Task torques: {tau_task.round(2)}")
            # print(f"Gravity torques: {np.array(Gq).round(2)}")
            # print(f"--Null torques: {null_projector @ tau_null.round(2)}")
            # print(f"Command torques: {tau_total.round(2)}")
            # print(f"Applied torques: {np.array(applied_torques).round(2)}")
            print("-"*40)

            p.stepSimulation()
            time.sleep(self.time_step)


if __name__ == "__main__":

    # Load the pybullet world
    GUI = True
    time_step = 1e-3
    n_iter = 100
    # client_id = 
    p.connect(p.GUI if GUI else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(0.01)
    p.setPhysicsEngineParameter(fixedTimeStep=time_step, numSolverIterations=n_iter, numSubSteps=10)
    p.setRealTimeSimulation(False)


    # Robot: KUKA iiwa14
    robot = Robot(robot_name='iiwa14')
    kuka_null_pos = [0, 0.52359878, 0, -1.58824962,  0, 0, 0]
    # kuka_null_pos = [0, 0.84823002, 0, -1.12713363, 0, -0.808132, 1.57]
    kuka_null_pos = [0, 1.30899694, 0, -1.3962634, 0, -1.35191731, 0.0]
    kuka_null_gain = [1.,80,30.,50,10.,20.,1.]
    robot.load(base_pose=[0.5, -0.7, 0, 0, 0, 0.707, 0.707])
    robot_q_init = [0, 0.52359878, 0, -1.58824962,  0, 0, 0]
    robot.set_joint_positions(robot_q_init)
    print("iiwa14 robot loaded")

    robot_desired_pose = np.array([0.32210892739372127, -0.2, 0.10331005425596486, 0, pi, pi/2])
    print("----desired pose: ", robot_desired_pose)
    robot.passive_DS(robot_desired_pose, null_gain=kuka_null_gain, null_pos=kuka_null_pos)

    # # Human
    # human = Robot(robot_name='human', controllable_joints = list(range(16, 26)))
    # human.load()
    # human_q_init = [0.0] * len(human.controllable_joints)
    # human_q_init[0] = 1.3          
    # human.set_joint_positions(human_q_init)

    # human_null_pos  = human_q_init
    # human_null_gain = [1, 1, 30, 50, 10, 20, 1, 1, 1, 1]
    # human_null_gain = [10, 10, 1, 1, 1, 1, 10, 10, 10, 10]

    # human_desired_pose = np.array([0.32210892739372127, -0.2, 0.10331005425596486, 0, pi/2, 0])
    # human_desired_pose = np.array([0.32210892739372127, -0.2, 0.10331005425596486, 0, 0, 0])
    # print("----desired pose: ", human_desired_pose)
    # human.passive_DS(human_desired_pose, null_gain=human_null_gain, null_pos=human_null_pos)