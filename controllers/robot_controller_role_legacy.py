######################################################

import pybullet as p
import pybullet_data
import numpy as np
from math import pi
import time
import os
from utils.util_functions import *
from utils.util_human_rom import *
from utils.util_visualization_GUI import *

class RobotGripper:
    def __init__(self, robot_name = 'iiwa14_gripper', arm_joints=None, gripper_joints=None, EE_index=None, time_step=1e-3):
        self.robot_name = robot_name
        self.robot = None
        self.arm_joints = arm_joints
        self.gripper_joints = gripper_joints  
        self.ee_index = EE_index
        self.time_step = time_step
        self.gripper_lower_limits = None
        self.gripper_upper_limits = None

    def load(self, base_pose=[0, 0, 0, 0, 0, 0, 1]):
        '''load the robot in pybullet'''
        
        urdf_file = 'agents/' + self.robot_name + '.urdf'
        base_pose, base_quat = base_pose[:3], base_pose[3:7]
        self.robot = p.loadURDF(urdf_file, basePosition=base_pose, baseOrientation=base_quat, useFixedBase=True)    #, flags=p.URDF_USE_INERTIA_FROM_FILE)

        if self.arm_joints is None:
            self.arm_joints = list(range(1, p.getNumJoints(self.robot)-1))
        if self.gripper_joints is None:
            self.gripper_joints = list(range(p.getNumJoints(self.robot)-2, p.getNumJoints(self.robot)-1))
        if self.ee_index is None:
            self.ee_index = self.arm_joints[-1]

        print('------------------------ \n ...Robot Initiated... robot =', self.robot, self.robot_name, ' \n------------------------')
        print('# All Joints:', p.getNumJoints(self.robot))
        print('# Arm Joints:', self.arm_joints)
        print('# Gripper Joints:', self.gripper_joints)
        print('# End-effector:', self.ee_index, p.getJointInfo(self.robot, self.ee_index)[1].decode('utf-8'))

        for j in range(p.getNumJoints(self.robot)):
        # for j in self.arm_joints:
            joint_info = p.getJointInfo(self.robot, j)
            print(f"Joint Index={joint_info[0]}: Name={joint_info[1]}, Type={joint_info[2]}, " 
                f"Link={joint_info[12].decode('utf-8')}, Lower Limit={joint_info[8]}, Upper Limit={joint_info[9]} ")

        self.gripper_lower_limits = [p.getJointInfo(self.robot, j)[8] for j in self.gripper_joints]
        self.gripper_upper_limits = [p.getJointInfo(self.robot, j)[9] for j in self.gripper_joints]
        print(f"Gripper limits: {self.gripper_lower_limits}, {self.gripper_upper_limits}")

        # Debug - pause to check if the urds and initial pose are correct
        # while True:
        #     for j in range(p.getNumJoints(self.robot)):
        #         p.resetJointState(self.robot, j, 0)
        #     p.stepSimulation()
        #     time.sleep(self.time_step)   
                                    
    def set_gripper_angles(self, indices, angles, use_limits=True, velocities=0):
        for i, (j, a) in enumerate(zip(indices, angles)):   
            p.resetJointState(self.robot, jointIndex=j, 
                              targetValue=min(max(a, self.gripper_lower_limits[j]), self.gripper_upper_limits[j]) if use_limits else a, 
                              targetVelocity=velocities if type(velocities) in [int, float] else velocities[i])

    def set_grip(self, open_ratio= 0.0, force=500, set_instantly=False):
        '''set the gripper to a certain open ratio
        Args:
            open_ratio: float, 0.0 (fully closed) to 1.0 (fully open)
        '''
        if open_ratio < 0.0 or open_ratio > 1.0:
            raise ValueError("open_ratio must be between 0.0 and 1.0")

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.gripper_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=np.array(self.gripper_upper_limits) * open_ratio,   
            # targetVelocities=[0] * len(self.gripper_joints),
            positionGains=[0.05] * len(self.gripper_joints),
            # velocityGains=[1] * len(self.gripper_joints),
            forces=[force]*len(self.gripper_joints)
        )
        if set_instantly:
            self.set_gripper_angles(self.gripper_joints, np.array(self.gripper_upper_limits) * open_ratio, use_limits=True)
    
    def gripper_torque_control(self, grip_torque=0.005, sim_time=2):
        '''set the gripper joint torques'''
        t = 0
        while t < sim_time:
            p.setJointMotorControlArray(self.robot, self.gripper_joints,
                                        p.TORQUE_CONTROL,
                                        forces=np.array(self.gripper_upper_limits) * grip_torque,  
                                        positionGains=[0]*len(self.gripper_joints),
                                        velocityGains=[0]*len(self.gripper_joints),
                                        )
            p.stepSimulation()
            time.sleep(self.time_step)
            t += self.time_step
        print("Debug - Gripper torque control finished.")

    def get_gripper_gravity_compensation(self):
        '''get the gravity compensation torques for the gripper joints'''
        joint_pos, joint_vel, joint_torques = self.get_joint_states(joints='all')
        G_q = np.asarray(p.calculateInverseDynamics(self.robot, joint_pos, [0]*len(joint_pos), [0]*len(joint_pos)))
        G_q_gripper = G_q[7:]
        # print(f"Debug - Gripper gravity compensation torques: {G_q_gripper.round(2)}")
        return G_q_gripper
                    
    def set_home_configuration(self, home_configuration=None):
        ''' set the robot to its home configuration
            Args:
                home_configuration: list, joint angles for the home position (default is all zeros)
        '''
        if home_configuration is None:
            home_configuration = [0] * p.getNumJoints(self.robot)  # Default home configuration
        # home_configuration = [state[0] for state in p.getJointStates(self.robot, range(p.getNumJoints(self.robot)))]
        for j in range(p.getNumJoints(self.robot)):
            p.resetJointState(self.robot, j, home_configuration[j])
        print("Robot set to home configuration.")

    def set_joint_positions(self, joint_positions, use_limits=False):
        '''set the joint positions of the robot
           Args:
               joint_positions: list, target joint positions
               use_limits: bool, whether to use joint limits (default is False)
        '''
        print("Setting joint positions:", joint_positions)
        arm_joints_limits = [p.getJointInfo(self.robot, j)[8:10] for j in self.arm_joints]
        if use_limits:
            joint_positions = np.clip(joint_positions,
                                      [lim[0] for lim in arm_joints_limits],
                                      [lim[1] for lim in arm_joints_limits])
        for i, pos in enumerate(joint_positions):
            p.resetJointState(self.robot, jointIndex=self.arm_joints[i], targetValue=pos)

    def get_joint_states(self, joints='arm'):
        '''get joint states (position, velocity, torque)
           joints = 'all' or 'controllable' (default)'''
        joints_to_use = self.arm_joints if joints == 'arm' else range(p.getNumJoints(self.robot))
        joint_states = p.getJointStates(self.robot, joints_to_use)
        if joints == 'all':
            joint_infos = [p.getJointInfo(self.robot, i) for i in joints_to_use]
            joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        # print("Debug - Retrieved jointStates positions, velocities, torques:", len(joint_positions), len(joint_velocities), len(joint_torques))
        return joint_positions, joint_velocities, joint_torques
            
    def get_jacobian(self, output='full'):
        '''get the Jacobian matrix of the robot
           Args:
               output: 'full' (default) or 'linrot' for linear and rotational parts separately
        '''
        joint_pos, join_vel , _ = self.get_joint_states(joints='all')
        # print("Debug - Joint positions:", joint_pos)
        ee_state = p.getLinkState(self.robot, self.ee_index)
        jac_t, jac_r = p.calculateJacobian(self.robot, self.ee_index, ee_state[2],
                                           list(joint_pos), [0.0]*len(joint_pos), [0.0]*len(joint_pos))
        J_t, J_r = np.asarray(jac_t), np.asarray(jac_r)
        # print(f"Idx {self.arm_joints}\n Jacobian:\n{J_t.round(2)}\n{J_r.round(2)}")
        J_t = J_t[:,:7] # only rows for arm joints
        J_r = J_r[:,:7] 
        if output == 'linrot':
            return J_t, J_r
        J = np.concatenate((J_t, J_r), axis=0)
        return J
    
    def joint_velocity_control(self, joint_velocities, sim_time=2, max_force=200):
        '''set the joint velocities of the robot'''
        t=0
        while t < sim_time:
            p.setJointMotorControlArray(self.robot, self.arm_joints,
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
        M_q = np.asarray(p.calculateMassMatrix(self.robot, joint_pos))
        C_q = np.asarray(p.calculateInverseDynamics(self.robot, joint_pos, joint_vel, joint_torques)) #, [0.0]*n_dof)) - G_q
        G_q = np.asarray(p.calculateInverseDynamics(self.robot, joint_pos, [0]*len(joint_pos), [0]*len(joint_pos)))
        # print(f"Debug - M_q:\n{M_q.round(2)}\nC_q:\n{C_q.round(2)}\nG_q:\n{G_q.round(2)}")
        M_q, C_q, G_q = M_q[:7, :7], C_q[:7], G_q[:7]  # Only consider the first 7 joints (arm joints)  
        return M_q, C_q, G_q
    
    def get_joint_limits(body, joints):
        return [p.getJointInfo(body, j)[8:10] for j in joints]

    def compute_external_torques(self, F_ext, F_ext_point, poc_index=None):
        """
        Compute joint torques caused by external forces applied at a specific point on the robot.
        Args:
            robot: The robot object.
            F_ext: External force vector (6D: [Fx, Fy, Fz, Mx, My, Mz]) in the robot's base frame.
            F_ext_point: Point of application of the external force in world coordinates.
        Returns:
            tau_ext: Joint torques caused by the external force.
        """
        # Transform the external force to the robot's base frame
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot)
        inv_base_pos, inv_base_quat = p.invertTransform(base_pos, base_quat)
        F_ext_base = np.concatenate((p.rotateVector(inv_base_quat, F_ext[:3]), 
                                     p.rotateVector(inv_base_quat, F_ext[3:])), axis=0)  
        # F_ext_base = np.array(p.multiplyTransforms(inv_base_pos, inv_base_quat, F_ext_point, [0, 0, 0, 1]))[0]  # Position in base frame
        # Compute the Jacobian at the point of application
        
        # Determine which link the force is applied to
        # for now:
        if poc_index is None:
            poc_index = self.ee_index
        joint_pos, joint_vel, _ = self.get_joint_states(joints='all')
        jac_t, jac_r = p.calculateJacobian(self.robot, poc_index, F_ext_point,
                                        list(joint_pos), [0.0]*len(joint_pos), [0.0]*len(joint_pos))
        J_t, J_r = np.asarray(jac_t), np.asarray(jac_r)
        J_t = J_t[:,:7]
        J_r = J_r[:,:7]
        # if len(self.arm_joints) != len(joint_pos):
        #     J_t = J_t[:, self.arm_joints]  # only rows for controllable joints
        #     J_r = J_r[:, self.arm_joints] 
        J = np.concatenate((J_t, J_r), axis=0)
        tau_ext = J.T @ F_ext_base
        return F_ext_base, tau_ext

    def update_damping_matrix_task(self, ds_vel, damping_eigval):
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
    
    def update_damping_matrix_role(self, ds_vel, damping_eigval, role=1.0):
        '''function to update the damping matrix
           change damping_eigval based on role [0, 1]
           role = 0.0 (robot follower) to 1.0 (robot leader)'''
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
    
    def passive_DS(self, desired_pose, null_pos=None, null_gain=None, role=1.0,
                   damping_eigval=None, k_lin = 20, k_ang = 5,
                   F_ext=None, F_ext_point=None):
        '''Function to do impedence control in task space'''
        # p.setRealTimeSimulation(False)
        n_dof = len(self.arm_joints)

        if damping_eigval is None:
            damping_eigval = [100.0, 100.0, 250.0, 10., 10., 1.0]   # [x y z roll pitch yaw] damping eigenvalues
        if F_ext is None:
            F_ext = np.zeros(6)
        elif len(F_ext) == 3:
            F_ext = np.concatenate((F_ext, np.zeros(3)), axis=0)
        if F_ext_point is None:
            F_ext_point = p.getLinkState(self.robot, self.ee_index)[0]  # End-effector position in world coordinates
            
        # Desired & End-effector pose
        des_pose = desired_pose
        if len(des_pose) == 6:
            des_pos, des_quat = des_pose[:3], p.getQuaternionFromEuler(des_pose[3:6])
        elif len(des_pose) == 7:
            des_pos, des_quat = des_pose[:3], des_pose[3:7]
        else:
            raise ValueError("Desired pose must be a 6D or 7D vector.")
        
        ee_state = p.getLinkState(self.robot, self.ee_index, computeForwardKinematics=True, computeLinkVelocity=True) 
        # = link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot
        ee_pos, ee_quat = ee_state[0], ee_state[1]    # [x,y,z], [qx,qy,qz,qw] # or [4], [5]
        
        # # Transform desired and end-effector poses to base frame
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot)       # robot base frame in world coordinates
        inv_base_pos, inv_base_quat = p.invertTransform(base_pos, base_quat)    # world -> base        
        des_pos_base, des_quat_base = p.multiplyTransforms(inv_base_pos, inv_base_quat, des_pos, des_quat)  
        ee_pos_base,  ee_quat_base  = p.multiplyTransforms(inv_base_pos, inv_base_quat, ee_pos, ee_quat)   

        # # End-effector velocity
        q, dq, _ = self.get_joint_states()
        J = self.get_jacobian()
        ee_vel = J @ dq
        ee_vel_lin, ee_vel_ang = ee_vel[:3], p.getQuaternionFromEuler(ee_vel[3:])  # [vx, vy, vz], [qx, qy, qz, qw]
        # ee_vel_lin, ee_vel_ang = ee_state[6], ee_state[7]  # [vx, vy, vz], [qx, qy, qz, qw]

        # # Desired velocity (DS)
        # # # # Linear DS 
        lin_error = np.array(des_pos_base) - np.array(ee_pos_base) 
        A_lin = k_lin*np.eye(3) if np.linalg.norm(lin_error) > 0.005 else np.zeros((3, 3))  # A matrix for linear DS
        ds_vel_lin = A_lin @ lin_error
        
        # k_lin = 20 if np.linalg.norm(lin_error) > 0.005 else 0
        # ds_vel_lin = k_lin * lin_error

        # # Velocity limiting
        # max_vel_lin = 1
        # if np.linalg.norm(ds_vel_lin) > max_vel_lin:
        #     ds_vel_lin = (ds_vel_lin / np.linalg.norm(ds_vel_lin)) * max_vel_lin

        # # # # Angular DS  
        dqd = quaternion_slerp(ee_quat_base, des_quat_base, t=0.5)
        # Convert to w-first format for calculations
        ee_quat_wxyz = np.array([ee_quat_base[3], ee_quat_base[0], ee_quat_base[1], ee_quat_base[2]])
        dqd_wxyz = np.array([dqd[3], dqd[0], dqd[1], dqd[2]])
        deltaQ = dqd_wxyz - ee_quat_wxyz
        q_conj = np.array([ee_quat_wxyz[0], -ee_quat_wxyz[1], -ee_quat_wxyz[2], -ee_quat_wxyz[3]])
        # Quaternion product (deltaQ ⨂ q_conj)
        w = deltaQ[0]*q_conj[0] - np.dot(deltaQ[1:], q_conj[1:])
        xyz = deltaQ[0]*q_conj[1:] + q_conj[0]*deltaQ[1:] + np.cross(deltaQ[1:], q_conj[1:])
        temp_angVel = np.concatenate(([w], xyz))
        # Extract angular velocity components
        tmp_angular_vel = temp_angVel[1:]  # Use vector part of quaternion
        # Velocity limiting
        maxDq = 0.2
        if np.linalg.norm(tmp_angular_vel) > maxDq:
            tmp_angular_vel = (tmp_angular_vel / np.linalg.norm(tmp_angular_vel)) * maxDq
        # Nonlinear scaling factor
        theta_gq = (-0.5 / (4 * maxDq**2)) * np.dot(tmp_angular_vel, tmp_angular_vel)
        # Final desired angular velocity (ds)
        ds_vel_ang = 2 * (1 + np.exp(theta_gq)) * tmp_angular_vel
        
        ang_error = np.array(des_quat_base) - np.array(ee_quat_base)
        A_ang = k_ang * np.eye(3) if np.linalg.norm(ang_error) > 0.005 else np.zeros((3, 3))  # A matrix for angular DS
        ds_vel_ang = A_ang @ ds_vel_ang

        # k_ang = 5 if np.linalg.norm(ang_error) > 0.01 else 0.0
        # ds_vel_ang = k_ang * ds_vel_ang

        # # Compute the wrench
        D_lin = self.update_damping_matrix_role(ds_vel_lin, damping_eigval[:3], role=role)  # damiping matrix (linear)
        wrench_lin = -D_lin @ (ee_vel_lin - ds_vel_lin) # np.array([0, 0, gripper_weight])  

        D_ang = self.update_damping_matrix_role(ds_vel_ang, damping_eigval[3:], role=role)  # damiping matrix (angular)
        wrench_ang = -D_ang @ (ee_vel_ang[:3] - ds_vel_ang)   # control output (quat_xyz components)  
        # wrench_ang = np.zeros(3)

        wrench = np.concatenate((wrench_lin, wrench_ang), axis=0) 
        # F_ext_base = F_ext 
        # wrench += F_ext_base

        # External forces compensation
        poc_index = 10 if self.robot_name == 'humanSubjectWithMesh' else self.ee_index
        if F_ext is not None and np.linalg.norm(F_ext) > 0:
            F_ext_base, tau_ext = self.compute_external_torques(F_ext, F_ext_point, poc_index=poc_index)
            wrench += F_ext_base  # Add external force to the wrench
            # print(f"External torque applied: {tau_ext.round(2)} = J.t @ F_ext = {(J.T @ F_ext_base).round(2)}")
        # # Compute the torques
        tau_task = J.T @ wrench

        J_t, J_r = self.get_jacobian(output='linrot')
        tau_trans = J_t.T @ wrench_lin
        tau_rot = J_r.T @ wrench_ang
        # tau_task2 = tau_trans + tau_rot
        # # compare tau_task and tau_task2
        # print(f"Torque check - task: {tau_task.round(2)}, "
        #       f"sum: {tau_task2.round(2)}")

        print(f"tau_trans: {tau_trans.round(2)}, tau_rot: {tau_rot.round(2)}")

        
        # if F_ext is not None and np.linalg.norm(F_ext) > 0:
        #     F_ext_base, tau_ext = self.compute_external_torques(F_ext, F_ext_point, poc_index=poc_index)
        #     # tau_task += tau_ext

        # Gravity compensation
        Mq, Gq, Cq = self.get_dyn_matrices()
        tau_total = tau_task + Gq #+ np.array(Cq) @ np.array(dq)
        
        # Null-space control
        if null_pos is not None and null_gain is not None:
            tau_null = - np.array(null_gain) * (np.array(q) - np.array(null_pos)) - 2.0 * np.array(dq)
            null_projector = np.eye(n_dof) - J.T @ np.linalg.pinv(J).T
            tau_total = tau_total + null_projector @ tau_null

        # max_torque = 50
        # tau_total = np.clip(tau_total, -max_torque, max_torque)
        
        # visualize target & ee pose
        ee_pose = list(ee_pos) + list(ee_quat)
        draw_pose_frame(ee_pose, length=0.1, lineWidth=2, life_time=0.2)
        draw_pose_frame(des_pose, length=0.1, lineWidth=3, life_time=0.2)

        # # # visualize the wrench
        # wrench_lin_world = p.rotateVector(base_quat, wrench_lin)  # linear wrench in world frame
        # wrench_ang_world = p.rotateVector(base_quat, wrench_ang)  # angular wrench in world frame
        # wrench_world = np.concatenate((wrench_lin_world, wrench_ang_world), axis=0)
        # draw_wrench_arrows(ee_pos, wrench_world, length=0.1, lineWidth=2, life_time=0.1)
        
        # # print(f"Forces: {F_ext.round(2)}")
        # print(f"-------- Error: {np.array(lin_error).round(2)}, {np.array(ang_error).round(2)}")
        # print( "--------ds_vel:", ds_vel_lin, ds_vel_ang)
        # print( "--------ee_vel:", ee_vel_lin, ee_vel_ang)
        # print(f"-wrench_linear: {wrench[:3].round(2)}")
        # print(f"wrench_angular: {wrench[3:].round(2)}")
        # print(f"Command wrench: {wrench.round(2)}")
        # print(f" Jacobian:\n{J.round(2)}")
        # print(f"--Task torques: {tau_task.round(2)}")
        # print(f"Gravity torques: {np.array(Gq).round(2)}")
        # print(f"--Null torques: {null_projector @ tau_null.round(2)}")
        # print(f"Command torques: {tau_total.round(2)}")
        # # applied_torques = self.get_joint_states()[2]
        # # print(f"Applied torques: {np.array(applied_torques).round(2)}")
        # print("-"*40)

        return tau_total


if __name__ == "__main__":

    # Load the pybullet world
    GUI = True
    time_step = 1e-3
    n_iter = 100
    # client_id = 
    # Reconnect the physics engine to forcefully clear memory when running long training scripts
    if p.isConnected():
        p.disconnect()
    p.connect(p.GUI if GUI else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(time_step)
    p.setPhysicsEngineParameter(fixedTimeStep=time_step, numSolverIterations=n_iter, numSubSteps=10)
    # Disable real time simulation so that the simulation only advances when we call stepSimulation
    p.setRealTimeSimulation(False)

    agent = 'iiwa14_gripper'

    # Robot: KUKA iiwa14
    if agent == 'iiwa14_gripper':
        # arm_joints, gripper_joints, EE_index = list(range(1, 8)), [9, 10], 8        # Franka_gripper
        arm_joints, gripper_joints, EE_index = list(range(1, 8)), [10], 8        # Robotiq_gripper
        # arm_joints, gripper_joints, EE_index = [1, 2, 3, 4, 5, 6, 7, 9], [10], 8        # Robotiq_gripper
        arm_joints, gripper_joints, EE_index = list(range(1, 8)), [10, 12, 14, 15, 17, 19], 8        # Robotiq_gripper

        base_pose=[0.5, -0.8, 0, 0, 0, 0.707, 0.707]
        q_init = [0, 0.52359878, 0, -1.58824962,  0, 0, 0]
        desired_pose = np.array([0.3, -0.2, 0.1,    0, pi, pi/2])
        # desired_pose = np.array([0.3, -0.2, -0.,    0, pi, pi/2])

        null_pos = [0, 1.30899694, 0, -1.58824962,  0, 0, 0]
        # null_pos = [0, 0.84823002, 0, -1.12713363, 0, -0.808132, 1.57]
        null_pos = [0, 1.30899694, 0, -1.3962634, 0, 0, 0.0]
        null_pos = [ 0.        ,  0.84823002,  0.        , -1.12713363,  0.        , -0.808132,  1.57        ]
        null_pos = [0, pi/4, 0, -1.5, 0, pi/4, 0]
        null_gain = [5.,20.0,30.,50.,10.,20.,1.]
        # null_pos, null_gain = None, None
        damping_eigval = [100.0, 200.0, 200.0, 10., 10., 10]   

    robot = RobotGripper(robot_name=agent, arm_joints=arm_joints, gripper_joints=gripper_joints, EE_index=EE_index, time_step=time_step)
    robot.load(base_pose=base_pose)
    print(f"########## Robot loaded: {robot.robot_name} #########")
    print("----desired pose: ", desired_pose)
    

    n_dof = len(robot.arm_joints)
    q_init = [0] * n_dof
    q_init[1] = 0.5336
    q_init[3] = -1.5882
    robot.set_joint_positions(q_init)

    if damping_eigval is None:
        damping_eigval = [100.0, 100.0, 100.0, 10., 10., 10.0]   # [x y z roll pitch yaw] damping eigenvalues

    # forward dynamics simulation loop
    # for turning off link and joint damping
    for link_idx in range(p.getNumJoints(robot.robot)+1):
        p.changeDynamics(robot.robot, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        p.changeDynamics(robot.robot, link_idx, maxJointVelocity=200)
        if link_idx in robot.gripper_joints:
            p.changeDynamics(robot.robot, link_idx, maxJointVelocity=0.1)

    # for j in robot.controllable_joints:
    #     p.changeDynamics(robot.robot, j, jointLowerLimit=-2*pi, jointUpperLimit=2*pi)

    # Enable torque control
    p.setJointMotorControlArray(robot.robot, robot.arm_joints,
                                p.VELOCITY_CONTROL, 
                                forces=np.zeros(n_dof))

    # define GUI sliders
    gui_sliders = GUIcontrol()
    goalGUIids = gui_sliders.task_space(goal=desired_pose, max_limit=3.14, min_limit=-3.14)
    ForceGUIids = gui_sliders.force(forces=np.zeros(n_dof), max_limit=10, min_limit=-10)        # Initial forces

    # self-collision avoidance
    # p.setCollisionFilterPair(robot.robot, robot.robot, -1, -1, enableCollision=0)  # Disable self-collision for the robot

    while True:
        tau = robot.passive_DS(desired_pose, null_gain=null_gain, null_pos=null_pos, 
                               damping_eigval=damping_eigval, k_lin=20, k_ang=5) 

        p.setJointMotorControlArray(robot.robot, robot.arm_joints,
                                    controlMode = p.TORQUE_CONTROL, 
                                    forces = tau,
                                    positionGains=[0]*len(robot.arm_joints),   # Critical for torque mode
                                    velocityGains=[0]*len(robot.arm_joints)    # Disable implicit PD
                                    )

        p.stepSimulation()
        time.sleep(time_step)