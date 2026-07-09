######################################################

import pybullet as p
import pybullet_data
import numpy as np
from math import pi
import time
import os, sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
from utils.pb_env_utils import *
from utils.sim_functions_utils import *
from utils.humanROM_utils import *
from utils.pb_vis_GUI_utils import *

class RobotGripper:
    def __init__(self, robot_name='iiwa14_gripper', arm_joints=None, gripper_joints=None,
                 EE_index=None, env=None):
        self.robot_name = robot_name
        self.robot_id = None
        self.arm_joints = arm_joints
        self.gripper_joints = gripper_joints  
        self.ee_index = EE_index
        self.env = env  # store shared env (PyBulletEnv or None)
        if env is not None:
            self.time_step = env.time_step
        else: 
            self.time_step = 1e-3

    def load(self, base_pose=(0,0,0,0,0,0,1), use_fixed_base=True):
        base_pos, base_quat = base_pose[:3], base_pose[3:7]
        # Use shared connection (assumes env.connect() already called)
        if self.env is not None:
            self.robot_id = self.env.load_urdf(f"{self.robot_name}.urdf", base_pos, base_quat, use_fixed_base=use_fixed_base)
        else:
            if not pb.isConnected():
                pb.connect(pb.GUI)
                pb.setAdditionalSearchPath(pybullet_data.getDataPath())
                pb.setTimeStep(self.time_step)
                pb.setGravity(0,0,-9.81)
            urdf_file = asset_path(f"{self.robot_name}.urdf")
            self.robot_id = pb.loadURDF(urdf_file, basePosition=base_pos, baseOrientation=base_quat,
                                    useFixedBase=use_fixed_base, flags=pb.URDF_USE_INERTIA_FROM_FILE)

        if self.arm_joints is None:
            self.arm_joints = list(range(1, pb.getNumJoints(self.robot_id)-1))
        if self.gripper_joints is None:
            self.gripper_joints = list(range(pb.getNumJoints(self.robot_id)-2, pb.getNumJoints(self.robot_id)-1))
        if self.ee_index is None:
            self.ee_index = self.arm_joints[-1]

        print('------------------------ \n ...Robot Initiated... robot =', self.robot_id, self.robot_name, ' \n------------------------')
        print('# All Joints:', pb.getNumJoints(self.robot_id))
        print('# Arm Joints:', self.arm_joints)
        print('# Gripper Joints:', self.gripper_joints)
        print('# End-effector:', self.ee_index, pb.getJointInfo(self.robot_id, self.ee_index)[1].decode('utf-8'))

        for j in range(pb.getNumJoints(self.robot_id)):
        # for j in self.arm_joints:
            joint_info = pb.getJointInfo(self.robot_id, j)
            print(f"Joint Index={joint_info[0]}: Name={joint_info[1]}, Type={joint_info[2]}, " 
                f"Link={joint_info[12].decode('utf-8')}, Lower Limit={joint_info[8]}, Upper Limit={joint_info[9]} ")

        self.joint_pos_limits = np.array([pb.getJointInfo(self.robot_id, i)[8:10] for i in self.arm_joints]).T
        self.joint_eff_limits = np.array([pb.getJointInfo(self.robot_id, i)[10] for i in self.arm_joints])
        self.joint_vel_limits = np.array([pb.getJointInfo(self.robot_id, i)[11] for i in self.arm_joints])
        self.joint_acc_limits = np.array([490.77, 490.80, 500.77, 650.71, 700.73, 900.66, 900.69])

        self.gripper_lower_limits = [pb.getJointInfo(self.robot_id, j)[8] for j in self.gripper_joints]
        self.gripper_upper_limits = [pb.getJointInfo(self.robot_id, j)[9] for j in self.gripper_joints]
        print(f"Gripper limits: {self.gripper_lower_limits}, {self.gripper_upper_limits}")

        # Debug - pause to check if the urds and initial pose are correct
        # while True:
        #     for j in range(pb.getNumJoints(self.robot_id)):
        #         pb.resetJointState(self.robot_id, j, 0)
        #     pb.stepSimulation()
        #     time.sleep(self.time_step)   
                                    
    def set_gripper_angles(self, indices, angles, use_limits=True, velocities=0):
        for i, (j, a) in enumerate(zip(indices, angles)):   
            pb.resetJointState(self.robot_id, jointIndex=j, 
                              targetValue=min(max(a, self.gripper_lower_limits[j]), self.gripper_upper_limits[j]) if use_limits else a, 
                              targetVelocity=velocities if type(velocities) in [int, float] else velocities[i])

    def set_grip(self, open_ratio= 0.0, force=500, set_instantly=False):
        '''set the gripper to a certain open ratio
        Args:
            open_ratio: float, 0.0 (fully closed) to 1.0 (fully open)
        '''
        if open_ratio < 0.0 or open_ratio > 1.0:
            raise ValueError("open_ratio must be between 0.0 and 1.0")

        pb.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.gripper_joints,
            controlMode=pb.POSITION_CONTROL,
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
            pb.setJointMotorControlArray(self.robot_id, self.gripper_joints,
                                        pb.TORQUE_CONTROL,
                                        forces=np.array(self.gripper_upper_limits) * grip_torque,  
                                        positionGains=[0]*len(self.gripper_joints),
                                        velocityGains=[0]*len(self.gripper_joints),
                                        )
            pb.stepSimulation()
            time.sleep(self.time_step)
            t += self.time_step
        print("Debug - Gripper torque control finished.")

    def get_gripper_gravity_trq(self):
        '''get the gravity compensation torques for the gripper joints'''
        joint_pos, joint_vel, joint_torques = self.get_joint_states(joints='all')
        G_q = np.asarray(pb.calculateInverseDynamics(self.robot_id, joint_pos, [0]*len(joint_pos), [0]*len(joint_pos)))
        G_q_gripper = G_q[7:]
        # print(f"Debug - Gripper gravity compensation torques: {G_q_gripper.round(2)}")
        return G_q_gripper
                    
    def set_home_configuration(self, home_configuration=None):
        ''' set the robot to its home configuration
            Args:
                home_configuration: list, joint angles for the home position (default is all zeros)
        '''
        if home_configuration is None:
            home_configuration = [0] * pb.getNumJoints(self.robot_id)  # Default home configuration
        # home_configuration = [state[0] for state in pb.getJointStates(self.robot_id, range(pb.getNumJoints(self.robot_id)))]
        for j in range(pb.getNumJoints(self.robot_id)):
            pb.resetJointState(self.robot_id, j, home_configuration[j])
        print("Robot set to home configuration.")

    def set_joint_positions(self, joint_positions, use_limits=False):
        '''set the joint positions of the robot
           Args:
               joint_positions: list, target joint positions
               use_limits: bool, whether to use joint limits (default is False)
        '''
        print("Setting joint positions:", joint_positions)
        arm_joints_limits = [pb.getJointInfo(self.robot_id, j)[8:10] for j in self.arm_joints]
        if use_limits:
            joint_positions = np.clip(joint_positions,
                                      [lim[0] for lim in arm_joints_limits],
                                      [lim[1] for lim in arm_joints_limits])
        for i, pos in enumerate(joint_positions):
            pb.resetJointState(self.robot_id, jointIndex=self.arm_joints[i], targetValue=pos)

    def get_joint_states(self, joints='arm'):
        '''get joint states (position, velocity, torque)
           joints = 'all' or 'controllable' (default)'''
        joints_to_use = self.arm_joints if joints == 'arm' else range(pb.getNumJoints(self.robot_id))
        joint_states = pb.getJointStates(self.robot_id, joints_to_use)
        if joints == 'all':
            joint_infos = [pb.getJointInfo(self.robot_id, i) for i in joints_to_use]
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
        ee_state = pb.getLinkState(self.robot_id, self.ee_index)
        jac_t, jac_r = pb.calculateJacobian(self.robot_id, self.ee_index, ee_state[2],
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
            pb.setJointMotorControlArray(self.robot_id, self.arm_joints,
                                        pb.VELOCITY_CONTROL,
                                        targetVelocities = joint_velocities,
                                        forces = [1]*len(joint_velocities))
        pb.stepSimulation()
        time.sleep(self.time_step)
        t += self.time_step
        print("Debug - Joint velocity control finished.")

    def get_dyn_matrices(self):
        '''get the dynamic matrices (inertia, coriolis and gravity) of the robot'''
        joint_pos, joint_vel, joint_torques = self.get_joint_states(joints='all')
        M_q = np.asarray(pb.calculateMassMatrix(self.robot_id, joint_pos))
        C_q = np.asarray(pb.calculateInverseDynamics(self.robot_id, joint_pos, joint_vel, joint_torques)) #, [0.0]*n_dof)) - G_q
        G_q = np.asarray(pb.calculateInverseDynamics(self.robot_id, joint_pos, [0]*len(joint_pos), [0]*len(joint_pos)))
        # print(f"Debug - M_q:\n{M_q.round(2)}\nC_q:\n{C_q.round(2)}\nG_q:\n{G_q.round(2)}")
        M_q, C_q, G_q = M_q[:7, :7], C_q[:7], G_q[:7]  # Only consider the first 7 joints (arm joints)  
        return M_q, C_q, G_q
    
    def get_joint_limits(body, joints):
        return [pb.getJointInfo(body, j)[8:10] for j in joints]

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
        base_pos, base_quat = pb.getBasePositionAndOrientation(self.robot_id)
        inv_base_pos, inv_base_quat = pb.invertTransform(base_pos, base_quat)
        F_ext_base = np.concatenate((pb.rotateVector(inv_base_quat, F_ext[:3]), 
                                     pb.rotateVector(inv_base_quat, F_ext[3:])), axis=0)  
        # F_ext_base = np.array(pb.multiplyTransforms(inv_base_pos, inv_base_quat, F_ext_point, [0, 0, 0, 1]))[0]  # Position in base frame
        # Compute the Jacobian at the point of application
        
        # Determine which link the force is applied to
        # for now:
        if poc_index is None:
            poc_index = self.ee_index
        joint_pos, joint_vel, _ = self.get_joint_states(joints='all')
        jac_t, jac_r = pb.calculateJacobian(self.robot_id, poc_index, F_ext_point,
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
    
    def passive_DS(self, desired_pose, null_pos=None, null_gain=None, role=1.0,
                   damping_eigval=None, k_lin = 20, k_ang = 5,
                   F_ext=None, F_ext_point=None):
        '''Function to do impedence control in task space'''
        n_dof = len(self.arm_joints)

        if damping_eigval is None:
            damping_eigval = [100.0, 100.0, 250.0, 10., 10., 1.0]   # [x y z roll pitch yaw] damping eigenvalues
        if F_ext is None:
            F_ext = np.zeros(6)
        elif len(F_ext) == 3:
            F_ext = np.concatenate((F_ext, np.zeros(3)), axis=0)
        if F_ext_point is None:
            F_ext_point = pb.getLinkState(self.robot_id, self.ee_index)[0]  # End-effector position in world coordinates
            
        # Desired & End-effector pose
        des_pose = desired_pose
        if len(des_pose) == 6:
            des_pos, des_quat = des_pose[:3], pb.getQuaternionFromEuler(des_pose[3:6])
        elif len(des_pose) == 7:
            des_pos, des_quat = des_pose[:3], des_pose[3:7]
        else:
            raise ValueError("Desired pose must be a 6D or 7D vector.")
        
        ee_state = pb.getLinkState(self.robot_id, self.ee_index, computeForwardKinematics=True, computeLinkVelocity=True) 
        # = link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot
        ee_pos, ee_quat = ee_state[0], ee_state[1]    # [x,y,z], [qx,qy,qz,qw] # or [4], [5]
        
        # # Transform desired and end-effector poses to base frame
        base_pos, base_quat = pb.getBasePositionAndOrientation(self.robot_id)       # robot base frame in world coordinates
        inv_base_pos, inv_base_quat = pb.invertTransform(base_pos, base_quat)    # world -> base        
        des_pos_base, des_quat_base = pb.multiplyTransforms(inv_base_pos, inv_base_quat, des_pos, des_quat)  
        ee_pos_base,  ee_quat_base  = pb.multiplyTransforms(inv_base_pos, inv_base_quat, ee_pos, ee_quat)   

        # # End-effector velocity
        q, dq, _ = self.get_joint_states()
        J = self.get_jacobian()
        ee_vel = J @ dq
        ee_vel_lin, ee_vel_ang = ee_vel[:3], pb.getQuaternionFromEuler(ee_vel[3:])  # [vx, vy, vz], [qx, qy, qz, qw]
        # ee_vel_lin, ee_vel_ang = ee_state[6], ee_state[7]  # [vx, vy, vz], [qx, qy, qz, qw]

        # # Desired velocity (DS)
        # # # # Linear DS 
        lin_error = np.array(des_pos_base) - np.array(ee_pos_base) 
        DS_vel_ang_dir = DS_linear_direction(ee_pos_base, des_pos_base)  # direction toward the target
        A_lin = k_lin*np.eye(3) if np.linalg.norm(lin_error) > 0.005 else np.zeros((3, 3))  # A matrix for linear DS
        ds_vel_lin = A_lin @ lin_error
        
        # # # # Angular DS  
        ang_error = np.array(des_quat_base) - np.array(ee_quat_base)
        DS_vel_ang_dir = DS_angular_direction(ee_quat_base, des_quat_base)  
        A_ang = k_ang * np.eye(3) if np.linalg.norm(ang_error) > 0.005 else np.zeros((3, 3))  # A matrix for angular DS
        ds_vel_ang = A_ang @ DS_vel_ang_dir

        # # Compute the wrench
        D_lin = update_damping_matrix_role(ds_vel_lin, damping_eigval[:3], role=role)  # damiping matrix (linear)
        wrench_lin = -D_lin @ (ee_vel_lin - ds_vel_lin) # np.array([0, 0, gripper_weight])  

        D_ang = update_damping_matrix_role(ds_vel_ang, damping_eigval[3:])  # damiping matrix (angular)
        wrench_ang = -D_ang @ (ee_vel_ang[:3] - ds_vel_ang)   # control output (quat_xyz components)  
        # wrench_ang = np.zeros(3)

        wrench = np.concatenate((wrench_lin, wrench_ang), axis=0) 
        # F_ext_base = F_ext 
        # wrench += F_ext_base

        # External forces compensation
        poc_index = 10 if self.robot_name == 'human_gazebo_shkey' else self.ee_index
        if F_ext is not None and np.linalg.norm(F_ext) > 0:
            F_ext_base, tau_ext = self.compute_external_torques(F_ext, F_ext_point, poc_index=poc_index)
            wrench += F_ext_base  # Add external force to the wrench
            # print(f"External torque applied: {tau_ext.round(2)} = J.t @ F_ext = {(J.T @ F_ext_base).round(2)}")
        # # Compute the torques
        tau_task = J.T @ wrench
        
        # if F_ext is not None and np.linalg.norm(F_ext) > 0:
        #     F_ext_base, tau_ext = self.compute_external_torques(F_ext, F_ext_point, poc_index=poc_index)
        #     # tau_task += tau_ext
        # Check taus
        J_t, J_r = self.get_jacobian(output='linrot')
        tau_trans = J_t.T @ wrench_lin
        tau_rot = J_r.T @ wrench_ang
        # print(f"tau_trans: {tau_trans.round(2)}, tau_rot: {tau_rot.round(2)}")

        # Gravity compensation
        Mq, Cq, Gq = self.get_dyn_matrices()
        tau_total = tau_task + Gq #+ np.array(Cq) @ np.array(dq)
        
        # Null-space control
        if null_pos is not None and null_gain is not None:
            tau_null = - np.array(null_gain) * (np.array(q) - np.array(null_pos)) - 2.0 * np.array(dq)
            null_projector = np.eye(n_dof) - np.linalg.pinv(J) @ J  
            tau_total = tau_total + null_projector @ tau_null

        # max_torque = 50
        # tau_total = np.clip(tau_total, -max_torque, max_torque)
        
        # # visualize target & ee pose
        # ee_pose = list(ee_pos) + list(ee_quat)
        # draw_pose_frame(ee_pose, length=0.1, lineWidth=2, life_time=0.2)
        # draw_pose_frame(des_pose, length=0.1, lineWidth=3, life_time=0.2)

        # # # visualize the wrench
        # wrench_lin_world = pb.rotateVector(base_quat, wrench_lin)  # linear wrench in world frame
        # wrench_ang_world = pb.rotateVector(base_quat, wrench_ang)  # angular wrench in world frame
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
        # print(f"dq: {np.array(dq).round(2)}")
        # print("-"*40)
        return tau_total

    def joint_impDS(self, desired_jntPos=None, desired_jntVel=None, damping_eigval=None,
                    null_pos=None, null_gain=None):
        '''Function to do impedence control in joint space'''
        if damping_eigval is None:
            damping_eigval = [100.0, 200.0, 200.0, 100., 100., 100, 100]   # [q1, ..., q7] damping eigenvalues

        q, dq, _ = self.get_joint_states()
        J = self.get_jacobian()       

        # # Desired velocity (DS)
        # # # # Joint DS 
        # joint_error = np.array(des_pose) - np.array(ee_pose)
        # DS_joint_vel = k_jnt * joint_error

        err_jntVel = np.array(dq) - desired_jntVel
        tau_jnt = - np.diag(damping_eigval) @ err_jntVel
        print(f"tau_jnt (PD): {tau_jnt.round(2)}")

        # Dmat_jnt = update_damping_matrix_joint(err_jntVel, lambda0=20, lambda1=5)
        # tau_jnt = - Dmat_jnt @ err_jntVel
        # print(f"tau_jnt (Dmat): {tau_jnt.round(2)}")

        # k_p = [50.0, 100.0, 100.0, 80.0, 80.0, 80.0, 50.0]  # proportional gains for joint position error
        # tau_jnt = np.diag(k_p) @ (np.array(desired_jntPos) - np.array(q)) #- np.diag(damping_eigval) @ np.array(dq) 
        
        # Gravity compensation
        Mq, Cq, Gq = self.get_dyn_matrices()
        tau_total = tau_jnt + Gq #+ np.array(Cq) @ np.array(dq)
        
        # Null-space control
        if null_pos is not None and null_gain is not None:
            tau_null = - np.array(null_gain) * (np.array(q) - np.array(null_pos)) - 2.0 * np.array(dq)
            null_projector = np.eye(len(self.arm_joints)) - np.linalg.pinv(J) @ J  
            tau_total = tau_total + null_projector @ tau_null
        # max_torque = 50
        # tau_total = np.clip(tau_total, -max_torque, max_torque)

        # print(f"Err_pos : {(np.array(desired_jntPos) - np.array(q)).round(2)}")
        # print(f"Err_vel : {(np.array(desired_jntVel) - np.array(dq)).round(2)}")
        # print(f"tau_jnt : {tau_jnt.round(2)}")

        return tau_total

# Note: the original demo/test block here (`if __name__ == "__main__":`) required
# `QPIK` from `utils.qpik_legacy`, which pulls in torch/cvxpy/pytorch_kinematics and a
# 229MB checkpoint directory. qpik_legacy is deferred (not part of this package), so the
# demo block was dropped rather than ported. See patient_robot_sim migration plan, Decision 2.