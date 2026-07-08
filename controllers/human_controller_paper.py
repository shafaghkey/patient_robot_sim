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

from utils.pb_env_utils import *
from utils.sim_functions_utils import *
from utils.humanROM_utils import *
from utils.pb_vis_GUI_utils import *

class Human:
    def __init__(self, robot_name = 'human', controllable_joints = None, EE_index = None, time_step = 1e-3):
        self.robot_name = robot_name
        self.robot = None
        self.controllable_joints = controllable_joints
        self.ee_index = EE_index
        self.time_step = time_step

    def load(self, base_pose=[0, 0, 0, 0, 0, 0, 1]):
        '''load the robot in pybullet'''
        
        urdf_file = 'agents/' + self.robot_name + '.urdf'
        base_pose, base_quat = base_pose[:3], base_pose[3:7]

        # # When loading agent URDFs, enable self-collision parameters
        # robot_flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
        # self.robot = p.loadURDF(urdf_file, basePosition=base_pose, baseOrientation=base_quat, useFixedBase=True, flags=robot_flags) 

        self.robot = p.loadURDF(urdf_file, basePosition=base_pose, baseOrientation=base_quat, useFixedBase=True) #flags=p.URDF_USE_INERTIA_FROM_FILE)

        if self.controllable_joints is None:
            self.controllable_joints = list(range(1, p.getNumJoints(self.robot)-1))
        if self.ee_index is None:
            self.ee_index = self.controllable_joints[-1]
        print('------------------------ \n ...Robot Initiated... robot =', self.robot, self.robot_name, ' \n------------------------')
        print('# All Joints:', p.getNumJoints(self.robot))
        print('# Controllable Joints:', self.controllable_joints.__len__(), self.controllable_joints)
        print('# End-effector:', self.ee_index, p.getJointInfo(self.robot, self.ee_index)[1].decode('utf-8'))

        for j in range(p.getNumJoints(self.robot)):
        # for j in self.controllable_joints:
            joint_info = p.getJointInfo(self.robot, j)
            print(f"Joint Index={joint_info[0]}: Name={joint_info[1]}, Type={joint_info[2]}, " 
                f"Link={joint_info[12].decode('utf-8')}, Lower Limit={joint_info[8]}, Upper Limit={joint_info[9]} ")

        # # Debug - pause to check if the urds and initial pose are correct
        # while True:
        #     for j in range(p.getNumJoints(self.robot)):
        #         p.resetJointState(self.robot, j, 0)
        #     p.stepSimulation()
        #     time.sleep(self.time_step)

        # self.create_fixed_constraints()
        self.fix_uncontrollable_joints()
        
    def fix_uncontrollable_joints(self):
        '''Fix all joints other than controllable joints'''
        for i in range(p.getNumJoints(self.robot)):
                    joint_info = p.getJointInfo(self.robot, i)
                    # joint_info[2] = 4   #p.JOINT_FIXED
                    if joint_info[2] != p.JOINT_FIXED:
                        if i not in self.controllable_joints:
                            # p.setJointMotorControl2(self.robot, joint_info[0], p.POSITION_CONTROL, targetPosition=0.0, force=5000)  # High force ensures rigidity
                            # p.setJointMotorControl2(self.robot, joint_info[0], p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)
                            # p.setJointMotorControl2(self.robot, joint_info[0], p.TORQUE_CONTROL, force=0.0)
                            p.changeDynamics(self.robot, joint_info[0], jointLowerLimit=0, jointUpperLimit=0, jointDamping=1000.0)  # Extreme damping resists movement
                            # p.changeDynamics(self.robot, joint_info[0], jointType='fixed', jointLowerLimit=0, jointUpperLimit=0, jointDamping=1000.0) 

    def create_fixed_constraints(self):
        if self.controllable_joints is None:
            print('All joints are controllable. No fixed constraints created.')
            return
        home_configuration = [state[0] for state in p.getJointStates(self.robot, range(p.getNumJoints(self.robot)))]
        for j in range(p.getNumJoints(self.robot)):
            if j not in self.controllable_joints:
                p.resetJointState(self.robot, j, home_configuration[j])
                p.createConstraint(
                    parentBodyUniqueId=self.robot, parentLinkIndex=j,
                    childBodyUniqueId=-1, childLinkIndex=-1,
                    jointType=p.JOINT_FIXED, jointAxis=[0,0,0], 
                    parentFramePosition=[0,0,0], childFramePosition=[0,0,0]
                    )
                
    def set_home_configuration(self, home_configuration=None):
        '''set the robot to its home configuration'''
        if home_configuration is None:
            home_configuration = [0] * p.getNumJoints(self.robot)  # Default home configuration
        # home_configuration = [state[0] for state in p.getJointStates(self.robot, range(p.getNumJoints(self.robot)))]
        for j in range(p.getNumJoints(self.robot)):
            p.resetJointState(self.robot, j, home_configuration[j])
        print("Robot set to home configuration.")

    def set_joint_positions(self, joint_positions):
        '''set the joint positions of the robot'''
        print("Setting joint positions:", joint_positions)
        for i, pos in enumerate(joint_positions):
            p.resetJointState(self.robot, self.controllable_joints[i], pos)

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
            
    def get_jacobian(self, output='full'):
        joint_pos, join_vel , _ = self.get_joint_states(joints='all')
        # print("Debug - Joint positions:", joint_pos)
        ee_state = p.getLinkState(self.robot, self.ee_index, )
        jac_t, jac_r = p.calculateJacobian(self.robot, self.ee_index, ee_state[2],
                                           list(joint_pos), [0.0]*len(joint_pos), [0.0]*len(joint_pos))
        J_t, J_r = np.asarray(jac_t), np.asarray(jac_r)
        # print(f"Idx {self.controllable_joints}\n Jacobian:\n{J_t.round(2)}\n{J_r.round(2)}")
        if len(self.controllable_joints) != len(joint_pos):
            J_t = J_t[:, self.controllable_joints]  # only rows for controllable joints
            J_r = J_r[:, self.controllable_joints]  
        if output == 'linrot':
            return J_t, J_r
        J = np.concatenate((J_t, J_r), axis=0)
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
        M_q = np.asarray(p.calculateMassMatrix(self.robot, joint_pos))
        C_q = np.asarray(p.calculateInverseDynamics(self.robot, joint_pos, joint_vel, joint_torques)) #, [0.0]*n_dof)) - G_q
        G_q = np.asarray(p.calculateInverseDynamics(self.robot, joint_pos, [0]*len(joint_pos), [0]*len(joint_pos)))
        if len(self.controllable_joints) != len(joint_pos):
            M_q = M_q[np.ix_(self.controllable_joints, self.controllable_joints)]
            C_q = C_q[self.controllable_joints]
            G_q = G_q[self.controllable_joints]
        return M_q, C_q, G_q
    
    def get_joint_limits(body, joints):
        return [p.getJointInfo(body, j)[8:10] for j in joints]
    
    def world_to_base(self, pos_world, quat_world):
        """Convert world coordinates to robot base frame."""
        robot_base_pos, robot_base_quat = p.getBasePositionAndOrientation(self.robot)
        inv_base_pos, inv_base_quat = p.invertTransform(robot_base_pos, robot_base_quat)
        pos_base, quat_base = p.multiplyTransforms(inv_base_pos, inv_base_quat, pos_world, quat_world)
        return pos_base, quat_base

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
        if len(self.controllable_joints) != len(joint_pos):
            J_t = J_t[:, self.controllable_joints]  # only rows for controllable joints
            J_r = J_r[:, self.controllable_joints] 
        J = np.concatenate((J_t, J_r), axis=0)
        tau_ext = J.T @ F_ext_base
        return tau_ext

    def update_damping_matrix(self, ds_vel, damping_eigval):
        '''function to update the damping matrix'''
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
    
    def passive_DS(self, desired_pose, damping_eigval=None, k_lin=20, k_ang=5, role=1.0,
                   null_pos=None, null_gain=None, F_ext=None, F_ext_point=None):
        '''Function to do impedence control in task space
           desired_pose: [x, y, z, roll, pitch, yaw] or [x, y, z, qx, qy, qz, qw]
           damping_eigval: [x, y, z, roll, pitch, yaw] damping eigenvalues
           k_lin & k_ang: linear & angular stiffness
           role: 0.0 (human follower) to 1.0 (human leader)'''

        n_dof = len(self.controllable_joints)

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
        
        ee_state = p.getLinkState(self.robot, self.ee_index) # = link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot
        ee_pos, ee_quat = ee_state[0], ee_state[1]    # [x,y,z], [qx,qy,qz,qw] # or [4],[5]
        
        # # Transform desired and end-effector poses to base frame 
        des_pos_base, des_quat_base = self.world_to_base(des_pos, des_quat)  
        ee_pos_base, ee_quat_base = self.world_to_base(ee_pos, ee_quat)  

        # # End-effector velocity
        q, dq, _ = self.get_joint_states()
        J = self.get_jacobian()
        ee_vel = J @ dq
        ee_vel_lin, ee_vel_ang = ee_vel[:3], p.getQuaternionFromEuler(ee_vel[3:])  # [vx, vy, vz], [qx, qy, qz, qw]

        # # Desired velocity (DS)
        # # # # Linear DS 
        lin_error = np.array(ee_pos_base) - np.array(des_pos_base)
        # k_lin = k_lin if np.linalg.norm(lin_error) > 0.005 else 0
        ds_vel_lin = -k_lin * lin_error

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
        tmp_angular_vel = temp_angVel[1:]  # Use vector part of quaternion
        # Velocity limiting
        maxDq = 0.2
        if np.linalg.norm(tmp_angular_vel) > maxDq:
            tmp_angular_vel = (tmp_angular_vel / np.linalg.norm(tmp_angular_vel)) * maxDq
        # Nonlinear scaling factor
        theta_gq = (-0.5 / (4 * maxDq**2)) * np.dot(tmp_angular_vel, tmp_angular_vel)
        # Final desired angular velocity (ds)
        ds_vel_ang = 2 * (1 + np.exp(theta_gq)) * tmp_angular_vel
        
        # ang_error = np.array(des_quat_base) - np.array(ee_quat_base)
        # k_ang = k_ang if np.linalg.norm(ang_error) > 0.01 else 0.0
        ds_vel_ang = k_ang * ds_vel_ang
                        
        
        # # Compute the wrench
        D_lin = self.update_damping_matrix_role(ds_vel_lin, damping_eigval[:3], role=role)  # damiping matrix (linear)
        wrench_lin = -D_lin @ (ee_vel_lin - ds_vel_lin)       # control output 

        D_ang = self.update_damping_matrix_role(ds_vel_ang, damping_eigval[3:])             # damiping matrix (angular)
        wrench_ang = -D_ang @ (ee_vel_ang[:3] - ds_vel_ang)   # control output (quat_xyz components)  

        wrench = np.concatenate((wrench_lin, wrench_ang), axis=0) 
        # F_ext_base = F_ext 
        # wrench += F_ext_base

        # # Compute the torques
        tau_task = J.T @ wrench

        # External forces compensation
        poc_index = 10 if self.robot_name == 'human_gazebo_shkey' else self.ee_index
        if F_ext is not None and np.linalg.norm(F_ext) > 0:
            tau_ext = self.compute_external_torques(F_ext, F_ext_point, poc_index=poc_index)
            tau_task += tau_ext

        # Gravity compensation
        Mq, Gq, Cq = self.get_dyn_matrices()
        tau_total = tau_task + Gq #+ np.array(Cq) @ np.array(dq)
        
        # Null-space control
        if null_pos is not None and null_gain is not None:
            er_null = np.array(q) - np.array(null_pos)
            # Clamp to avoid high torques when far away
            print(f"Null-space error: {er_null.round(2)}, norm: {np.linalg.norm(er_null):.2f}")
            if np.linalg.norm(er_null) > 0.4: 
                er_null = 0.4 * np.array(er_null) / np.linalg.norm(er_null)
            tau_null = - np.array(null_gain) * er_null - 2.0 * np.array(dq)
            # tau_null = - null_stiffness * er_null - null_damping * np.array(dq)
            # print(f"shapes of J : {J.shape}, J.T: {J.T.shape}, p.linalg.pinv(J): {np.linalg.pinv(J).shape}, tau_null: {tau_null.shape}")
            null_projector = np.eye(n_dof) - J.T @ np.linalg.pinv(J).T  # Null-space projector
            tau_total = tau_total + null_projector @ tau_null


        # max_torque = 50
        # tau_total = np.clip(tau_total, -max_torque, max_torque)

        
        # visualize target & ee pose
        ee_pose = list(ee_pos) + list(ee_quat)
        draw_pose_frame(ee_pose, length=0.1, lineWidth=2, life_time=0.2)
        draw_pose_frame(des_pose, length=0.1, lineWidth=3, life_time=0.2)

        # # visualize the wrench
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
        # print(f" Jacobian: \n{J.round(2)}")
        # print(f"--Task torques: {tau_task.round(2)}")
        # print(f"Gravity torques: {np.array(Gq).round(2)}")
        # print(f"--- Null torques: {(null_projector @ tau_null).round(2)}")
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
    p.connect(p.GUI if GUI else p.DIRECT)
    configure_pybullet_search_paths()
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(time_step)
    p.setPhysicsEngineParameter(fixedTimeStep=time_step, numSolverIterations=n_iter, numSubSteps=10)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=20, cameraPitch=-20, cameraTargetPosition=[0.3, 0, 0.6])
    p.setRealTimeSimulation(False)
    EE_index = None

    ## Load assets & agent
    plane = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
    # Load cup & table
    visual_filename = "agents/objects/plastic_coffee_cup.obj"
    collision_filename = "agents/objects/plastic_coffee_cup_vhacd.obj"
    cup_scale= [0.05]*3
    cup_pos, cup_quat = [0.55, -0.15, 0.6], p.getQuaternionFromEuler([pi/2, 0, 0])
    tool_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=cup_scale, rgbaColor=[1, .5, 1, .9])
    tool_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=cup_scale)
    cup = p.createMultiBody(baseMass=0.0,  # Zero mass makes it immovable 
                            baseCollisionShapeIndex=tool_collision, baseVisualShapeIndex=tool_visual,
                            basePosition=cup_pos, baseOrientation=cup_quat,
                            useMaximalCoordinates=True)  # Better stability for static objects 
    table_cup = p.loadURDF("agents/table_cup_seated.urdf", [0.7, 0, 0], useFixedBase=True)
    table_human = p.loadURDF("agents/table_human_stool.urdf", [0, 0, 0], useFixedBase=True)

    desired_pose = np.array([cup_pos[0], cup_pos[1]-.055, cup_pos[2]+.07, pi/2, -pi/2, 0])
    desired_pose = list([cup_pos[0], cup_pos[1]-.055, cup_pos[2]+.07]) + list(p.getQuaternionFromEuler([pi/2, -pi/2, 0]))

    agent = 'human_gazebo_shkey'
    controllable_joints = list(range(6, 14))
    base_pose=[0, 0, 0.5, 0, 0, 0, 1]

    human = Human(robot_name=agent, controllable_joints=controllable_joints, EE_index=EE_index, time_step=time_step)
    human.load(base_pose=base_pose)

    home_config = [0.0] * p.getNumJoints(human.robot)  # Initialize all joints to 0
    home_config[23] = -1.3  # Left shoulder_rotx joint to -1.3 radians
    home_config[49] = -pi/2  # Right Hip
    home_config[68] = -pi/2  # Left Hip
    home_config[51] =  pi/2  # Right Knee
    home_config[70] =  pi/2  # Left Knee

    # Joint Index=48: Name=b'jRightHip_rotx', Type=0, Link=RightUpperLeg_f1, Lower Limit=-0.785398, Upper Limit=0.523599 
    # Joint Index=49: Name=b'jRightHip_roty', Type=0, Link=RightUpperLeg_f2, Lower Limit=-2.0944, Upper Limit=0.261799 
    # Joint Index=50: Name=b'jRightHip_rotz', Type=0, Link=RightUpperLeg, Lower Limit=-0.785398, Upper Limit=0.785398 
    # Joint Index=51: Name=b'jRightKnee_roty', Type=0, Link=RightLowerLeg_f1, Lower Limit=0.0, Upper Limit=2.35619 
    # Joint Index=52: Name=b'jRightKnee_rotz', Type=0, Link=RightLowerLeg, Lower Limit=-0.698132, Upper Limit=0.523599 
    # Joint Index=67: Name=b'jLeftHip_rotx', Type=0, Link=LeftUpperLeg_f1, Lower Limit=-0.523599, Upper Limit=0.785398 
    # Joint Index=68: Name=b'jLeftHip_roty', Type=0, Link=LeftUpperLeg_f2, Lower Limit=-2.0944, Upper Limit=0.261799 
    # Joint Index=69: Name=b'jLeftHip_rotz', Type=0, Link=LeftUpperLeg, Lower Limit=-0.785398, Upper Limit=0.785398 
    # Joint Index=70: Name=b'jLeftKnee_roty', Type=0, Link=LeftLowerLeg_f1, Lower Limit=0.0, Upper Limit=2.35619 
    # Joint Index=71: Name=b'jLeftKnee_rotz', Type=0, Link=LeftLowerLeg, Lower Limit=-0.523599, Upper Limit=0.698132 
    human.set_home_configuration(home_config)
    
    q_init = [0.0] * len(controllable_joints)
    q_init[1] = -1.3    # Test: pi/180 * -60
    human.set_joint_positions(q_init)

    # desired_pose = np.array([0.55, -0.2, 1.12, pi/2, -pi/2, 0])
    # desired_pose = np.array([0.62, -0.2, 1.28, pi/2, -pi/2*1.3, 0])
    # desired_pose = np.array([0.6, -0.2, .8, pi/2, -pi/2, 0])
    print("----desired pose: ", desired_pose)
    
    null_pos, null_gain = None, None
    # null_pos  = q_init
    # null_pos = [1.2, -0.6, 0.2, 0.8, -1.2, -0.5, 0.3, 0.0]
    # null_gain = [10, 1, 10, 10, 10, 1, 1, 1]
    # null_gain = [0, 0, 0, 0, 0, 0, 0, 0]
    # null_gain = [5.0, 80.0, 10.0, 30.0, 5.0, 5.0, 2.0 ,2.0]

    damping_eigval = [250.0, 200.0, 200.0, 2., 5., 5.]      # [x y z roll pitch yaw] damping eigenvalues
    damping_eigval = [200.0, 100.0, 100.0, 2, 2, 2]
    # damping_eigval = [200, 100, 100, 10, 10, 10]
    k_lin, k_ang = 20, 5    

    # # Learn the ROM for the human subjec
    # human_rom = human_ROM_4d(subject=1, impaired_arm="R")
    # human_rom.train_svm_model(scaled=False)
    # p.changeDynamics(human.robot, 6, jointLowerLimit=-0.9, jointUpperLimit= 0.4)
    # p.changeDynamics(human.robot, 7, jointLowerLimit=-1.8, jointUpperLimit=-0.9)

    n_dof = len(human.controllable_joints)

    # forward dynamics simulation loop
    # for turning off link and joint damping
    for link_idx in range(p.getNumJoints(human.robot)+1):
        p.changeDynamics(human.robot, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        p.changeDynamics(human.robot, link_idx, maxJointVelocity=200)

    # for j in robot.controllable_joints:
    #     p.changeDynamics(robot.robot, j, jointLowerLimit=-2*pi, jointUpperLimit=2*pi)

    # Enable torque control
    p.setJointMotorControlArray(human.robot, human.controllable_joints,
                                p.VELOCITY_CONTROL, 
                                forces=np.zeros(n_dof))

    # define GUI sliders
    gui_sliders = GUIcontrol()
    ForceGUIids = gui_sliders.force(forces=np.zeros(n_dof), max_limit=10, min_limit=-10)        # Initial forces

    # self-collision avoidance
    p.setCollisionFilterPair(human.robot, human.robot, -1, -1, enableCollision=0)  # Disable self-collision for the robot
    

    while True:

        # rom_tau = np.zeros(n_dof)
        # q = np.array(human.get_joint_states()[0]) #* 180/pi  # Convert to degrees
        # Gamma, Gamma_grad = human_rom.calc_Gamma_and_derivative(q[:4], condition='impaired')
        # rom_tau[:4] = human_rom.enforce_ROM_tau(q[:4]) #* 1e2
        # # TODO: either PD law (on q_dot) or QP (**DS_CBF)

        tau = human.passive_DS(desired_pose, null_gain=null_gain, null_pos=null_pos, damping_eigval=damping_eigval, 
                               k_lin=k_lin, k_ang=k_ang)
        # print(f"tau: {tau.round(1)}, rom_tau: {rom_tau.round(1)}, Gamma: {Gamma.round(2)}")
        # print(f"angles: {(np.array(robot.get_joint_states()[0]) ).round(1)}, Gamma: {Gamma.round(1)}")
        # tau += rom_tau

        p.setJointMotorControlArray(human.robot, human.controllable_joints,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=tau,
                                    positionGains=[0]*len(human.controllable_joints),   # Critical for torque mode
                                    velocityGains=[0]*len(human.controllable_joints)    # Disable implicit PD
                                    )

        p.stepSimulation()
        time.sleep(time_step)




# iiwa_toolkit:
# passive_track_params.yaml
    # options:
    # filter_gain: 0.2

    # control:
    #     dsGainPos: 6.0
    #     lambda0Pos: 70.0
    #     lambda1Pos: 35.0
    #     dsGainOri: 3.0
    #     lambda0Ori: 5.0
    #     lambda1Ori: 2.5

    # target:
    # iiwa:
    #     pos: [0.5, -0.25, 0.3]
    #     quat: [0.7071068, -0.7071068 , 0.0,  0.0]

    # inertia:
    # iiwa:
    #     null_gains: [5.0, 80.0, 10.0, 30.0, 5.0, 2.0 ,2.0]
    #     gain: 0.001
    #     desired : 5.0

# passive_track_params_dual_real.yaml
    # use_rqt: False # set to True to use rqt reconfigure 
    # use_inertia_shaping: False # set to True to use null space inertia shaping to find good hitting configurations

    # control:
    #     iiwa1:
    #       dsGainPos: 1.8 #4.5 #8.0  # Used on desidred vel ONLY when receiving a desired Pose 
    #       lambda0Pos: 120 # 100 # config 1: 120 #     #70.0 #80.0
    #       lambda1Pos: 70 # config 1: 80 #   ##35.0 40.0
    #       alphaPos: 50 # 50 ##150.0  # gain on desired vel during hit (too high from lam0 makes it deviate)
    #       lambda0PosHit: 40 # lambda 0 used during hit
    #       dsGainOri: 1.0 #2.5 #3.0
    #       lambda0Ori: 10.0 # 6.0 #5.0
    #       lambda1Ori: 5.0 # 3.0 #2.5
    #       alphaOri: 10.0
    #       ImpedanceOriStiffness: [50, 50, 42] # [260,260,100] # Task space gains in x,y,z  
    #       ImpedanceOriDamping: [2.5, 2.5, 2.5] ## [3,3.4,1.6]
    #       ImpedancePosStiffness: [1800.0, 1800.0, 2000.0]
    #       ImpedancePosDamping: [43.3, 31.68, 37.19]
    #     iiwa2:
    #       dsGainPos: 2.0 #1.3 #1.5 # 2.5
    #       lambda0Pos: 90 #120 # 90 #100.0
    #       lambda1Pos: 35 #60 # 30 #50.0 (lower than 30 to avoid oscillations when hitting) IMPROVE THIS ONE??
    #       alphaPos: 70 #70 #150.0
    #       lambda0PosHit: 60 #50 # lambda 0 used during hit
    #       dsGainOri: 1.0
    #       lambda0Ori: 8.0 #10.0
    #       lambda1Ori: 4.0 # 5.0
    #       alphaOri: 10.0
    #       ImpedanceOriStiffness: [45, 50, 50] #[70.0, 70.0, 70.0]
    #       ImpedanceOriDamping: [2.5, 2.5, 2.5] #[2.7, 2.3, 2.7] # UNTESTED used to be 3 3 2.7
    #       ImpedancePosStiffness: [1500.0, 1500.0, 1800.0] #[500.0, 500.0, 500.0] #[1800.0, 1800.0, 2000.0]
    #       ImpedancePosDamping: [40.3, 40.68, 40.19] #[10.0, 10.0, 10.0] # 

    # target:
    #     iiwa1:
    #       pos: [0.55, -0.05, 0.22] 
    #       quat: [0.707, -0.707, 0.0, 0.0] 
    #       null_pos:  [-0.48, 1.04,  0.02, -1.35, -1.92, -1.81, -0.81] # config 1: [-0.48, 1.04,  0.02, -1.35, -1.92, -1.81, -0.81] ## config 2 : [-1.125, 1.530, 1.193, -1.399, -2.819, -1.306, -0.355]
    #     iiwa2:
    #       pos: [0.55, -0.05, 0.22] ## config 1 : [0.55, 0.0, 0.22] ## golf : [0.55, -0.05, 0.22]
    #       quat: [0.707, -0.707, 0.0, 0.0] ## config 1: [0.5, 0.5, -0.5, 0.5] # golf : [0.707, -0.707, 0.0, 0.0] ###[0.707, 0.707, 0.0, 0.0]  #[-0.707, -0.707, 0.0, 0.0]
    #       null_pos: [-0.47, 1.12, -0.05, -1.39, -1.94, -1.89, -1.01] ## config 1:  [0.543, 1.196, -0.133, -1.315, -1.057, 1.786, -0.587] # config 2 : [0.524, 0.753, 0-0.199, -1.464, 0.136, 0.999, 1.849] ### golf : [-0.47, 1.12, -0.05, -1.39, -1.94, -1.89, -1.01]

    # inertia:
    #     iiwa1:
    #       null_stiffness: [50.0, 300.0, 40.0, 50.0, 15.0, 10.0, 10.0] # [5.0, 80.0, 10.0, 30.0, 5.0, 2.0 ,1.0] # # 
    #       null_damping: [5.0, 12.0, 5.0, 5.0, 2.0, 3.0 ,1.0] 
    #       gain: 1.0 #1.0 #0.001
    #       desired : 5.0
    #       direction : [0.0, 1.0, 0.0] # must be the same as the direction of hit_direction of AirHockey
    #     iiwa2:
    #       null_stiffness: [50.0, 300.0,  40.0, 50.0, 15.0, 3.0, 10.0] #[50.0, 120.0, 10.0, 40.0, 0.1, 0.001, 5.0] ##[10.0, 70.0, 0.00, 10.0, 0.00, 0.00, 1.0] # [5.0, 80.0, 10.0, 30.0, 5.0, 2.0 ,1.0] # # 
    #       null_damping: [5.0, 12.0, 5.0, 5.0, 2.0, 1.0 ,1.0] 
    #       gain: 0.1 #0.001 #1.0 #0.5 #0.001100.0
    #       desired : 5.0
    #       direction :  [0.0, -1.0, 0.0] # must be the same as the direction of hit_direction of AirHockey

    # start: # PD gains for starting phase 
    #     iiwa1:
    #       stiffness: [110., 60.5, 60.5, 33., 20.625, 10.7375, 6.75] # 4.7375, 2.75
    #       damping: [18.75, 22.5, 18.75, 22.5 , 6., 1.0, 0.6 ] #0.6, 0.3 ]
    #     iiwa2:
    #       stiffness: [150.0, 90.0, 80.0, 50.0, 20.5, 6.25, 4.5]
    #       damping: [22.0, 20.0, 15.0, 12.0, 5.0, 1.1 , 0.8]