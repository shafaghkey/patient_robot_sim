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

        self.arm_lower_limits = [p.getJointInfo(self.robot, j)[8] for j in self.arm_joints]
        self.arm_upper_limits = [p.getJointInfo(self.robot, j)[9] for j in self.arm_joints]

        self.gripper_lower_limits = [p.getJointInfo(self.robot, j)[8] for j in self.gripper_joints]
        self.gripper_upper_limits = [p.getJointInfo(self.robot, j)[9] for j in self.gripper_joints]
        print(f"Gripper limits: {self.gripper_lower_limits}, {self.gripper_upper_limits}")

        # self.create_fixed_constraints()
        # self.fix_uncontrollable_joints()
        # self.mimic_gripper_joints()  # Set gripper joints to mimic each other

        # Debug - pause to check if the urds and initial pose are correct
        # while True:
        #     for j in range(p.getNumJoints(self.robot)):
        #         p.resetJointState(self.robot, j, 0)
        #     p.stepSimulation()
        #     time.sleep(self.time_step)

    def mimic_gripper_joints(self, motor_joint_idx=10):
        '''Set the gripper joints to mimic each other'''
        for j in self.gripper_joints:
            joint_info = p.getJointInfo(self.robot, j)
            if joint_info[2] != p.JOINT_FIXED and j != motor_joint_idx:
                p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)
                c = p.createConstraint(self.robot, motor_joint_idx, self.robot, j, 
                                       jointType=p.JOINT_GEAR, jointAxis=[1, 0, 0], 
                                       parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
                if j in [12, 17]:
                    p.changeConstraint(c, gearRatio=-1, maxForce=10000,erp=0.2)  # Set gear ratio to -1 for mirroring
                else:
                    p.changeConstraint(c, gearRatio=1, maxForce=10000,erp=0.2)
                                

    def set_joint_angles(self, indices, angles, use_limits=True, velocities=0):
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
            self.set_joint_angles(self.gripper_joints, np.array(self.gripper_upper_limits) * open_ratio, use_limits=True)
    
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
        
    def fix_uncontrollable_joints(self):
        '''Fix all joints other than controllable joints'''
        for i in range(p.getNumJoints(self.robot)):
                    joint_info = p.getJointInfo(self.robot, i)
                    # joint_info[2] = 4   #p.JOINT_FIXED
                    if joint_info[2] != p.JOINT_FIXED:
                        if i not in self.arm_joints:
                            # p.setJointMotorControl2(self.robot, joint_info[0], p.POSITION_CONTROL, targetPosition=0.0, force=5000)  # High force ensures rigidity
                            # p.setJointMotorControl2(self.robot, joint_info[0], p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)
                            # p.setJointMotorControl2(self.robot, joint_info[0], p.TORQUE_CONTROL, force=0.0)
                            p.changeDynamics(self.robot, joint_info[0], jointLowerLimit=0, jointUpperLimit=0, jointDamping=1000.0)  # Extreme damping resists movement
                            # p.changeDynamics(self.robot, joint_info[0], jointType='fixed', jointLowerLimit=0, jointUpperLimit=0, jointDamping=1000.0) 

    def create_fixed_constraints(self):
        if self.arm_joints is None:
            print('All joints are controllable. No fixed constraints created.')
            return
        home_configuration = [state[0] for state in p.getJointStates(self.robot, range(p.getNumJoints(self.robot)))]
        for j in range(p.getNumJoints(self.robot)):
            if j not in self.arm_joints:
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
            p.resetJointState(self.robot, self.arm_joints[i], pos)

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
    
    def get_gripper_gravity_compensation(self):
        '''get the gravity compensation torques for the gripper joints'''
        joint_pos, joint_vel, joint_torques = self.get_joint_states(joints='all')
        G_q = np.asarray(p.calculateInverseDynamics(self.robot, joint_pos, [0]*len(joint_pos), [0]*len(joint_pos)))
        G_q_gripper = G_q[7:]
        # print(f"Debug - Gripper gravity compensation torques: {G_q_gripper.round(2)}")
        return G_q_gripper
    
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

    def qp_ik_solver(self, q, ds_vel_R, method='cvxopt', iter=1000):
        ''' input:  q: current joint angles
                    ds_vel_R: desired end-effector velocity in task space (6D vector) in robot's base frame
            output: q_dot_ik: joint velocities to achieve the desired end-effector velocity ds_vel
            lb: theta_dot_min, ub: theta_dot_max (output limits)
            L_q : joint lower limits, U_q: joint upper limits
            L_Dq: joint velocity lower limits, U_Dq: joint velocity upper limits
            mu_p>0: intensity coefficients, determine the magnitude of decelerations
            eta_p: scaling factor for joint velocity limits'''
        
        N_links = len(self.arm_joints)
        mu_p = 0.5  # intensity coefficient
        eta_p = 2   # scaling factor for joint velocity limits

        q_dot = np.ones(N_links)  
        theta_dot_lb = np.zeros(N_links)
        theta_dot_ub = np.zeros(N_links)

        L_q = self.arm_lower_limits
        U_q = self.arm_upper_limits
        L_Dq = np.array([-5] * N_links)  # Lower limits for joint velocities
        U_Dq = np.array([ 5] * N_links)  # Upper limits for joint velocities

        for j in range(N_links):
            theta_dot_lb[j] = max(mu_p * (eta_p * L_q[j] - q[j]), L_Dq[j])
            theta_dot_ub[j] = min(mu_p * (eta_p * U_q[j] - q[j]), U_Dq[j])

        # Construct the QP problem
        J = self.get_jacobian()  # Get the Jacobian matrix
        W = np.eye(N_links)  # Weight matrix for the objective function
        W = np.diag([100, 100, 100, 100, 10, 10, 10])  # Assuming equal weights for all joints
        # Constraints
        obj_func = q_dot.T @ W @ q_dot  # Objective function: minimize ||q_dot||^2
        A_eq, b_eq = J, ds_vel_R  # Equality constraint: J * q_dot = psi_dot
        A_ineq_1, b_ineq_1 = np.eye(N_links), theta_dot_ub  # Upper bound constraints: q_dot <= ub
        A_ineq_2, b_ineq_2 = -np.eye(N_links), -theta_dot_lb  # Lower bound constraints: q_dot >= lb

        # Solve the QP problem using a solver (e.g., cvxopt, quadprog, etc.)
        if method == 'cvxopt':
            from cvxopt import matrix, solvers
            # Convert to cvxopt format
            P = matrix(W)
            q = matrix(np.zeros(N_links))
            G = matrix(np.vstack((A_ineq_1, A_ineq_2)))
            h = matrix(np.hstack((b_ineq_1, -b_ineq_2)))
            A = matrix(A_eq)
            b = matrix(b_eq)
            sol = solvers.qp(P, q, G, h, A, b)
            if sol['status'] == 'optimal':
                q_dot_ik = np.array(sol['x']).flatten()
                print("QP IK solver success:", sol['message'])
            else:
                print("QP IK solver failed:", sol['message'])
                q_dot_ik = np.zeros(N_links)

        elif method == 'scipy':
            from scipy.optimize import minimize
            def objective(q_dot):
                return q_dot.T @ W @ q_dot
            def constraint_eq(q_dot):
                return J @ q_dot - ds_vel_R
            def constraint_ineq_1(q_dot):
                return theta_dot_ub - q_dot
            def constraint_ineq_2(q_dot):
                return q_dot - theta_dot_lb
            constraints = [{'type': 'eq', 'fun': constraint_eq},    
                           {'type': 'ineq', 'fun': constraint_ineq_1},
                           {'type': 'ineq', 'fun': constraint_ineq_2}]
            res = minimize(objective, q_dot, constraints=constraints, method='SLSQP', options={'maxiter': iter})
            if res.success:
                q_dot_ik = res.x
                print("QP IK solver success:", res.message)
            else:
                print("QP IK solver failed:", res.message)
                q_dot_ik = np.zeros(N_links)

        elif method == 'dynamical':
            U_Handle_ = np.zeros(N_links)
            t1 = time.time()

            # Construct_boundaries_vel()

            # Fill blocks of the M and b matrices
            Dimension_constraint = 6  # Number of constraints (3 linear + 3 angular)
            M = np.zeros((Dimension_constraint + N_links, Dimension_constraint + N_links))
            M[0:N_links, 0:N_links] = W
            M[0:N_links, N_links:N_links + Dimension_constraint] = -J.T
            M[N_links:N_links + Dimension_constraint, 0:N_links] = J
            # M[Dimension_constraint + N_links, 0:N_links] = DGamma.T
            # M[0:N_links, Dimension_constraint + N_links] = -DGamma

            b = np.concatenate((np.zeros(N_links), -ds_vel_R))  # , lambda * np.log(Gamma - 1)

            Handle_M_I_ = np.eye(Dimension_constraint + N_links) + M.T

            counter = 0
            duration = 0
            while np.linalg.norm(U_Handle_ - U_[:N_links]) > 0.0001:
                U_Handle_ = U_[:N_links].copy()
                Handle_projection = U_ - (M @ U_ + b)
                for i in range(len(Handle_projection)):
                    Handle_projection[i] = min(max(Handle_projection[i], U_minus_[i]), U_plus_[i]) - U_[i]
                U_ += Handle_M_I_ @ Handle_projection * dt
                counter += 1
                duration = time.time() - t1
            q_dot_ik = U_Handle_[:N_links]

        return q_dot_ik

    def passive_DS(self, desired_pose, mode='task', k_lin=20, k_ang=5, damping_eigval=None, 
                   F_ext=None, F_ext_point=None, null_pos=None, null_gain=None):
        '''Function to do impedence control in task or joint space'''
        
        if damping_eigval is None: 
            damping_eigval = [100.0, 200.0, 200.0, 10., 10., 10.]   # [x y z roll pitch yaw] damping eigenvalues
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
        
        # visualize target & ee pose
        ee_pose = list(ee_pos) + list(ee_quat)
        draw_pose_frame(ee_pose, length=0.1, lineWidth=2, life_time=0.2)
        draw_pose_frame(des_pose, length=0.1, lineWidth=3, life_time=0.2)

        # # Transform desired and end-effector poses to base frame
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot)       # robot base frame in world coordinates
        inv_base_pos, inv_base_quat = p.invertTransform(base_pos, base_quat)    # world -> base        
        des_pos_base, des_quat_base = p.multiplyTransforms(inv_base_pos, inv_base_quat, des_pos, des_quat)  
        ee_pos_base,  ee_quat_base  = p.multiplyTransforms(inv_base_pos, inv_base_quat, ee_pos, ee_quat)   

        # # Desired velocity (DS)
        # # # # Linear DS 
        lin_error = np.array(des_pos_base) - np.array(ee_pos_base) 
        A_lin = k_lin * np.eye(3) if np.linalg.norm(lin_error) > 0.005 else np.zeros((3, 3))  # A matrix for linear DS
        ds_vel_lin = A_lin @ lin_error

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
        ds_vel_ang = 2 * 2.50 * (1 + np.exp(theta_gq)) * tmp_angular_vel
        
        ang_error = np.array(des_quat_base) - np.array(ee_quat_base)
        A_ang = k_ang * np.eye(3) if np.linalg.norm(ang_error) > 0.005 else np.zeros((3, 3))  # A matrix for angular DS
        ds_vel_ang = A_ang @ ds_vel_ang
        
        ds_vel_R = np.concatenate((ds_vel_lin, ds_vel_ang), axis=0)  # Desired end-effector velocity in task space (6D vector)

        # # Current End-effector velocity
        q, dq, _ = self.get_joint_states()
        J = self.get_jacobian()
        ee_vel = J @ dq
        ee_vel_lin, ee_vel_ang = ee_vel[:3], p.getQuaternionFromEuler(ee_vel[3:])  # [vx, vy, vz], [qx, qy, qz, qw]
        # ee_vel_lin, ee_vel_ang = ee_state[6], ee_state[7]  # [vx, vy, vz], [qx, qy, qz, qw]

        # # Compute the Torques
        if mode == 'task':
            D_lin = self.update_damping_matrix(ds_vel_lin, damping_eigval[:3])  # damiping matrix (linear)
            wrench_lin = -D_lin @ (ee_vel_lin - ds_vel_lin) # np.array([0, 0, gripper_weight])  

            D_ang = self.update_damping_matrix(ds_vel_ang, damping_eigval[3:])  # damiping matrix (angular)
            wrench_ang = -D_ang @ (ee_vel_ang[:3] - ds_vel_ang)   # control output (quat_xyz components)  

            wrench = np.concatenate((wrench_lin, wrench_ang), axis=0) 
            # F_ext_base = F_ext 
            # wrench += F_ext_base

            tau_task = J.T @ wrench

        elif mode == 'joint':
            q_des = p.calculateInverseKinematics(self.robot, self.ee_index, 
                                                 targetPosition=des_pos,
                                                 targetOrientation=des_quat)[:7]
            
            # print(f"Desired joint angles: {180/pi*np.array(q_des).round(2)}")
            
            q_dot_ik = self.qp_ik_solver(q, ds_vel_R, iter=1000)  # Compute joint velocities for the desired pose

            m_joint_stiffness = np.diag([100.0, 100.0, 100.0, 10., 10., 10., 10.])  # Joint stiffness matrix
            m_joint_damping = 2 * m_joint_stiffness  # Joint damping matrix
            stiffness_torque = m_joint_stiffness @ (np.array(q_des) - np.array(q))
            damping_torque = -m_joint_damping @ (np.array(q_dot_ik) - np.array(dq))  # Damping term
            tau_task = damping_torque + stiffness_torque  # Total joint torques

            print(f"q_dot_ik: {180/pi*np.array(q_dot_ik).round(2)}")
            print(f"Joint stiffness torque: {stiffness_torque.round(2)}")
            print(f"Joint damping torque: {damping_torque.round(2)}")

        # External forces compensation
        if F_ext is not None and np.linalg.norm(F_ext) > 0:
            poc_index = 10 if self.robot_name == 'humanSubjectWithMesh' else self.ee_index
            tau_ext = self.compute_external_torques(F_ext, F_ext_point, poc_index=poc_index)
            tau_task += tau_ext

        # Gravity compensation
        Mq, Gq, Cq = self.get_dyn_matrices()
        tau_total = tau_task + Gq #+ np.array(Cq) @ np.array(dq)
        
        # Null-space control
        if null_pos is not None and null_gain is not None:
            tau_null = - np.array(null_gain) * (np.array(q) - np.array(null_pos)) - 2.0 * np.array(dq)
            null_projector = np.eye(len(self.arm_joints)) - J.T @ np.linalg.pinv(J).T
            tau_total = tau_total + null_projector @ tau_null

        # max_torque = 50
        # tau_total = np.clip(tau_total, -max_torque, max_torque)
        
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
        print(f"Command torques: {tau_total.round(2)}")
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
        tau = robot.passive_DS(desired_pose, mode='joint', damping_eigval=damping_eigval, k_lin=20, k_ang=5,
                                null_gain=null_gain, null_pos=null_pos)

        p.setJointMotorControlArray(robot.robot, robot.arm_joints,
                                    controlMode = p.TORQUE_CONTROL, 
                                    forces = tau,
                                    positionGains=[0]*len(robot.arm_joints),   # Critical for torque mode
                                    velocityGains=[0]*len(robot.arm_joints)    # Disable implicit PD
                                    )

        p.stepSimulation()
        time.sleep(time_step)