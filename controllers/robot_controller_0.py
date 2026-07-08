######################################################

import pybullet as p
import pybullet_data
import numpy as np
from math import pi
import time
from utils.util_functions import *
from utils.util_human_rom import *
from utils.util_visualization_GUI import *


# class World:
#     def __init__(self, gravity = -9.81, time_step = 0.01):
#         self.gravity = gravity
#         self.time_step = time_step
#         self.client_id = None

#     def load(self, GUI=True):
#         '''load the pybullet world'''
#         if self.client_id is not None:
#             p.disconnect()
#         self.client_id = p.connect(p.GUI if GUI else p.DIRECT)

#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         p.resetSimulation()

#         p.setGravity(0, 0, self.gravity)
#         p.setTimeStep(self.time_step)
#         p.setPhysicsEngineParameter(fixedTimeStep=self.time_step, numSolverIterations=100, numSubSteps=10)
#         p.setRealTimeSimulation(False)
#         # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

class Robot:
    def __init__(self, robot_name = 'iiwa14', controllable_joints = None, EE_index = None, time_step = 1e-3):
        self.robot_name = robot_name
        self.robot = None
        self.controllable_joints = controllable_joints
        self.ee_index = EE_index
        self.time_step = time_step

    def load(self, base_pose=[0, 0, 0, 0, 0, 0, 1]):
        '''load the robot in pybullet'''
        
        urdf_file = 'agents/' + self.robot_name + '.urdf'
        base_pose, base_quat = base_pose[:3], base_pose[3:7]
        self.robot = p.loadURDF(urdf_file, basePosition=base_pose, baseOrientation=base_quat, useFixedBase=True)

        if self.controllable_joints is None:
            self.controllable_joints = list(range(1, p.getNumJoints(self.robot)-1))
        if self.ee_index is None:
            self.ee_index = self.controllable_joints[-1]
        print('------------------------ \n ...Robot Initiated... robot =', self.robot, self.robot_name, ' \n------------------------')
        print('# All Joints:', p.getNumJoints(self.robot))
        print('# Controllable Joints:', self.controllable_joints.__len__(), self.controllable_joints)
        print('# End-effector:', self.ee_index, p.getJointInfo(self.robot, self.ee_index)[1].decode('utf-8'))

        # for j in range(p.getNumJoints(self.robot)):
            # p.resetJointState(self.robot, j, 0)
        for j in self.controllable_joints:
            joint_info = p.getJointInfo(self.robot, j)
            print(f"Joint Index={joint_info[0]}: Name={joint_info[1]}, Type={joint_info[2]}, " 
                f"Link={joint_info[12].decode('utf-8')}, Lower Limit={joint_info[8]}, Upper Limit={joint_info[9]} ")

        # Debug - pause to check if the urds and initial pose are correct
        # while True:
        #     for j in range(p.getNumJoints(self.robot)):
        #         p.resetJointState(self.robot, j, 0)
        #     p.stepSimulation()
        #     time.sleep(self.time_step)

        # self.create_fixed_constraints()

    def create_fixed_constraints(self):
        if self.controllable_joints is None:
            print('All joints are controllable. No fixed constraints created.')
            return
        for j in range(p.getNumJoints(self.robot)):
            if j not in self.controllable_joints:
                p.createConstraint(
                    parentBodyUniqueId=self.robot, parentLinkIndex=j,
                    childBodyUniqueId=-1, childLinkIndex=-1,
                    jointType=p.JOINT_FIXED, jointAxis=[0,0,0], 
                    parentFramePosition=[0,0,0], childFramePosition=[0,0,0])

    def set_joint_positions(self, joint_positions):
        '''set the joint positions of the robot'''
        print("Setting joint positions:", joint_positions)
        for i, pos in enumerate(joint_positions):
            p.resetJointState(self.robot, self.controllable_joints[i], pos)
        # while True:
        #     p.setJointMotorControlArray(self.robot, self.controllable_joints,
        #                             p.POSITION_CONTROL,
        #                             targetPositions = joint_positions,
        #                             targetVelocities = [0]*len(joint_positions),
        #                             positionGains = [1]*len(joint_positions),
        #                             velocityGains = [1]*len(joint_positions))
        #     joint_positions_current, _, _ = self.get_joint_states()
        #     if np.allclose(joint_positions_current, joint_positions, atol=1e-2):
        #         break
        # p.stepSimulation()

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
        if len(self.controllable_joints) != len(joint_pos):
            J_t = J_t[:, self.controllable_joints]  # only rows for controllable joints
            J_r = J_r[:, self.controllable_joints]  
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
        M_q = np.asarray(p.calculateMassMatrix(self.robot, joint_pos))
        C_q = np.asarray(p.calculateInverseDynamics(self.robot, joint_pos, joint_vel, joint_torques)) #, [0.0]*n_dof)) - G_q
        G_q = np.asarray(p.calculateInverseDynamics(self.robot, joint_pos, [0]*len(joint_pos), [0]*len(joint_pos)))
        if len(self.controllable_joints) != len(joint_pos):
            M_q = M_q[np.ix_(self.controllable_joints, self.controllable_joints)]
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
    
    def passive_DS(self, desired_pose, null_pos=None, null_gain=None, damping_eigval=None):
        '''Function to do impedence control in task space'''
        p.setRealTimeSimulation(False)

        n_dof = len(self.controllable_joints)

        if damping_eigval is None:
            damping_eigval = [100.0, 100.0, 250.0, 10., 10., 1.0]   # [x y z roll pitch yaw] damping eigenvalues

        # forward dynamics simulation loop
        # for turning off link and joint damping
        for link_idx in range(p.getNumJoints(self.robot)+1):
            p.changeDynamics(self.robot, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
            p.changeDynamics(self.robot, link_idx, maxJointVelocity=200)

        # for j in self.controllable_joints:
        #     p.changeDynamics(self.robot, j, jointLowerLimit=-2*pi, jointUpperLimit=2*pi)

        # Enable torque control
        p.setJointMotorControlArray(self.robot, self.controllable_joints,
                                    p.VELOCITY_CONTROL, 
                                    forces=np.zeros(n_dof))

        # define GUI sliders
        gui_sliders = GUIcontrol()
        goalGUIids = gui_sliders.task_space(goal=desired_pose, max_limit=3.14, min_limit=-3.14)
        ForceGUIids = gui_sliders.force(forces=np.zeros(n_dof), max_limit=10, min_limit=-10)        # Initial forces

        while True:
            # read GUI values
            des_pose = gui_sliders.readGUIparams(goalGUIids) # task space goal
            F_ext = gui_sliders.readGUIparams(ForceGUIids) # applied external forces
            
            # Desired & End-effector pose
            des_pos, des_quat = des_pose[:3], p.getQuaternionFromEuler(des_pose[3:6])
            
            ee_state = p.getLinkState(self.robot, self.ee_index) # = link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot
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

            # # Desired velocity (DS)
            # # # # Linear DS 
            lin_error = np.array(des_pos_base) - np.array(ee_pos_base) 
            k_lin = 10 if np.linalg.norm(lin_error) > 0.01 else 0
            ds_vel_lin = k_lin * lin_error

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
            k_ang = 5 if np.linalg.norm(ang_error) > 0.01 else 0.0
            ds_vel_ang = k_ang * ds_vel_ang
                         
            
            # # Compute the wrench
            D_lin = self.update_damping_matrix(ds_vel_lin, damping_eigval[:3])  # damiping matrix (linear)
            wrench_lin = -D_lin @ (ee_vel_lin - ds_vel_lin)       # control output 

            D_ang = self.update_damping_matrix(ds_vel_ang, damping_eigval[3:])  # damiping matrix (angular)
            wrench_ang = -D_ang @ (ee_vel_ang[:3] - ds_vel_ang)   # control output (quat_xyz components)  

            # wrench_lin = np.zeros(3) # 100 * lin_error
            # wrench_ang = np.zeros(3) # 10 * ds_vel_ang
            wrench = np.concatenate((wrench_lin, wrench_ang), axis=0) 
            F_ext_base = F_ext 
            wrench += F_ext_base

            # # Compute the torques
            tau_task = J.T @ wrench

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

            # Activate torque control  
            p.setJointMotorControlArray(self.robot, self.controllable_joints,
                                        controlMode = p.TORQUE_CONTROL, 
                                        forces = tau_total,
                                        positionGains=[0]*len(self.controllable_joints),   # Critical for torque mode
                                        velocityGains=[0]*len(self.controllable_joints)    # Disable implicit PD
                                        )
            
            # visualize target & ee pose
            ee_pose = list(ee_pos) + list(ee_quat)
            draw_pose_frame(ee_pose, length=0.1, lineWidth=3, life_time=0.1)
            draw_pose_frame(des_pose, length=0.1, lineWidth=2, life_time=0.1)

            # visualize the wrench
            wrench_lin_world = p.rotateVector(base_quat, wrench_lin)  # linear wrench in world frame
            wrench_ang_world = p.rotateVector(base_quat, wrench_ang)  # angular wrench in world frame
            wrench_world = np.concatenate((wrench_lin_world, wrench_ang_world), axis=0)
            draw_wrench_arrows(ee_pos, wrench_world, length=0.1, lineWidth=2, life_time=0.1)
            
            # print(f"Forces: {F_ext.round(2)}")
            print(f"-------- Error: {np.array(lin_error).round(2)}, {np.array(ang_error).round(2)}")
            print( "--------ds_vel:", ds_vel_lin, ds_vel_ang)
            print( "--------ee_vel:", ee_vel_lin, ee_vel_ang)
            print(f"-wrench_linear: {wrench[:3].round(2)}")
            print(f"wrench_angular: {wrench[3:].round(2)}")
            print(f"Command wrench: {wrench.round(2)}")
            print(f"--Task torques: {tau_task.round(2)}")
            print(f"Gravity torques: {np.array(Gq).round(2)}")
            # print(f"--Null torques: {null_projector @ tau_null.round(2)}")
            print(f"Command torques: {tau_total.round(2)}")
            # applied_torques = self.get_joint_states()[2]
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
    p.setTimeStep(time_step)
    p.setPhysicsEngineParameter(fixedTimeStep=time_step, numSolverIterations=n_iter, numSubSteps=10)
    p.setRealTimeSimulation(False)

    agent = 'iiwa14'
    # agent = 'human'
    # agent = 'humanSubjectWithMesh'

    # Robot: KUKA iiwa14
    if agent == 'iiwa14':
        robot = Robot(robot_name='iiwa14')

        kuka_null_pos = [0, 1.30899694, 0, -1.58824962,  0, 0, 0]
        # kuka_null_pos = [0, 0.84823002, 0, -1.12713363, 0, -0.808132, 1.57]
        kuka_null_pos = [0, 1.30899694, 0, -1.3962634, 0, 0, 0.0]
        kuka_null_pos = [ 0.        ,  0.84823002,  0.        , -1.12713363,  0.        , -0.808132,  1.57        ]
        kuka_null_pos = [0, pi/4, 0, -1.5, 0, pi/4, 0]
        kuka_null_gain = [5.,20.0,30.,50.,10.,20.,1.]
        robot.load(base_pose=[0.5, -0.7, 0, 0, 0, 0.707, 0.707])
        robot_q_init = [0, 0.52359878, 0, -1.58824962,  0, 0, 0]
        # robot_q_init = [0, pi/4, 0, -2, 0, 0, 0]
        robot.set_joint_positions(robot_q_init)
        print("iiwa14 robot loaded")

        kuka_damping_eigval = [100.0, 100.0, 250.0, 10., 10., 1.0]
        robot_desired_pose = np.array([0.32210892739372127, -0.2, 0.10331005425596486, 0, pi, pi/2])
        print("----desired pose: ", robot_desired_pose)
        robot.passive_DS(robot_desired_pose, null_gain=kuka_null_gain, null_pos=kuka_null_pos, damping_eigval=kuka_damping_eigval)

    # Human
    elif agent == 'human':
        human = Robot(robot_name='human', controllable_joints = list(range(16, 26)))
        human.load(base_pose=[0, 0, 0.5, 0, 0, 0, 1])
        human_q_init = [0.0] * len(human.controllable_joints)
        human_q_init[0] = 1.3    
        # human_q_init = [1.3, 0, 0, 0.5, 0., 0., 0., pi/8, 0, 0]      
        human.set_joint_positions(human_q_init)
        print("Human robot loaded")

        human_null_pos  = human_q_init
        human_null_gain = [1, 1, 30, 50, 10, 20, 1, 1, 1, 1]
        human_null_gain = [10, 10, 1, 1, 1, 1, 10, 10, 10, 10]
        human_null_pos, human_null_gain = None, None

        human_desired_pose = np.array([0.35, -0.2, 0.55, 0, pi/2, 0])
        human_desired_pose = np.array([0.35, -0.2, 0.55, 0, 0, 0])
        print("----desired pose: ", human_desired_pose)
        human.passive_DS(human_desired_pose, null_gain=human_null_gain, null_pos=human_null_pos)

    # Human Subject with Mesh
    elif agent == 'humanSubjectWithMesh':
        human = Robot(robot_name='humanSubjectWithMesh', controllable_joints = list(range(6, 14)))
        human.load(base_pose=[0, 0, 0.5, 0, 0, 0, 1])
        human_q_init = [0.0] * len(human.controllable_joints)
        human_q_init[0] = 1.3    
        # human_q_init = [1.3, 0, 0, 0.5, 0., 0., 0., pi/8, 0, 0]      
        human.set_joint_positions(human_q_init)
        print("Human robot loaded")

        human_null_pos  = human_q_init
        human_null_gain = [1, 1, 30, 50, 10, 20, 1, 1, 1, 1]
        human_null_gain = [10, 10, 1, 1, 1, 1, 10, 10, 10, 10]
        human_null_pos = None
        human_null_gain = None

        human_desired_pose = np.array([0.35, -0.2, 0.55, pi/2, -pi/2, 0])
        print("----desired pose: ", human_desired_pose)
        human.passive_DS(human_desired_pose)