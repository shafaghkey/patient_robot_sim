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

class Human:
    def __init__(self, robot_name='human', controllable_joints=None, EE_index=None, 
                 env=None, time_step=1e-3):
        self.robot_name = robot_name
        self.robot_id = None
        self.controllable_joints = controllable_joints
        self.ee_index = EE_index
        self.env = env  # shared env
        if env is not None:
            self.time_step = env.time_step
        else: 
            self.time_step = time_step

    def load(self, base_pose=(0,0,0,0,0,0,1), use_fixed_base=True):
        base_pos, base_quat = base_pose[:3], base_pose[3:7]
        # Use shared connection (assumes env.connect() already called)
        if self.env is not None:
            self.robot_id = self.env.load_urdf(f"{self.robot_name}.urdf", base_pos, base_quat, use_fixed_base=use_fixed_base)
        else:
            if not p.isConnected():
                p.connect(p.GUI)
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setTimeStep(self.time_step)
                p.setGravity(0,0,-9.81)
            urdf_file = asset_path(f"{self.robot_name}.urdf")
            self.robot_id = p.loadURDF(urdf_file, basePosition=base_pos, baseOrientation=base_quat,
                                    useFixedBase=use_fixed_base)    #, flags=p.URDF_USE_INERTIA_FROM_FILE)


        if self.controllable_joints is None:
            self.controllable_joints = list(range(1, p.getNumJoints(self.robot_id)-1))
        if self.ee_index is None:
            self.ee_index = self.controllable_joints[-1]
        print('------------------------ \n ...Robot Initiated... robot =', self.robot_id, self.robot_name, ' \n------------------------')
        print('# All Joints:', p.getNumJoints(self.robot_id))
        print('# Controllable Joints:', self.controllable_joints.__len__(), self.controllable_joints)
        print('# End-effector:', self.ee_index, p.getJointInfo(self.robot_id, self.ee_index)[1].decode('utf-8'))

        for j in range(p.getNumJoints(self.robot_id)):
        # for j in self.controllable_joints:
            joint_info = p.getJointInfo(self.robot_id, j)
            print(f"Joint Index={joint_info[0]}: Name={joint_info[1]}, Type={joint_info[2]}, " 
                f"Link={joint_info[12].decode('utf-8')}, Lower Limit={joint_info[8]}, Upper Limit={joint_info[9]} ")

        self.fix_uncontrollable_joints()
        
    def fix_uncontrollable_joints(self):
        '''Fix all joints other than controllable joints'''
        for i in range(p.getNumJoints(self.robot_id)):
                    joint_info = p.getJointInfo(self.robot_id, i)
                    # joint_info[2] = 4   #p.JOINT_FIXED
                    if joint_info[2] != p.JOINT_FIXED:
                        if i not in self.controllable_joints:
                            p.changeDynamics(self.robot_id, joint_info[0], jointLowerLimit=0, jointUpperLimit=0, jointDamping=1000.0)  # Extreme damping resists movement
                
    def set_home_configuration(self, home_configuration=None):
        '''set the robot to its home configuration'''
        if home_configuration is None:
            home_configuration = [0] * p.getNumJoints(self.robot_id)  # Default home configuration
        # home_configuration = [state[0] for state in p.getJointStates(self.robot_id, range(p.getNumJoints(self.robot_id)))]
        for j in range(p.getNumJoints(self.robot_id)):
            p.resetJointState(self.robot_id, j, home_configuration[j])
        print("Robot set to home configuration.")

    def set_joint_positions(self, joint_positions):
        '''set the joint positions of the robot'''
        print("Setting joint positions:", joint_positions)
        for i, pos in enumerate(joint_positions):
            p.resetJointState(self.robot_id, self.controllable_joints[i], pos)

    def get_joint_states(self, joints='controllable'):
        '''get joint states (position, velocity, torque)
           joints = 'all' or 'controllable' (default)'''
        joints_to_use = self.controllable_joints if joints == 'controllable' else range(p.getNumJoints(self.robot_id))
        joint_states = p.getJointStates(self.robot_id, joints_to_use)
        if joints == 'all':
            joint_infos = [p.getJointInfo(self.robot_id, i) for i in joints_to_use]
            joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        # print("Debug - Retrieved jointStates positions, velocities, torques:", len(joint_positions), len(joint_velocities), len(joint_torques))
        return joint_positions, joint_velocities, joint_torques
            
    def get_jacobian(self, output='full'):
        joint_pos, join_vel , _ = self.get_joint_states(joints='all')
        # print("Debug - Joint positions:", joint_pos)
        ee_state = p.getLinkState(self.robot_id, self.ee_index, )
        jac_t, jac_r = p.calculateJacobian(self.robot_id, self.ee_index, ee_state[2],
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
            p.setJointMotorControlArray(self.robot_id, self.controllable_joints,
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
        M_q = np.asarray(p.calculateMassMatrix(self.robot_id, joint_pos))
        C_q = np.asarray(p.calculateInverseDynamics(self.robot_id, joint_pos, joint_vel, joint_torques)) #, [0.0]*n_dof)) - G_q
        G_q = np.asarray(p.calculateInverseDynamics(self.robot_id, joint_pos, [0]*len(joint_pos), [0]*len(joint_pos)))
        if len(self.controllable_joints) != len(joint_pos):
            M_q = M_q[np.ix_(self.controllable_joints, self.controllable_joints)]
            C_q = C_q[self.controllable_joints]
            G_q = G_q[self.controllable_joints]
        return M_q, C_q, G_q
    
    def get_joint_limits(body, joints):
        return [p.getJointInfo(body, j)[8:10] for j in joints]
    
    def world_to_base(self, pos_world, quat_world):
        """Convert world coordinates to robot base frame."""
        robot_base_pos, robot_base_quat = p.getBasePositionAndOrientation(self.robot_id)
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
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
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
        jac_t, jac_r = p.calculateJacobian(self.robot_id, poc_index, F_ext_point,
                                        list(joint_pos), [0.0]*len(joint_pos), [0.0]*len(joint_pos))
        J_t, J_r = np.asarray(jac_t), np.asarray(jac_r)
        if len(self.controllable_joints) != len(joint_pos):
            J_t = J_t[:, self.controllable_joints]  # only rows for controllable joints
            J_r = J_r[:, self.controllable_joints] 
        J = np.concatenate((J_t, J_r), axis=0)
        # print(f"Jacobian at contact point (link {poc_index}):\n{J.round(2)}")
        tau_ext = J.T @ F_ext_base
        return tau_ext

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
            F_ext_point = p.getLinkState(self.robot_id, self.ee_index)[0]  # End-effector position in world coordinates
            
        # Desired & End-effector pose
        des_pose = desired_pose
        if len(des_pose) == 6:
            des_pos, des_quat = des_pose[:3], p.getQuaternionFromEuler(des_pose[3:6])
        elif len(des_pose) == 7:
            des_pos, des_quat = des_pose[:3], des_pose[3:7]
        else:
            raise ValueError("Desired pose must be a 6D or 7D vector.")
        
        ee_state = p.getLinkState(self.robot_id, self.ee_index) # = link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot
        ee_pos, ee_quat = ee_state[0], ee_state[1]    # [x,y,z], [qx,qy,qz,qw] # or [4],[5]
        
        des_pos_base, des_quat_base = self.world_to_base(des_pos, des_quat)  
        ee_pos_base, ee_quat_base = self.world_to_base(ee_pos, ee_quat)  

        # # End-effector velocity
        q, dq, _ = self.get_joint_states()
        J = self.get_jacobian()
        ee_vel = J @ dq
        ee_vel_lin, ee_vel_ang = ee_vel[:3], p.getQuaternionFromEuler(ee_vel[3:])  # [vx, vy, vz], [qx, qy, qz, qw]

        # # Desired velocity (DS)
        # # # # Linear DS 
        lin_error = np.array(des_pos_base) - np.array(ee_pos_base) 
        DS_vel_ang_dir = lin_error / (np.linalg.norm(lin_error) + 1e-6)  # direction toward the target
        A_lin = k_lin*np.eye(3) if np.linalg.norm(lin_error) > 0.005 else np.zeros((3, 3))  # A matrix for linear DS
        ds_vel_lin = A_lin @ lin_error
        
        # # # # Angular DS  
        ang_error = np.array(des_quat_base) - np.array(ee_quat_base)
        DS_vel_ang_dir = DS_angular_direction(ee_quat_base, des_quat_base)  
        A_ang = k_ang * np.eye(3) if np.linalg.norm(ang_error) > 0.005 else np.zeros((3, 3))  # A matrix for angular DS
        ds_vel_ang = A_ang @ DS_vel_ang_dir
        
        # # Compute the wrench
        D_lin = update_damping_matrix_role(ds_vel_lin, damping_eigval[:3], role=role)  # damiping matrix (linear)
        wrench_lin = -D_lin @ (ee_vel_lin - ds_vel_lin)       # control output 

        D_ang = update_damping_matrix_role(ds_vel_ang, damping_eigval[3:])             # damiping matrix (angular)
        wrench_ang = -D_ang @ (ee_vel_ang[:3] - ds_vel_ang)   # control output (quat_xyz components)  

        wrench = np.concatenate((wrench_lin, wrench_ang), axis=0) 
        tau_task = J.T @ wrench

        # External forces compensation
        poc_index = 10 if self.robot_name == 'human_gazebo_shkey' else self.ee_index
        if F_ext is not None and np.linalg.norm(F_ext) > 0:
            tau_ext = self.compute_external_torques(F_ext, F_ext_point, poc_index=poc_index)
            tau_task += tau_ext

        # Gravity compensation
        Mq, Cq, Gq = self.get_dyn_matrices()
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
        # ee_pose = list(ee_pos) + list(ee_quat)
        # draw_pose_frame(ee_pose, length=0.1, lineWidth=2, life_time=0.2)
        # draw_pose_frame(des_pose, length=0.1, lineWidth=3, life_time=0.2)

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
    p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.9])
    p.setRealTimeSimulation(False)
    EE_index = None

    ## Load assets & agent
    # plane = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

    agent = 'human_gazebo_shkey'
    controllable_joints = list(range(6, 14))
    base_pose=[0, 0, 0.98, 0, 0, 0, 1]

    human = Human(robot_name=agent, controllable_joints=controllable_joints, EE_index=EE_index, time_step=time_step)
    human.load(base_pose=base_pose)

    home_config = [0.0] * p.getNumJoints(human.robot_id)  # Initialize all joints to 0
    home_config[23] = -1.3  # Left shoulder_rotx joint to -1.3 radians
    human.set_home_configuration(home_config)
    
    q_init = [0.0] * len(controllable_joints)
    q_init[1] = -1.3    # Test: pi/180 * -60
    human.set_joint_positions(q_init)

    desired_pose = np.array([0.55, -0.2, 1.12, pi/2, -pi/2, 0])
    # desired_pose = np.array([0.62, -0.2, 1.28, pi/2, -pi/2*1.3, 0])
    # desired_pose = np.array([0.6, -0.2, .8, pi/2, -pi/2, 0])
    
    null_pos, null_gain = None, None
    # null_pos  = q_init
    # null_pos = [1.2, -0.6, 0.2, 0.8, -1.2, -0.5, 0.3, 0.0]
    # null_gain = [10, 1, 10, 10, 10, 1, 1, 1]
    # null_gain = [5.0, 80.0, 10.0, 30.0, 5.0, 5.0, 2.0 ,2.0]

    damping_eigval = [250.0, 200.0, 200.0, 2., 5., 5.]      # [x y z roll pitch yaw] damping eigenvalues
    damping_eigval = [200.0, 100.0, 100.0, 2, 2, 2]
    # damping_eigval = [200, 100, 100, 10, 10, 10]
    k_lin, k_ang = 20, 5    

    # Learn the ROM for the human subjec
    human_rom = human_ROM_4d(subject=1, impaired_arm="R")
    human_rom.train_svm_model(scaled=False)
    # p.changeDynamics(human.robot_id, 6, jointLowerLimit=-0.9, jointUpperLimit= 0.4)
    # p.changeDynamics(human.robot_id, 7, jointLowerLimit=-1.8, jointUpperLimit=-0.9)

    n_dof = len(human.controllable_joints)

    # forward dynamics simulation loop
    # for turning off link and joint damping
    for link_idx in range(p.getNumJoints(human.robot_id)+1):
        p.changeDynamics(human.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        p.changeDynamics(human.robot_id, link_idx, maxJointVelocity=200)

    # for j in robot.controllable_joints:
    #     p.changeDynamics(robot.robot_id, j, jointLowerLimit=-2*pi, jointUpperLimit=2*pi)

    # Enable torque control
    p.setJointMotorControlArray(human.robot_id, human.controllable_joints,
                                p.VELOCITY_CONTROL, 
                                forces=np.zeros(n_dof))

    # define GUI sliders
    gui_sliders = GUIcontrol()
    ForceGUIids = gui_sliders.force(forces=np.zeros(n_dof), max_limit=10, min_limit=-10)        # Initial forces

    # self-collision avoidance
    p.setCollisionFilterPair(human.robot_id, human.robot_id, -1, -1, enableCollision=0)  # Disable self-collision for the robot

    while True:

        rom_tau = np.zeros(n_dof)
        q = np.array(human.get_joint_states()[0]) #* 180/pi  # Convert to degrees
        Gamma, Gamma_grad = human_rom.calc_Gamma_and_derivative(q[:4], condition='impaired')

        rom_tau[:4] = human_rom.enforce_ROM_tau(q[:4]) #* 1e2
        # TODO: either PD law (on q_dot) or QP (**DS_CBF)

        tau = human.passive_DS(desired_pose, null_gain=null_gain, null_pos=null_pos, damping_eigval=damping_eigval, 
                               k_lin=k_lin, k_ang=k_ang)
        # print(f"tau: {tau.round(1)}, rom_tau: {rom_tau.round(1)}, Gamma: {Gamma.round(2)}")
        # print(f"angles: {(np.array(robot.get_joint_states()[0]) ).round(1)}, Gamma: {Gamma.round(1)}")
        # tau += rom_tau

        p.setJointMotorControlArray(human.robot_id, human.controllable_joints,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=tau,
                                    positionGains=[0]*len(human.controllable_joints),   # Critical for torque mode
                                    velocityGains=[0]*len(human.controllable_joints)    # Disable implicit PD
                                    )

        p.stepSimulation()
        time.sleep(time_step)

