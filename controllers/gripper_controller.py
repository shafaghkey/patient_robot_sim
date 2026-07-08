import numpy as np
import pybullet as p
import pybullet_data as pd
import os
import time
from utils.util_functions import *
from utils.util_human_rom import *
from utils.util_visualization_GUI import *


class Gripper:
    def __init__(self, gripper_name='gripper', controllable_joints=None, time_step=1e-3):
        self.gripper_name = gripper_name
        self.controllable_joints = controllable_joints
        self.gripper_id = None
        self.ee_index = None  # End-effector index
        self.lower_limits = None
        self.upper_limits = None
        self.time_step = time_step

    def load(self, base_pose=[0, 0, 0, 0, 0, 0, 1], connect_to=None):
        '''load the gripper in pybullet'''
        
        # Load the gripper URDF file
        urdf_file = 'agents/' + self.gripper_name + '.urdf'
        urdf_file = 'agents/robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f_85_model.urdf'
        base_pose, base_quat = base_pose[:3], base_pose[3:7]
        useFixedBase = True if connect_to is None else False  # Use fixed base if not connected to a simulation
        self.gripper_id = p.loadURDF(urdf_file, basePosition=base_pose, baseOrientation=base_quat, useFixedBase=useFixedBase)
        
        if self.controllable_joints is None:
            self.controllable_joints = list(range(p.getNumJoints(self.gripper_id)-1))
        self.ee_index = p.getNumJoints(self.gripper_id) - 1
        print(f"Gripper loaded with ID: {self.gripper_id}, Controllable Joints: {self.controllable_joints}, End-Effector Index: {self.ee_index}")

        for j in range(p.getNumJoints(self.gripper_id)):
        # for j in self.controllable_joints:
            joint_info = p.getJointInfo(self.gripper_id, j)
            print(f"Joint Index={joint_info[0]}: Name={joint_info[1]}, Type={joint_info[2]}, " 
                f"Link={joint_info[12].decode('utf-8')}, Lower Limit={joint_info[8]}, Upper Limit={joint_info[9]} ")
            
        self.lower_limits = [p.getJointInfo(self.gripper_id, j)[8] for j in self.controllable_joints]
        self.upper_limits = [p.getJointInfo(self.gripper_id, j)[9] for j in self.controllable_joints]
        self.lower_limits = [p.getJointInfo(self.gripper_id, j)[8] for j in list(range(p.getNumJoints(self.gripper_id)))]
        self.upper_limits = [p.getJointInfo(self.gripper_id, j)[9] for j in list(range(p.getNumJoints(self.gripper_id)))]
        print(f"Lower limits: {self.lower_limits}")
        print(f"Upper limits: {self.upper_limits}")

        # constraint: gripper joint 1 mirror gripper joint 0
        # self.mimic_gripper_joints(motor_joint_idx=0)
        
        # # Debug - pause to check if the urds and initial pose are correct
        # while True:
        #     # for j in range(p.getNumJoints(self.gripper_id)):
        #     #     p.resetJointState(self.gripper_id, j, 0)
        #     gripper_ee_state = p.getLinkState(gripper.gripper_id, gripper.ee_index)  # Get gripper end-effector state
        #     gripper_ee_pos, gripper_ee_quat = gripper_ee_state[0], gripper_ee_state[1]
        #     gripper_ee_pose = list(gripper_ee_pos) + list(gripper_ee_quat)  # Combine position and quaternion
        #     draw_pose_frame(gripper_ee_pose, length=0.1, lineWidth=3, life_time=0.2)  # Draw gripper end-effector frame                              
        #     p.stepSimulation()
        #     time.sleep(1e-3)

    def get_joint_states(self, joints='controllable'):
        '''get joint states (position, velocity, torque)
           joints = 'all' or 'controllable' (default)'''
        joints_to_use = self.controllable_joints if joints == 'controllable' else range(p.getNumJoints(self.gripper_id))
        joint_states = p.getJointStates(self.gripper_id, joints_to_use)
        if joints == 'all':
            joint_infos = [p.getJointInfo(self.gripper_id, i) for i in joints_to_use]
            joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        # print("Debug - Retrieved jointStates positions, velocities, torques:", len(joint_positions), len(joint_velocities), len(joint_torques))
        return joint_positions, joint_velocities, joint_torques

    def set_joint_angles(self, indices, angles, use_limits=True, velocities=0):
        for i, (j, a) in enumerate(zip(indices, angles)):
            p.resetJointState(self.gripper_id, jointIndex=j, 
                              targetValue=min(max(a, self.lower_limits[j]), self.upper_limits[j]) if use_limits else a, 
                              targetVelocity=velocities if type(velocities) in [int, float] else velocities[i],
                              )

    def set_grip(self, open_ratio= 0.0, force=500, set_instantly=False):
        '''set the gripper to a certain open ratio
        Args:
            open_ratio: float, 0.0 (fully closed) to 1.0 (fully open)
        '''
        if open_ratio < 0.0 or open_ratio > 1.0:
            raise ValueError("open_ratio must be between 0.0 and 1.0")

        p.setJointMotorControlArray(
            bodyUniqueId=self.gripper_id,
            jointIndices=self.controllable_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=np.array(self.upper_limits) * open_ratio,   
            # targetVelocities=[0] * len(self.controllable_joints),
            positionGains=[1] * len(self.controllable_joints),
            velocityGains=[1] * len(self.controllable_joints),
            forces=[force]*len(self.controllable_joints)
        )

        # p.setJointMotorControlArray(self.gripper_id, self.controllable_joints,
        #                         controlMode=p.TORQUE_CONTROL,
        #                         forces=np.array(self.upper_limits) * open_ratio * 100,
        #                         positionGains=[0]*len(self.controllable_joints),  # Critical for torque mode
        #                         velocityGains=[0]*len(self.controllable_joints)   # Disable implicit PD
        #                         )
        if set_instantly:
            self.set_joint_angles(self.controllable_joints, np.array(self.upper_limits) * open_ratio, use_limits=True)
            
        # p.stepSimulation()
        # time.sleep(self.time_step)

    def mimic_gripper_joints(self, motor_joint_idx=0):
        '''Set the gripper joints to mimic the motor joint (e.g., joint 10)'''
        if motor_joint_idx not in self.controllable_joints:
            raise ValueError(f"Motor joint {motor_joint_idx} is not in controllable joints: {self.controllable_joints}")
        
        '''Set the gripper joints to mimic each other'''
        for j in self.controllable_joints:
            joint_info = p.getJointInfo(self.gripper_id, j)
            if joint_info[2] != p.JOINT_FIXED and j != motor_joint_idx:
                p.setJointMotorControl2(self.gripper_id, j, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)
                c = p.createConstraint(self.gripper_id, motor_joint_idx, self.gripper_id, j, 
                                       jointType=p.JOINT_GEAR, jointAxis=[1, 0, 0], 
                                       parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
                if j in [2, 7]:
                    p.changeConstraint(c, gearRatio=-1, maxForce=10000,erp=0.2)  # Set gear ratio to -1 for mirroring
                else:
                    p.changeConstraint(c, gearRatio=1, maxForce=10000,erp=0.2)

if __name__ == "__main__":
    # Load the pybullet world
    GUI = True
    time_step = 1e-3
    n_iter = 100
    # client_id = 
    p.connect(p.GUI if GUI else p.DIRECT)
    p.setAdditionalSearchPath(pd.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(time_step)
    p.setPhysicsEngineParameter(fixedTimeStep=time_step, numSolverIterations=n_iter, numSubSteps=10)
    p.setRealTimeSimulation(False)
    
    # Load the gripper model
    urdf_file = 'agents/gripper.urdf'
    urdf_file = '/agents/robotiq_2f_85_gripper_visualization/urdf/robotiq_arg2f_85_model.urdf'

    # base_pose, base_quat = base_pose[:3], base_pose[3:7]
    # gripper = p.loadURDF(urdf_file, basePosition=base_pose, baseOrientation=base_quat, useFixedBase=True)

    gripper = Gripper('gripper', controllable_joints=[0, 2, 4, 5, 7, 9], time_step=time_step)  
    # gripper = Gripper('gripper', controllable_joints=[0], time_step=time_step)  
    gripper.load(base_pose=[0, 0, 0, 0, 0, 0, 1])  # Adjust base pose as needed

    # # Debug - pause to check if the urds and initial pose are correct
    # while True:
    #     p.stepSimulation()
    #     time.sleep(time_step)

    # for turning off link and joint damping
    for link_idx in range(p.getNumJoints(gripper.gripper_id)):
        p.changeDynamics(gripper.gripper_id, link_idx, linearDamping=0, angularDamping=0)
        p.changeDynamics(gripper.gripper_id, link_idx, maxJointVelocity=0.1)

    # Enable torque control
    p.setJointMotorControlArray(
        bodyUniqueId=gripper.gripper_id,
        jointIndices=gripper.controllable_joints,
        controlMode=p.VELOCITY_CONTROL, 
        forces=np.zeros(len(gripper.controllable_joints)),
    )

    # Enable contact
    # p.setCollisionFilterPair(gripper.gripper_id, gripper.gripper_id, -1, -1, enableCollision=0)  # Disable self-collision

    gripper.set_joint_angles(gripper.controllable_joints, [0]*len(gripper.controllable_joints), use_limits=True)
    
    t = 0.0
    while True:
        t += time_step
        # # Open the gripper fully
        # gripper.set_grip(1.0)
        # time.sleep(1.0)  # Wait for a second to observe the gripper state
        
        # # Close the gripper fully
        # gripper.set_grip(0.0)
        # time.sleep(1.0)

        # Gravity compensation: set the gripper to torque control mode
        # G_q_gripper = np.asarray(p.calculateInverseDynamics(gripper.gripper_id, 
        #                                                     gripper.controllable_joints, 
        #                                                     [0]*len(gripper.controllable_joints), 
        #                                                     [0]*len(gripper.controllable_joints)))
        joint_pos, joint_vel, joint_torques = gripper.get_joint_states(joints='all')
        G_q = np.asarray(p.calculateInverseDynamics(gripper.gripper_id,
                                                    joint_pos, [0]*len(joint_pos), [0]*len(joint_pos)))
        # G_q_gripper = G_q[gripper.controllable_joints]  # Get gravity compensation torques for controllable joints
        G_q_gripper = G_q
                                                            
        tau = np.zeros(len(gripper.controllable_joints))  # Initialize tau to zero
        tau = np.array(gripper.upper_limits)[gripper.controllable_joints] * 0.005 
        tau = np.array([1.2, 0.2, 0.1, 1.2, 0.2, 0.1])
        tau += G_q_gripper  # Add gravity compensation torques

        p.setJointMotorControlArray(
            bodyUniqueId=gripper.gripper_id,
            jointIndices=gripper.controllable_joints,
            controlMode=p.TORQUE_CONTROL,
            forces=tau,
            positionGains=[0]*len(gripper.controllable_joints),  # Critical for torque mode
            velocityGains=[0]*len(gripper.controllable_joints)   # Disable implicit PD
        )

        # Get the current joint states
        joint_positions, joint_velocities, joint_torques = gripper.get_joint_states('controllable')
        if t % 1 < 1e-3:  # Print every second
            print(f"{round(t,2)}  Positions: {[round(pos, 2) for pos in joint_positions]}  ",
                    f"Velocities: {[round(vel, 2) for vel in joint_velocities]}  ",
                    # f"Torques: {[round(torque, 2) for torque in joint_torques]}",
                    f"Torques: {[round(torque, 2) for torque in tau]}",
                    f"Gravity: {[round(g, 2) for g in G_q_gripper]}",
                    )

        p.stepSimulation()
        time.sleep(time_step)


    
    # p.disconnect()