######################################################

import pybullet as p
import pybullet_data
import numpy as np
from math import pi
from scipy.spatial.transform import Rotation as R
import time
import os, sys
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from utils.pb_env_utils import *
from utils.humanROM_utils import *
from utils.pb_vis_GUI_utils import *

from controllers.human_controller_role import Human
from controllers.robot_controller_role import RobotGripper

import threading

import threading

# Load the pybullet world
GUI = True
time_step = 1e-3
p.connect(p.GUI if GUI else p.DIRECT)
configure_pybullet_search_paths()
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.setTimeStep(time_step)
p.setPhysicsEngineParameter(fixedTimeStep=time_step,
                            numSolverIterations=100,    # More accurate collisions
                            numSubSteps=10,             # More accurate physics simulation
                            enableConeFriction=1,       # Enable cone friction for better contact handling
                            contactBreakingThreshold=0.0001,
                            deterministicOverlappingPairs=0)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.9])
p.setRealTimeSimulation(False)

## Load assets & agents
plane = p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

# Load cup & table
visual_filename = "agents/objects/plastic_coffee_cup.obj"
collision_filename = "agents/objects/plastic_coffee_cup_vhacd.obj"
cup_scale= [0.05]*3
cup_pos, cup_quat = [0.55, -0.15, 1.05], p.getQuaternionFromEuler([pi/2, 0, 0])
tool_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=cup_scale, rgbaColor=[1, .5, 1, .9])
tool_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=cup_scale)
cup = p.createMultiBody(baseMass=0.0,  # Zero mass makes it immovable 
                        baseCollisionShapeIndex=tool_collision, baseVisualShapeIndex=tool_visual,
                        basePosition=cup_pos, baseOrientation=cup_quat,
                        useMaximalCoordinates=True)  # Better stability for static objects 
# table_cup = p.loadURDF("agents/table_cup.urdf", [0.7, -0.175, .1], useFixedBase=True)
target_pose = np.array([cup_pos[0], cup_pos[1]-.055, cup_pos[2]+.07, pi/2, -pi/2, 0])
target_pose = list([cup_pos[0], cup_pos[1]-.055, cup_pos[2]+.07]) + list(p.getQuaternionFromEuler([pi/2, -pi/2, 0]))

# Robot & table
kuka_base=[0.1, -1.05, 0.6, 0, 0, 0.707, 0.707]
table_kuka = p.loadURDF("agents/table_kuka.urdf", [kuka_base[0], kuka_base[1], 0], useFixedBase=True)
arm_joints, gripper_joints, EE_index = list(range(1, 8)), [10, 12, 14, 15, 17, 19], 8        # Robotiq_gripper
robot = RobotGripper(robot_name='iiwa14_gripper', arm_joints=arm_joints, gripper_joints=gripper_joints, EE_index=EE_index, time_step=time_step)
# robot = Robot(robot_name='iiwa14_gripper', arm_joints = list(range(1,8)), time_step=time_step)
robot.load(base_pose=kuka_base) 
robot_q_init = [-0.5, 0.52359878, 0, -1.58824962,  0, 0, pi]
robot_desired_pose = np.array([kuka_base[0]+0.2, kuka_base[1]+0.5, 0.8, 0, pi, pi/2])
robot_desired_pose = np.array([kuka_base[0]+0.3, kuka_base[1]+0.5, 0.8, 0, pi, pi/2])
# robot_q_init = [0, pi/4, 0, -2, 0, 0, 0]
robot.set_joint_positions(robot_q_init)
print("iiwa14 agent loaded")
kuka_null_pos = [0, pi/4, 0, -1.5, 0, pi/4, 0]
kuka_null_gain = [5.,20.0,30.,50.,10.,20.,1.]
kuka_null_gain = [1.,60,10.,40,5.,1.,1.]
kuka_damping_eigval = [100.0, 100.0, 250.0, 10., 10., 1.]
kuka_damping_eigval = [100.0, 200.0, 200.0, 10., 10., 10]

# Human
human = Human(robot_name='human_gazebo_shkey', controllable_joints = list(range(6, 14)), time_step=time_step)
human.load(base_pose=[0, 0, 0.98, 0, 0, 0, 1])
home_config = [0.0] * p.getNumJoints(human.robot)  # Initialize all joints to 0
home_config[23] = -1.3  # Left shoulder_rotx joint to -1.3 radians
human.set_home_configuration(home_config)
human_q_init = [0.0] * len(human.controllable_joints)
human_q_init[1] = -1.3    
human.set_joint_positions(human_q_init)
human_desired_pose = np.array(target_pose)
print("Human agent loaded")

human_null_pos  = human_q_init
# human_null_gain = [0.01, 10, 10, 10, 10, 10, 10, 10]
human_null_gain = [1, 1, 1, 1, 1, 1, 1, 1]
# human_null_gain = [1, 0.01, 0.01, 0.01, 0, 0, 0, 0]
human_null_pos, human_null_gain = None, None
human_damping_eigval = [200.0, 100.0, 100.0, 2, 2, 2]   #[250.0, 200.0, 200.0, 2., 2., 2.]

# # # Enforce joint limits for human
human_impairment_method = 'ROM'# or 'joint_limits' OR 'mid_pose'
if human_impairment_method == 'ROM':
    human_rom = human_ROM_4d(subject=3, impaired_arm="R")
    human_rom.train_svm_model(scaled=False)
elif human_impairment_method == 'joint_limits':
    p.changeDynamics(human.robot, 6, jointLowerLimit=-0.9, jointUpperLimit= 0.4)
    p.changeDynamics(human.robot, 7, jointLowerLimit=-1.8, jointUpperLimit=-0.9)
elif human_impairment_method == 'mid_pose':      # Set a different target to simulate joint limits 
    human_desired_pose[:3] = human_desired_pose[:3] + [-0.1, 0, -0.05] 
else:
    raise ValueError(f"Unknown impairment method: {human_impairment_method}")

# for turning off link and joint damping
for link_idx in range(p.getNumJoints(robot.robot)+1):
    p.changeDynamics(robot.robot, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
    p.changeDynamics(robot.robot, link_idx, maxJointVelocity=200)
    if link_idx in robot.gripper_joints:
        p.changeDynamics(robot.robot, link_idx, maxJointVelocity=0.1)

for link_idx in range(p.getNumJoints(human.robot)+1):
    p.changeDynamics(human.robot, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
    p.changeDynamics(human.robot, link_idx, maxJointVelocity=200)

# Enable torque control
p.setJointMotorControlArray(robot.robot, robot.arm_joints,
                            p.VELOCITY_CONTROL, 
                            forces=np.zeros(len(robot.arm_joints)))

p.setJointMotorControlArray(robot.robot, robot.gripper_joints,
                            p.VELOCITY_CONTROL,
                            forces=np.zeros(len(robot.gripper_joints)))

p.setJointMotorControlArray(human.robot, human.controllable_joints,
                            p.VELOCITY_CONTROL, 
                            forces=np.zeros(len(human.controllable_joints)))
 

p.setCollisionFilterPair(human.robot, cup, -1, -1, enableCollision=1)           # Enable collision between cup and human
p.setCollisionFilterPair(robot.robot, human.robot, -1, -1, enableCollision=0)   # Disable initial robot-human interaction
p.setCollisionFilterPair(human.robot, human.robot, -1, -1, enableCollision=0)   # Disable collision between agents

# Phase control variables
phase = "observe"
print("Starting in observing phase...")
human_stuck_counter = 0
reach_counter = 0
contact_counter = 0
lose_contact_counter = 0
prev_human_pos = None  
# Thresholds
STUCK_THRESHOLD = 40  # .04 sec at 1000Hz
REACH_THRESHOLD = 20  # Iterations to switch to grasp phase
CONTACT_THRESHOLD = 20

# robot_desired_pose = Initial pose of the robot end-effector
robot_ee_state = p.getLinkState(robot.robot, robot.ee_index)  
robot_initial_pose = list(robot_ee_state[0]) + list(p.getEulerFromQuaternion(robot_ee_state[1])) 

F_ext, F_ext_point = None, None  
kuka_k_lin, kuka_k_ang = 30, 4
human_k_lin, human_k_ang = 30, 4
human_damping_eigval = [200.0, 100.0, 100.0, 2, 2, 2]   #[250.0, 200.0, 200.0, 2., 2., 2.]
# kuka_damping_eigval = [200.0, 100.0, 100.0, 10, 10, 10]  #[100.0, 100.0, 250.0, 10., 10., 1.]
kuka_damping_eigval = [100.0, 200.0, 200.0, 10., 10., 10.]


t = 0
while True:
    t += time_step                                
    
    # If target is moving, update the target pose 
    # cup_pos, cup_quat = p.getBasePositionAndOrientation(cup)  # cup position and orientation
    # cup_to_handle_pos, cup_to_handle_quat = [0.0, -0.055, 0.07], [0.0, -0.707, 0.0, 0.707]        # Cup-to-handle transform
    # target_pose = p.multiplyTransforms(cup_pos, cup_quat, cup_to_handle_pos, cup_to_handle_quat)  # Desired pose of the human end-effector

    human_ee_state = p.getLinkState(human.robot, human.ee_index)  # human end-effector state
    human_poc_pose = get_poc_pose(human.robot)      # poc (point-of-contact) on human forearm
    robot_ee_state = p.getLinkState(robot.robot, robot.ee_index)  

    human_target_err_pos = np.array(target_pose[:3]) - np.array(human_ee_state[0])   
    human_target_err_quat = np.array(target_pose[3]) - np.array(human_ee_state[1])   
    human_target_err = list(human_target_err_pos) + list(human_target_err_quat)     # human_ee to target error

    robot_poc_err_pos = np.array(human_poc_pose[:3]) - np.array(robot_ee_state[0])   
    robot_poc_err_quat = np.array(human_poc_pose[3:]) - np.array(robot_ee_state[1])  
    robot_poc_err = list(robot_poc_err_pos) + list(robot_poc_err_quat)              # robot_ee to poc error

    if np.linalg.norm(human_target_err_pos) < 1e-3 and np.linalg.norm(human_target_err_quat) < 1e-2:
        cup_contacts = p.getContactPoints(cup, human.robot)
        if cup_contacts:
            for c in cup_contacts:
                draw_contact_point(c[5], radius=0.05, color=[0, 1, 0], life_time=0.1)
        
    contacts = p.getContactPoints(robot.robot, human.robot)
    if contacts:
        for c in contacts:
            draw_contact_point(c[5], radius=0.01, color=[1, 0, 0], life_time=0.1)  # Draw contact point in red 

    if phase == "observe":
        human_role = 1.0        # Human damping is the default value
        robot_desired_pose = robot_initial_pose
        tau_gripper = np.array([1.2, 0.2, 0.1, 1.2, 0.2, 0.1]) * -10        # Fully open

        if prev_human_pos is not None and prev_human_quat is not None:
            movement_lin = np.linalg.norm(np.array(human_ee_state[0]) - prev_human_pos)
            movement_ang = np.linalg.norm(np.array(human_ee_state[1]) - np.array(prev_human_quat))
            # If human is not moving and is not close to desired pose
            if movement_lin < 1e-3 and movement_ang < 1e-2 and \
                np.linalg.norm(human_target_err_pos) > 1e-3 and np.linalg.norm(human_target_err_quat) > 1e-2:
                human_stuck_counter += 1
                print(f"OBSERVE --- stuck_counter+ = {human_stuck_counter}... Human not moving... Movement = {1e3 * movement_lin:.2f}mm, {movement_ang*180/pi:.2f}deg")
                # # Check joint limits or ROM constraints
                # joint_pos = human.get_joint_states()[0]  # current joint positions
                # joint_info = [p.getJointInfo(human.robot, j) for j in human.controllable_joints]
                # limits = [p.getJointInfo(human.robot, j)[8:10] for j in human.controllable_joints if p.getJointInfo(human.robot, j)[2] != p.JOINT_FIXED]
                # at_limit = any(pos <= low+0.01 or pos >= upp-0.01 
                #             for (low, upp), pos in zip(limits, joint_pos))
                # # Gamma = rom_r.calc_Gamma(joint_pos)
                # # if Gamma < 0.0:  # If Gamma is close to zero, human is at joint limits
                # if at_limit:
                #     human_stuck_counter += 1
                #     print(f"OBSERVE --- stuck_counter+ = {human_stuck_counter}... Human not moving... Movement = {1e3 * movement:.2f}mm + Joints at limit")
                # else:
                #     human_stuck_counter = max(0, human_stuck_counter-1)
                #     print(f"OBSERVE --- stuck_counter- = {human_stuck_counter}... Human not moving... Movement = {1e3 * movement:.2f}mm")
            else:
                human_stuck_counter = max(0, human_stuck_counter-1)
                print(f"OBSERVE --- stuck_counter- = {human_stuck_counter}... Human moving... Movement = {1e3 * movement_lin:.2f}mm, {movement_ang*180/pi:.2f}deg")

        # Transition to approach phase
        if human_stuck_counter >= STUCK_THRESHOLD:
            print("Human stuck! Robot starts approaching...")
            phase = "approach"
            human_stuck_counter = 0
            human_poc_pose_init = human_poc_pose # Store initial poc pose for faster approach phase
            human_poc_pose_mid = human_poc_pose  

      
    elif phase == "approach":
        human_role = 1.0        # Human damping is the default value
        tau_gripper = np.array([1.2, 0.2, 0.1, 1.2, 0.2, 0.1]) * -10        # Fully open

        # Offset the end-effector position by 5cm along its z-axis
        robot_ee_z_axis  = np.array(p.getMatrixFromQuaternion(robot_ee_state[1])).reshape(3,3)[:, 2]
        robot_ee_offset = robot_ee_state[0] + 0.05 * robot_ee_z_axis 
        draw_contact_point(robot_ee_offset, radius=0.01, color=[0, 0, 0], life_time=0.1) 

        if np.linalg.norm(robot_poc_err_pos) > 0.16:
            print(f"APPROACH -- Fast movement towards human forearm... Error: {np.linalg.norm(robot_poc_err_pos):.4f}m")
            robot_desired_pose = human_poc_pose_init + [0, -0.08, 0, 0, 0, 0, 0]  
            human_poc_pose_mid = human_poc_pose 
            # human_poc_pose_mid[:3] -= 0.05 * robot_ee_z_axis
            kuka_k_lin, kuka_k_ang = 20, 5
        elif np.linalg.norm(robot_poc_err_pos) > 0.12:
            print(f"APPROACH -- Mid-speed movement towards human forearm... Error: {np.linalg.norm(robot_poc_err_pos):.4f}m")
            robot_desired_pose = human_poc_pose_mid + [0, -0.05, 0, 0, 0, 0, 0]
            human_poc_pose_init = human_poc_pose  # Update initial poc pose in case robot got far
            kuka_k_lin, kuka_k_ang = 20, 5
            # kuka_k_ang = 4
        elif np.linalg.norm(robot_poc_err_pos) > 0.08:
            print(f"APPROACH -- Slow movement towards human forearm... Error: {np.linalg.norm(robot_poc_err_pos):.4f}m")
            robot_desired_pose = human_poc_pose
            robot_desired_pose[:3] -= 0.02 * robot_ee_z_axis  # Offset the poc position by 2cm along robot's z-axis
            human_poc_pose_init = human_poc_pose
            human_poc_pose_mid = human_poc_pose  
        else:
            robot_desired_pose = human_poc_pose
            human_poc_pose_init = human_poc_pose
            human_poc_pose_mid = human_poc_pose
            kuka_k_lin, kuka_k_ang = 20, 5
            if np.linalg.norm(robot_poc_err_pos) < 0.03 and np.linalg.norm(robot_poc_err_quat) < 0.12:
                p.setCollisionFilterPair(robot.robot, human.robot, -1, -1, enableCollision=1)
                reach_counter += 1
                print(f"APPROACH -- reach_counter+ = {reach_counter}... Robot close to human forearm: {1e3 * np.linalg.norm(robot_poc_err_pos):.2f}mm, {np.linalg.norm(robot_poc_err_quat):.3f}")
            else:
                reach_counter = max(0, reach_counter-1)
                print(f"APPROACH -- reach_counter- = {reach_counter}... Error to human forearm: {1e3 * np.linalg.norm(robot_poc_err_pos):.2f}mm, {np.linalg.norm(robot_poc_err_quat):.4f}")
        
        # Start grasping if close enough to human forearm
        if reach_counter > REACH_THRESHOLD/2:
            print("APPROACH -- Robot reached human forearm! Transitioning to grasp phase.")
            tau_gripper = np.array([1.2, 0.2, 0.1, 1.2, 0.2, 0.1])
            # p.setJointMotorControlArray(robot.robot, robot.gripper_joints,
            #                 p.VELOCITY_CONTROL,
            #                 forces=np.zeros(len(robot.gripper_joints)))

            if reach_counter >= REACH_THRESHOLD:
                phase = "grasp"
                reach_counter = 0
                # p.setCollisionFilterPair(robot.robot, human.robot, -1, -1, enableCollision=1) 
                # Increase contact stiffness for better force transfer
                # for body in [robot.robot, human.robot]:
                p.changeDynamics(human.robot, -1, contactStiffness=2e5, contactDamping=1.0)
                p.changeDynamics(human.robot, -1, lateralFriction=1.0, spinningFriction=0.1, rollingFriction=0.1)
                

    elif phase == "grasp":
        human_role = 0.5        # Human damping is the default value
        tau_gripper = np.array([1.2, 0.2, 0.1, 1.2, 0.2, 0.1]) * 5      # Close the gripper
        # Monitor for sustained contact
        if not contacts:
            lose_contact_counter += 1
            F_ext, F_ext_point = None, None  # Reset external force if no contact
            print(f"GRASP ----- lose_contact_counter = {lose_contact_counter}... No contact detected.")
            if lose_contact_counter >= CONTACT_THRESHOLD:
                print("GRASP ----- Contact lost! Transitioning to approach phase.")
                # p.setCollisionFilterPair(robot.robot, human.robot, -1, -1, enableCollision=0)
                phase = "approach"
                lose_contact_counter = 0
                contact_counter = 0
                tau_gripper = np.array([1.2, 0.2, 0.1, 1.2, 0.2, 0.1]) * -10     # Open the gripper
        else:
            lose_contact_counter = max(0, lose_contact_counter-1)  
            # # Print contact details
            # gripper_joint_states = p.getJointStates(robot.robot, robot.gripper_joints)
            # gripper_joint_positions = [state[0] for state in gripper_joint_states]
            # gripper_joint_torques = [state[3] for state in gripper_joint_states]
            print(f"GRASP ----- contact_counter = {contact_counter}: between: Robot={[c[3] for c in contacts]}, Human={[c[4] for c in contacts]} :::: Normal force: {contacts[0][9]:.2f}N")
            
            # # Check if contact involves the robot end-effector and human forearm
            if all(link in [c[3] for c in contacts] for link in [13, 14, 18, 19]) and all(c[4] == 10 for c in contacts):
                contact_counter += 1
            else:
                contact_counter = max(0, contact_counter-1)
            
        if contact_counter >= CONTACT_THRESHOLD:
            print("Contact established! Initiating assistance.")
            phase = "assist"
            contact_counter = 0

    elif phase == "assist":
        human_role = 0.05        # Human follower (human damping is low to allow robot to assist)
        kuka_k_lin, kuka_k_ang = 50, 5
        tau_gripper = np.array([1.2, 0.2, 0.1, 1.2, 0.2, 0.1]) * 10     # Close the gripper firmly check the maximum force
        # # Robot desired pose in the assist phase
        # T_human_poc_to_ee = inv(T_poc) @ T_ee
        # robot_desired_pose = T_human_desired_pose @ inv(T_human_poc_to_ee)
        inv_human_poc_pos, inv_human_poc_quat = p.invertTransform(human_poc_pose[:3], human_poc_pose[3:])
        human_poc_to_ee_pos, human_poc_to_ee_quat = p.multiplyTransforms(inv_human_poc_pos, inv_human_poc_quat, human_ee_state[0], human_ee_state[1])
        human_ee_to_poc_pos, human_ee_to_poc_quat = p.invertTransform(human_poc_to_ee_pos, human_poc_to_ee_quat)
        robot_desired_pos, robot_desired_quat = p.multiplyTransforms(target_pose[:3], target_pose[3:],
                                                                    human_ee_to_poc_pos, human_ee_to_poc_quat)
        robot_desired_pose = list(robot_desired_pos) + list(robot_desired_quat)

        if not contacts:
            lose_contact_counter += 1
            F_ext, F_ext_point = None, None  # Reset external force if no contact
            print(f"ASSIST ----- lose_contact_counter = {lose_contact_counter}... No contact detected.")
            if lose_contact_counter >= CONTACT_THRESHOLD:
                print("ASSIST ----- Contact lost! Transitioning to approach phase.")
                # p.setCollisionFilterPair(robot.robot, human.robot, -1, -1, enableCollision=0)
                phase = "approach"
                lose_contact_counter = 0
        else:
            lose_contact_counter = max(0, lose_contact_counter-1)  # Reset contact counter if in contact
            # human moves with robot's contact force
            F_ext = np.zeros(3)  # Initialize external force
            for c in contacts:
                normal_force = np.array(c[7]) * c[9]  # Normal force (towards robot) = normal vec on B * contact force (element-wise multiplication)
                lateral_friction1 = np.array(c[11]) * c[10]  # Lateral friction force (tangent to the contact surface)
                lateral_friction2 = np.array(c[13]) * c[12]  # Lateral friction force (perpendicular to the first)
                F_ext_c = normal_force + lateral_friction1 + lateral_friction2  # Total contact force at this point
                F_ext += F_ext_c  # Sum forces from all contact points
                draw_contact_Force(F_ext_c, c[5], length=0.03, color=[0, 0, 0], life_time=0.2)  # Draw contact force in black
                # print(f"Normal: {np.linalg.norm(F_ext):.2f}N, Latfriction1: {np.linalg.norm(lateral_friction1):.2f}N, Latfriction2: {np.linalg.norm(lateral_friction2):.2f}N")

            F_ext_point = human_poc_pose[:3]    #p.getLinkState(human.robot, 10)[0]
            print(f"ASSIST ----- External force: {F_ext.round(2)}N, contact points: Robot={[c[3] for c in contacts]}")    
            # at {np.array(F_ext_point).round(2)}m")

    else:
        print("Unknown phase! Switch to observing...")
        phase = "observe"


    # Human control (always active)
    tau_human = human.passive_DS(human_desired_pose, null_gain=human_null_gain, null_pos=human_null_pos, damping_eigval=human_damping_eigval,
                                    k_lin=human_k_lin, k_ang=human_k_ang, role=human_role,
                                 F_ext=F_ext if F_ext is not None else None, F_ext_point=F_ext_point
                                 )  
    
    if human_impairment_method == 'ROM':
        tau_rom = np.zeros_like(tau_human)
        q = np.array(human.get_joint_states()[0]) #* 180/pi  # Convert to degrees
        Gamma, Gamma_grad = human_rom.calc_Gamma_and_derivative(q[:4], condition='impaired')

        # Idea: compensate for the torque projection on the Gamma gradient direction
        if Gamma < -0.01:  # If Gamma is close to zero, human is at joint
            e_gamma_grad = np.zeros_like(tau_human)
            e_gamma_grad[:len(Gamma_grad)] = Gamma_grad / np.linalg.norm(Gamma_grad) if np.linalg.norm(Gamma_grad) > 0 else np.zeros_like(Gamma_grad)
            # projection of the human torque on the Gamma gradient direction
            # print('np.dot(tau_human, e_gamma_grad):', np.dot(tau_human, e_gamma_grad), '   e_gamma_grad:', e_gamma_grad)
            tau_human_proj = np.dot(tau_human, e_gamma_grad) * e_gamma_grad
            tau_human += tau_human_proj  # Remove the projection from the human torque
            print('Gamma=', Gamma)
        # tau_rom[:4] = human_rom.enforce_ROM_tau(q[:4]) #* 1e2
        # tau_human += tau_rom  # Add ROM torque to human control

    p.setJointMotorControlArray(human.robot, human.controllable_joints,
                                controlMode = p.TORQUE_CONTROL, 
                                forces = tau_human,
                                positionGains=[0]*len(human.controllable_joints),   # Critical for torque mode
                                velocityGains=[0]*len(human.controllable_joints)    # Disable implicit PD
                                )
    
    # Robot control (always active)
    tau_robot = robot.passive_DS(robot_desired_pose, null_gain=kuka_null_gain, null_pos=kuka_null_pos, 
                                 damping_eigval=kuka_damping_eigval, k_lin=kuka_k_lin, k_ang=kuka_k_ang,
                                 F_ext=F_ext if F_ext is not None else None, F_ext_point=F_ext_point
                                 )  
    
    p.setJointMotorControlArray(robot.robot, robot.arm_joints,
                               controlMode=p.TORQUE_CONTROL, 
                               forces=tau_robot,
                               positionGains=[0]*len(robot.arm_joints),    # Critical for torque mode
                               velocityGains=[0]*len(robot.arm_joints)    # Disable implicit PD
                            )

    # # Gripper control    
     # tau_gripper = np.array(robot.gripper_upper_limits) * 0.005
    G_q_gripper = np.array(robot.get_gripper_gravity_trq())
    tau_gripper += G_q_gripper 
    # print(f"tau_gripper (with gravity compensation): {tau_gripper.round(2)}")
    p.setJointMotorControlArray(robot.robot, robot.gripper_joints,
                                controlMode=p.TORQUE_CONTROL, 
                                forces=tau_gripper,
                                positionGains=[0]*len(robot.gripper_joints),  # Critical for torque mode
                                velocityGains=[0]*len(robot.gripper_joints)   # Disable implicit PD
                                )

    prev_human_pos = human_ee_state[0]   # Update previous position
    prev_human_quat = human_ee_state[1]  # Update previous quaternion

    p.stepSimulation()
    time.sleep(time_step)


# Columns of p.getContactPoints()
# Index	Name	            Description
# 0	    contactFlag	        Internal flag for contact status/type (not commonly used in user scripts).
# 1	    bodyUniqueIdA   	Unique ID of the first body in contact (e.g., robot).
# 2	    bodyUniqueIdB	    Unique ID of the second body in contact (e.g., human).
# 3	    linkIndexA	        Link index on body A involved in the contact (-1 for base).
# 4	    linkIndexB	        Link index on body B involved in the contact (-1 for base).
# 5	    positionOnAInWS 	[x, y, z] position of the contact point on body A (in world coordinates).
# 6	    positionOnBInWS 	[x, y, z] position of the contact point on body B (in world coordinates).
# 7	    contactNormalOnB	[x, y, z] normal vector at the contact point on body B (points toward body A).
# 8	    contactDistance	    Penetration depth (negative = penetration, positive = separation).
# 9	    normalForce	        Magnitude of the normal force at the contact point (in Newtons).
# 10	lateralFriction1	Magnitude of the first lateral friction force (tangent to the contact surface).
# 11	lateralFrictionDir1	[x, y, z] direction vector of the first lateral friction force.
# 12	lateralFriction2	Magnitude of the second lateral friction force (perpendicular to the first).
# 13	lateralFrictionDir2	[x, y, z] direction vector of the second lateral friction force.



# TODO:
# filter the human motion when the robot is far. distribution of the poses and reach the mean
# robot can move faster when far from the forearm, and slower when close to the forearm ---
# # preshapeing the grasping pose: coupling reaching and grasping for the robot

    # # Enable collision between agents
    # # p.setCollisionFilterPair(robot.robot, human.robot, -1, -1, enableCollision=1)  # -1 = all links
    # for robot_link in range(robot.ee_index-3, robot.ee_index):
    #     for human_link in range(human.ee_index-5, human.ee_index):
    #         p.setCollisionFilterPair(robot.robot, human.robot, 
    #                                     robot_link, human_link, 
    #                                     enableCollision=1,
    #                                     )

# Initially disable interaction forces between agents
# p.changeDynamics(human.robot, -1,
#                 contactStiffness=1e4,  # Responsive force generation
#                 contactDamping=0.1,    # Minimal velocity damping
#                 lateralFriction=0.5,   # Realistic sliding
#                 contactProcessingThreshold=0,
#                 frictionAnchor=0)      # Prevent pre-contact friction effects
# p.changeDynamics(human.robot, -1, frictionAnchor=1)  # frictionAnchor=1: Prevent sliding
# p.changeDynamics(human.robot, -1, contactStiffness=0, contactDamping=0)

# p.changeDynamics(cup, -1, contactStiffness=1e4, contactDamping=1e3, contactProcessingThreshold=0)  # Better collision resolution
# # p.changeDynamics(table_cup, -1, lateralFriction=1, spinningFriction=0.5, rollingFriction=0.5, frictionAnchor=1) # frictionAnchor=1: Prevent sliding
# # p.changeDynamics(table_cup, -1, contactStiffness=0.1, contactDamping=0.1, frictionAnchor=0)       # Prevent pre-contact friction effects

# p.setCollisionFilterPair(robot.robot, robot.robot, -1, -1, enableCollision=0)  # Disable self-collision
# p.setCollisionFilterPair(human.robot, human.robot, -1, -1, enableCollision=0)  # Disable self-collision

    # if phase == "grasp" or phase == "assist":
    #     tau_gripper = np.array([0.8, 0.1, 0.05, 0.8, 0.1, 0.05])  # Apply some force to the gripper joints
    #     # tau_gripper = np.array(robot.gripper_upper_limits) * 0.05
    # else:
    #     tau_gripper = np.zeros(len(robot.gripper_joints))

#     # Close the gripper if in grasp or assist phase
    #     tau_gripper[0] = -10  
    #     tau_gripper = np.array(robot.gripper_upper_limits) * 0.005  # Apply some force to the gripper joints
    #     # tau_gripper = -np.array(robot.gripper_upper_limits) * 0.1  # Apply some force to the gripper joints
    #     gripper_open_ratio = 0.0  # Fully closed
    # else:
    #     # tau_gripper = np.zeros(len(robot.gripper_joints))  # No force applied
    #     gripper_open_ratio = 1.0
    # # # p.setJointMotorControlArray(robot.robot, robot.gripper_joints,
    # # #                             controlMode=p.POSITION_CONTROL, 
    # # #                             targetPositions=np.array(robot.gripper_upper_limits) * gripper_open_ratio,  
    # # #                             # positionGains=[0.05],    # Slow position control
    # # #                             positionGains=[1.0],     # Fast velocity control
    # # #                             # forces=[500],            # Apply force to close gripper
    # # #                             )
    # # p.setJointMotorControlArray(robot.robot, [10],
    # #                             controlMode=p.TORQUE_CONTROL,
    # #                             forces=[-100],
    # #                             positionGains=[0],  # Critical for torque mode
    # #                             velocityGains=[0]   # Disable implicit PD
    # #                             )