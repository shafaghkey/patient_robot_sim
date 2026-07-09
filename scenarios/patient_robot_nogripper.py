######################################################

import pybullet as p
import pybullet_data
import numpy as np
from math import pi
from scipy.spatial.transform import Rotation as R
import time
import os, sys
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
from utils.util_functions import *
from utils.util_human_rom import *
from utils.util_visualization_GUI import *
from controllers.robot_nogripper_controller import Robot

import threading
from pybullet_utils.bullet_client import BulletClient


# Load the pybullet world
GUI = True
time_step = 1e-3
n_iter = 50
p.connect(p.GUI if GUI else p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -9.81)
p.setTimeStep(time_step)
# p.setPhysicsEngineParameter(fixedTimeStep=time_step, numSolverIterations=n_iter, numSubSteps=10)
p.setPhysicsEngineParameter(numSolverIterations=100,  # More accurate collisions
                            enableConeFriction=1, # Enable cone friction for better contact handling
                            contactBreakingThreshold=0.0001,
                            deterministicOverlappingPairs=1)
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

# Robot & table
kuka_base=[0.1, -1.05, 0.6, 0, 0, 0.707, 0.707]
table_kuka = p.loadURDF("agents/table_kuka.urdf", [kuka_base[0], kuka_base[1], 0], useFixedBase=True)
robot = Robot(robot_name='iiwa14', EE_index=8, time_step=time_step)
# robot = Robot(robot_name='iiwa14_gripper', controllable_joints = list(range(1,8)), time_step=time_step)
robot.load(base_pose=kuka_base) 
robot_q_init = [-0.5, 0.52359878, 0, -1.58824962,  0, 0, 0]
robot_desired_pose = np.array([kuka_base[0]+0.2, kuka_base[1]+0.5, 0.8, 0, pi, pi/2])
robot_desired_pose = np.array([kuka_base[0]+0.3, kuka_base[1]+0.5, 0.8, 0, pi, pi/2])
# robot_q_init = [0, pi/4, 0, -2, 0, 0, 0]
robot.set_joint_positions(robot_q_init)
print("iiwa14 agent loaded")
kuka_null_pos = [0, pi/4, 0, -1.5, 0, pi/4, 0]
kuka_null_gain = [5.,20.0,30.,50.,10.,20.,1.]
kuka_damping_eigval = [100.0, 100.0, 250.0, 10., 10., 1.]

# for turning off link and joint damping
for link_idx in range(p.getNumJoints(robot.robot)+1):
    p.changeDynamics(robot.robot, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
    p.changeDynamics(robot.robot, link_idx, maxJointVelocity=200)

# Enable torque control
p.setJointMotorControlArray(robot.robot, robot.controllable_joints,
                            p.VELOCITY_CONTROL, 
                            forces=np.zeros(len(robot.controllable_joints)))


# Human
human = Robot(robot_name='humanSubjectWithMesh', controllable_joints = list(range(6, 14)), time_step=time_step)
human.load(base_pose=[0, 0, 0.98, 0, 0, 0, 1])
home_config = [0.0] * p.getNumJoints(human.robot)  # Initialize all joints to 0
home_config[23] = -1.3  # Left shoulder_rotx joint to -1.3 radians
human.set_home_configuration(home_config)
human_q_init = [0.0] * len(human.controllable_joints)
human_q_init[0] = 1.3    
human.set_joint_positions(human_q_init)
human_desired_pose = np.array([cup_pos[0], cup_pos[1]-.055, cup_pos[2]+.07, pi/2, -pi/2, 0])
human_desired_pose = list([cup_pos[0], cup_pos[1]-.055, cup_pos[2]+.07]) + list(p.getQuaternionFromEuler([pi/2, -pi/2, 0]))
print("Human agent loaded")
human_null_pos  = human_q_init
# human_null_gain = [0.01, 10, 10, 10, 10, 10, 10, 10]
human_null_gain = [1, 1, 1, 1, 1, 1, 1, 1]
# human_null_gain = [1, 0.01, 0.01, 0.01, 0, 0, 0, 0]
human_null_pos, human_null_gain = None, None
human_damping_eigval = [250.0, 200.0, 200.0, 10., 10., 1.]

for link_idx in range(p.getNumJoints(human.robot)+1):
    p.changeDynamics(human.robot, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
    p.changeDynamics(human.robot, link_idx, maxJointVelocity=200)

p.setJointMotorControlArray(human.robot, human.controllable_joints,
                            p.VELOCITY_CONTROL, 
                            forces=np.zeros(len(human.controllable_joints)))

# Enforce joint limits for human
# Joint Index=8: Name=b'jRightShoulder_rotz', Type=0, Link=RightUpperArm, Lower Limit=-0.785398, Upper Limit=3.14159
p.changeDynamics(human.robot, 8, jointLowerLimit=-0.785398, jointUpperLimit=0.4)
# # Joint Index=10: Name=b'jRightElbow_rotz', Type=0, Link=RightForeArm, Lower Limit=0.0, Upper Limit=2.53073
# p.changeDynamics(human.robot, 10, jointLowerLimit=0.0, jointUpperLimit=1.53073)


p.setCollisionFilterPair(human.robot, cup, -1, -1, enableCollision=1)           # Enable collision between cup and human
p.setCollisionFilterPair(robot.robot, human.robot, -1, -1, enableCollision=0)   # Disable initial robot-human interaction

p.setCollisionFilterPair(human.robot, human.robot, -1, -1, enableCollision=1)  # Disable collision between agents

# Phase control variables
phase = "observing"
print("Starting in observing phase...")
stuck_counter = 0
contact_counter = 0
lose_contact_counter = 0
prev_human_pos = None
F_ext, F_ext_point = None, None  # Initialize external force and point of application

# Thresholds
STUCK_THRESHOLD = 40  # .04 sec at 1000Hz
CONTACT_THRESHOLD = 10
MOVEMENT_THRESHOLD = 0.010  # 10mm

# robot_desired_pose = Initial pose of the robot end-effector
ee_state = p.getLinkState(robot.robot, robot.ee_index)  
robot_initial_pose = list(ee_state[0]) + list(p.getEulerFromQuaternion(ee_state[1])) 

t = 0
while True:
    t += time_step                                    
    # Control for human 
    # cup_pos, cup_quat = p.getBasePositionAndOrientation(cup)  # Get cup position and orientation
    # cup_to_handle_pos, cup_to_handle_quat = [0.0, -0.055, 0.07], [0.0, -0.707, 0.0, 0.707]  # Cup-to-handle transform
    # human_desired_pose = p.multiplyTransforms(cup_pos, cup_quat, cup_to_handle_pos, cup_to_handle_quat)  # Desired pose of the human end-effector

    tau_human = human.passive_DS(human_desired_pose, null_gain=human_null_gain, null_pos=human_null_pos, damping_eigval=human_damping_eigval,
                                 F_ext=F_ext, F_ext_point=F_ext_point)   
    # Apply torques simultaneously
    p.setJointMotorControlArray(human.robot, human.controllable_joints,
                                controlMode = p.TORQUE_CONTROL, 
                                forces = tau_human,
                                positionGains=[0]*len(human.controllable_joints),   # Critical for torque mode
                                velocityGains=[0]*len(human.controllable_joints)    # Disable implicit PD
                                )

    # Control for robot
    human_ee_state = p.getLinkState(human.robot, human.ee_index)  # Get human end-effector state
    human_poc_pose = get_poc_pose(human.robot)      # Get point-of-contact on human forearm

    human_ee_err_pos = human_desired_pose[:3] - np.array(human_ee_state[0])   # Get error position
    human_ee_err_quat = p.getDifferenceQuaternion(human_desired_pose[3:], human_ee_state[1])  # Get error quaternion
    human_ee_err = list(human_ee_err_pos) + list(human_ee_err_quat)  # Combine position and quaternion errors
    
    if phase == "observing":
        robot_desired_pose = robot_initial_pose
        # Monitor human progress
        if prev_human_pos is not None:
            movement = np.linalg.norm(np.array(human_ee_state[0]) - prev_human_pos)
            # If human is not moving and is not close to desired pose
            if movement < MOVEMENT_THRESHOLD and np.linalg.norm(human_ee_err_pos) > 1e-2:  
                # Check joint limits
                joint_pos = human.get_joint_states()[0]  # Get current joint positions
                joint_info = [p.getJointInfo(human.robot, j) for j in human.controllable_joints]
                limits = [p.getJointInfo(human.robot, j)[8:10] for j in human.controllable_joints if p.getJointInfo(human.robot, j)[2] != p.JOINT_FIXED]
                at_limit = any(pos <= low+0.01 or pos >= upp-0.01 
                            for (low, upp), pos in zip(limits, joint_pos))

                if at_limit:
                    print(f"stuck_counter = {stuck_counter}... Human not moving... Joints at limit")
                    stuck_counter += 1
                else:
                    stuck_counter = max(0, stuck_counter-1)
            else:
                stuck_counter = max(0, stuck_counter-1)

        # Transition to approach phase
        if stuck_counter >= STUCK_THRESHOLD:
            print("Human stuck! Robot starts approaching...")
            p.setCollisionFilterPair(robot.robot, human.robot, -1, -1, enableCollision=1)
            phase = "approaching"
            
        prev_human_pos = human_ee_state[0]  # Update previous position
        
    elif phase == "approaching":
        robot_desired_pose = human_poc_pose
        # robot_pos = p.getLinkState(robot.robot, robot.ee_index)[0]  # Get robot end-effector state
        # Monitor for sustained contact
        contacts = p.getContactPoints(robot.robot, human.robot)
        # Check if contact involves the robot end-effector and human forearm
        if contacts and contacts[0][3] == robot.ee_index-1 and contacts[0][4] == 10:
        # if contacts:
            print("Robot reached human forearm! Monitoring contact...")
            contact_counter += 1
            # Print contact details
            print(f"contact_counter = {contact_counter}: between: Robot={contacts[0][3]}, Human={contacts[0][4]} :::: Normal force: {contacts[0][9]:.2f}N")
            # print(f"Contact position: {contacts[0][5]}")                # Contact point coordinates on body A's surface (world frame)
            # print(f"Contact distance: {contacts[0][8]}")                # Penetration depth (negative if penetrating)
            # print(f"Contact normal force: {contacts[0][9]:.2f}N")       # Magnitude of normal force (in Newtons)
            # print(f"Contact normal on human vec: {contacts[0][7]}")     # Normal vector on body B's surface (points toward body A)
            # print(f"Contact lateralFriction1: {contacts[0][10]:.2f}")   # Magnitude of lateral friction force (1)
            # print(f"Contact lateralFriction1 vec: {contacts[0][11]:}")  # Vector of lateral friction force (1)
            # print(f"Contact lateralFriction2: {contacts[0][12]:.2f}")   # Magnitude of lateral friction force (2)
            # print(f"Contact lateralFriction2 vec: {contacts[0][13]}")   # Vector of lateral friction force (2)
            # print("----------------------------------------------")
            
            if contact_counter >= CONTACT_THRESHOLD:
                print("Contact established! Initiating assistance.")
                # T_human_poc_to_ee = inv(T_poc) @ T_ee
                inv_human_poc_pos, inv_human_poc_quat = p.invertTransform(human_poc_pose[:3], human_poc_pose[3:])
                human_poc_to_ee_pos, human_poc_to_ee_quat = p.multiplyTransforms(inv_human_poc_pos, inv_human_poc_quat, human_ee_state[0], human_ee_state[1])
                # robot_desired_pose = T_human_desired_pose @ inv(T_human_poc_to_ee)
                human_ee_to_poc_pos, human_ee_to_poc_quat = p.invertTransform(human_poc_to_ee_pos, human_poc_to_ee_quat)
                robot_desired_pos, robot_desired_quat = p.multiplyTransforms(human_desired_pose[:3], human_desired_pose[3:],
                                                                            human_ee_to_poc_pos, human_ee_to_poc_quat)
                robot_desired_pose = list(robot_desired_pos) + list(robot_desired_quat)
                # Increase contact stiffness for better force transfer
                for body in [robot.robot, human.robot]:
                    p.changeDynamics(body, -1, 
                                    contactStiffness=2e5, 
                                    contactDamping=0.5)
                phase = "assisting"
                contact_counter = 0
        else:
            contact_counter = max(0, contact_counter-1)
            
    elif phase == "assisting":
        # # Maintain target at cup position
        # robot_desired_pose[:3] = cup_pos  # Keep updating in case cup moves.
        contacts = p.getContactPoints(robot.robot, human.robot)
        if not contacts:
            lose_contact_counter += 1
            F_ext, F_ext_point = None, None  # Reset external force if no contact
        else:
            lose_contact_counter = max(0, lose_contact_counter-1)  # Reset contact counter if no contact
            # human moves with robot's contact force
            normal_force = np.array(contacts[0][7]) * contacts[0][9]  # Normal force = normal vec on B * contact force (element-wise multiplication) 
            lateral_friction1 = np.array(contacts[0][11]) * contacts[0][10]  # Lateral friction force (1)
            lateral_friction2 = np.array(contacts[0][13]) * contacts[0][12]  # Lateral friction force (2)
            F_ext = normal_force + lateral_friction1 + lateral_friction2  # Total external force
            F_ext_point = np.array(contacts[0][5])  # Contact point position in world frame
            print(f"lose_contact_counter = {lose_contact_counter}: between: Robot={contacts[0][3]}, Human={contacts[0][4]} :::: Total force: {F_ext}N")
        
        if lose_contact_counter >= CONTACT_THRESHOLD:
            print("Contact lost! Transitioning to approching phase.")
            # p.setCollisionFilterPair(robot.robot, human.robot, -1, -1, enableCollision=0)
            phase = "approaching"

    else:
        print(f"Unknown phase: {phase}. Resetting to observing phase.")
        phase = "observing"
        stuck_counter = 0
        contact_counter = 0
        prev_human_pos = None
        p.setCollisionFilterPair(robot.robot, human.robot, -1, -1, enableCollision=0)
        robot_desired_pose = robot_initial_pose.copy()

    # Robot control (always active)
    tau_robot = robot.passive_DS(robot_desired_pose, null_gain=kuka_null_gain, null_pos=kuka_null_pos, damping_eigval=kuka_damping_eigval,
                                 F_ext=-F_ext if F_ext is not None else None
                                 , F_ext_point=F_ext_point)  
    
    p.setJointMotorControlArray(robot.robot, robot.controllable_joints,
                               controlMode=p.TORQUE_CONTROL, 
                               forces=tau_robot,
                               positionGains=[0]*len(robot.controllable_joints),    # Critical for torque mode
                               velocityGains=[0]*len(robot.controllable_joints)    # Disable implicit PD
                            )

    p.stepSimulation()
    time.sleep(time_step)


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