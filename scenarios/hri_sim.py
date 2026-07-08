######################################################
# Controller for robot-assisted feeding task with impaired human
import pybullet as pb
import pybullet_data
import numpy as np
from math import pi
from scipy.spatial.transform import Rotation as R
import time
# Add util path
import os, sys
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
from utils.pb_env_utils import *
from utils.sim_functions_utils import *
from utils.humanROM_utils import *
from utils.pb_vis_GUI_utils import *
from config.hri_config import *
from controllers.human_controller_role import Human
from controllers.robot_controller_role import RobotGripper


# --- Load world / assets ---
env = PyBulletEnv()
physics_client = env.connect(gui=True)
env.load_world()

stool_human = env.load_urdf("table_human_stool.urdf", use_fixed_base=True)
table_kuka = env.load_urdf("table_kuka_seated.urdf", base_pos=[kuka_base[0], kuka_base[1], 0])

cup_quat = pb.getQuaternionFromEuler(cup_euler)
cup = env.load_mesh_object("plastic_coffee_cup.obj",
                           collision_rel="plastic_coffee_cup_vhacd.obj",
                           pos=cup_pos, quat=cup_quat, mass=0.0,
                           rgba=cup_color, scale=cup_scale)
target_pose = list(np.array(cup_pos) + np.array(cup_handle_offset)) + list(pb.getQuaternionFromEuler(cup_handle_euler))

# --- Robot setup ---
robot = RobotGripper(robot_name=robot_name,
                     arm_joints=robot_arm_joints,
                     gripper_joints=robot_gripper_joints,
                     EE_index=robot_ee_index,
                     env=env)
robot.load(base_pose=kuka_base)
robot.set_joint_positions(robot_q_init)

# --- Human setup ---
human = Human(robot_name=human_name,
              controllable_joints=human_controllable_joints,
              env=env)
human.load(base_pose=[0, 0, 0.5, 0, 0, 0, 1])

home_config = [0.0] * pb.getNumJoints(human.robot_id)
for idx, val in human_home_overrides.items():
    if idx < len(home_config):
        home_config[idx] = val
human.set_home_configuration(home_config)

human_q_init = [0.0] * len(human.controllable_joints)
for rel_idx, val in human_init_states.items():
    if rel_idx < len(human_q_init):
        human_q_init[rel_idx] = val
human.set_joint_positions(human_q_init)
human_desired_pose = np.array(target_pose)

# # # Enforce joint limits for human
human_impairment_method = human_impairment_method # 'ROM' or 'joint_limits' OR 'mid_pose'
if human_impairment_method == 'ROM':
    human_rom = human_ROM_4d(subject=1, impaired_arm="R")
    human_rom.train_svm_model(scaled=False)
elif human_impairment_method == 'joint_limits':
    pb.changeDynamics(human.robot_id, 6, jointLowerLimit=-0.9, jointUpperLimit= 0.4)
    pb.changeDynamics(human.robot_id, 7, jointLowerLimit=-1.8, jointUpperLimit=-0.9)
elif human_impairment_method == 'mid_pose':      # Set a different target to simulate joint limits 
    human_desired_pose[:3] = human_desired_pose[:3] + [-0.1, 0, -0.05] 
else:
    raise ValueError(f"Unknown impairment method: {human_impairment_method}")

# for turning off link and joint damping
for link_idx in range(pb.getNumJoints(robot.robot_id)+1):
    pb.changeDynamics(robot.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
    pb.changeDynamics(robot.robot_id, link_idx, maxJointVelocity=200)
    # soften contact/friction and remove bounce
    pb.changeDynamics(robot.robot_id, link_idx, restitution=0.0, lateralFriction=0.6, spinningFriction=0.1, rollingFriction=0.1)
    if link_idx in robot.gripper_joints:
        pb.changeDynamics(robot.robot_id, link_idx, maxJointVelocity=0.1)

for link_idx in range(pb.getNumJoints(human.robot_id)+1):
    pb.changeDynamics(human.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
    pb.changeDynamics(human.robot_id, link_idx, maxJointVelocity=200)
    pb.changeDynamics(human.robot_id, link_idx, restitution=0.0, lateralFriction=0.6, spinningFriction=0.1, rollingFriction=0.1)

# Enable torque control
disable_default_motors(robot.robot_id, robot.arm_joints + robot.gripper_joints)
disable_default_motors(human.robot_id, human.controllable_joints)

pb.setCollisionFilterPair(human.robot_id, cup, -1, -1, enableCollision=1)           # Enable collision between cup and human
pb.setCollisionFilterPair(robot.robot_id, human.robot_id, -1, -1, enableCollision=0)   # Disable initial robot-human interaction
pb.setCollisionFilterPair(human.robot_id, human.robot_id, -1, -1, enableCollision=0)   # Disable collision between agents

kuka_k_lin, kuka_k_ang = kuka_k_lin_init, kuka_k_ang_init
human_k_lin, human_k_ang = human_k_lin_init, human_k_ang_init

# Phase control variables
phase = "observe"
human_stuck_counter = 0
reach_counter = 0
contact_counter = 0
lose_contact_counter = 0
task_done_counter = 0
prev_human_pos = None  

# robot_desired_pose = Initial pose of the robot end-effector
robot_ee_state = pb.getLinkState(robot.robot_id, robot.ee_index)  
robot_initial_pose = list(robot_ee_state[0]) + list(pb.getEulerFromQuaternion(robot_ee_state[1])) 

F_ext, F_ext_point = None, None  


t = 0
while True:
    t += env.time_step                                
    
    # If target is moving, update the target pose 
    # cup_pos, cup_quat = pb.getBasePositionAndOrientation(cup)  # cup position and orientation
    # cup_to_handle_pos, cup_to_handle_quat = [0.0, -0.055, 0.07], [0.0, -0.707, 0.0, 0.707]        # Cup-to-handle transform
    # target_pose = pb.multiplyTransforms(cup_pos, cup_quat, cup_to_handle_pos, cup_to_handle_quat)  # Desired pose of the human end-effector

    human_ee_state = pb.getLinkState(human.robot_id, human.ee_index)  # (pos, quat) -> indices 0,1
    human_ee_pos, human_ee_quat = list(human_ee_state[0]), list(human_ee_state[1])
    human_poc_pose = get_poc_pose(human.robot_id)  # Expect [x,y,z,qx,qy,qz,qw]
    robot_ee_state = pb.getLinkState(robot.robot_id, robot.ee_index)
    robot_ee_pos, robot_ee_quat = list(robot_ee_state[0]), list(robot_ee_state[1])

    # Errors (pos / quat)
    human_target_err_pos = np.array(target_pose[:3]) - np.array(human_ee_pos)
    human_target_err_quat_vec = quat_diff(target_pose[3:], human_ee_quat)
    human_target_err = list(human_target_err_pos) + list(human_target_err_quat_vec)

    robot_poc_err_pos = np.array(human_poc_pose[:3]) - np.array(robot_ee_pos)
    robot_poc_err_quat_vec = quat_diff(human_poc_pose[3:], robot_ee_quat)
    robot_poc_err = list(robot_poc_err_pos) + list(robot_poc_err_quat_vec)

    if np.linalg.norm(human_target_err_pos) < 1e-3 and np.linalg.norm(human_target_err_quat_vec) < 1e-2:
        cup_contacts = pb.getContactPoints(cup, human.robot_id)
        if cup_contacts:
            for c in cup_contacts:
                draw_contact_point(c[5], radius=0.05, color=[0, 1, 0], life_time=0.1)
        
    contacts = pb.getContactPoints(robot.robot_id, human.robot_id)
    if contacts:
        for c in contacts:
            draw_contact_point(c[5], radius=0.01, color=[1, 0, 0], life_time=0.1)  # Draw contact point in red 

    # Check the task is done
    if np.linalg.norm(human_target_err_pos) < 1e-3 and np.linalg.norm(human_target_err_quat_vec) < 1e-2:
        task_done_counter += 1
        if task_done_counter >= TASK_THRESHOLD:
            print("Task completed successfully... Human reached the target!")
            pb.setTimeStep(0)        # pause the simulation
    else:
        task_done_counter = max(0, task_done_counter-1)  
       

    if phase == "observe":
        human_role = human_role_observe
        robot_desired_pose = robot_initial_pose
        tau_gripper = np.array(gripper_open_tau) * gripper_open_scale        # Fully open
        kuka_k_lin, kuka_k_ang = kuka_k_lin_init, kuka_k_ang_init

        if prev_human_pos is not None and prev_human_quat is not None:
            movement_lin = np.linalg.norm(np.array(human_ee_state[0]) - prev_human_pos)
            movement_ang = np.linalg.norm(np.array(human_ee_state[1]) - np.array(prev_human_quat))
            # If human is not moving and is not close to desired pose
            if movement_lin < 1e-3 and movement_ang < 1e-2 and \
                np.linalg.norm(human_target_err_pos) > 1e-3 and np.linalg.norm(human_target_err_quat_vec) > 1e-2:
                human_stuck_counter += 1
                print(f"OBSERVE --- stuck_counter+ = {human_stuck_counter}... Human not moving... Movement = {1e3 * movement_lin:.2f}mm, {movement_ang*180/pi:.2f}deg")
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
        human_role = human_role_observe
        tau_gripper = np.array(gripper_open_tau) * gripper_open_scale        # Fully open

        # Offset the end-effector position by 5cm along its z-axis
        robot_ee_z_axis  = np.array(pb.getMatrixFromQuaternion(robot_ee_state[1])).reshape(3,3)[:, 2]
        robot_ee_offset = robot_ee_state[0] + 0.05 * robot_ee_z_axis 
        draw_contact_point(robot_ee_offset, radius=0.01, color=[0, 0, 0], life_time=0.1) 

        if np.linalg.norm(robot_poc_err_pos) > 0.15:
            print(f"APPROACH -- Fast movement towards human forearm... Error: {np.linalg.norm(robot_poc_err_pos):.4f}m")
            # robot_desired_pose = human_poc_pose
            # robot_desired_pose[:3] -= 0.15 * robot_ee_z_axis
            robot_desired_pose = add_pose_offset_xyz_quat(human_poc_pose, [0, -0.10, 0])
            human_poc_pose_mid = human_poc_pose 
            # human_poc_pose_mid[:3] -= 0.05 * robot_ee_z_axis
            kuka_k_lin, kuka_k_ang = kuka_k_lin_init, kuka_k_ang_init
        elif np.linalg.norm(robot_poc_err_pos) > 0.12:
            print(f"APPROACH -- Mid-speed movement towards human forearm... Error: {np.linalg.norm(robot_poc_err_pos):.4f}m")
            robot_desired_pose = add_pose_offset_xyz_quat(human_poc_pose, [0, -0.08, 0])
            human_poc_pose_init = human_poc_pose  # Update initial poc pose in case robot got far
            kuka_k_lin, kuka_k_ang = kuka_k_lin_init, kuka_k_ang_init
            # kuka_k_ang = 4
        elif np.linalg.norm(robot_poc_err_pos) > 0.05:
            print(f"APPROACH -- Slow movement towards human forearm... Error: {np.linalg.norm(robot_poc_err_pos):.4f}m")
            # robot_desired_pose = human_poc_pose
            # robot_desired_pose[:3] -= 0.02 * robot_ee_z_axis  # Offset the poc position by 2cm along robot ee's z-axis
            robot_desired_pose = add_pose_offset_xyz_quat(human_poc_pose, [0, -0.04, 0])
            human_poc_pose_init = human_poc_pose
            human_poc_pose_mid = human_poc_pose  
        else:
            robot_desired_pose = human_poc_pose.copy()
            human_poc_pose_init = human_poc_pose
            human_poc_pose_mid = human_poc_pose
            kuka_k_lin, kuka_k_ang = kuka_k_lin_init, kuka_k_ang_init
            if np.linalg.norm(robot_poc_err_pos) < 0.02 and np.linalg.norm(robot_poc_err_quat_vec) < 0.22:
                pb.setCollisionFilterPair(robot.robot_id, human.robot_id, -1, -1, enableCollision=1)
                reach_counter += 1
                print(f"APPROACH -- reach_counter+ = {reach_counter}... Robot close to human forearm: {1e3 * np.linalg.norm(robot_poc_err_pos):.2f}mm, {np.linalg.norm(robot_poc_err_quat_vec):.3f}")
            else:
                reach_counter = max(0, reach_counter-1)
                print(f"APPROACH -- reach_counter- = {reach_counter}... Error to human forearm: {1e3 * np.linalg.norm(robot_poc_err_pos):.2f}mm, {np.linalg.norm(robot_poc_err_quat_vec):.4f}")
        
        # Start grasping if close enough to human forearm
        if reach_counter > REACH_THRESHOLD/2:
            print("APPROACH -- Robot reached human forearm! Transitioning to grasp phase.")
            tau_gripper = np.array(gripper_open_tau)
            if reach_counter >= REACH_THRESHOLD:
                phase = "grasp"
                reach_counter = 0
                # pb.setCollisionFilterPair(robot.robot_id, human.robot_id, -1, -1, enableCollision=1) 
                # Increase contact stiffness for better force transfer
                # for body in [robot.robot_id, human.robot_id]:
                # pb.changeDynamics(human.robot_id, -1, contactStiffness=2e5, contactDamping=1.0)
                # pb.changeDynamics(human.robot_id, -1, lateralFriction=1.0, spinningFriction=0.1, rollingFriction=0.1)
                

    elif phase == "grasp":
        human_role = human_role_grasp
        tau_gripper = np.array(gripper_open_tau) * gripper_close_scale_grasp     # Close the gripper
        # Monitor for sustained contact
        if not contacts:
            lose_contact_counter += 1
            F_ext, F_ext_point = None, None  # Reset external force if no contact
            print(f"GRASP ----- lose_contact_counter = {lose_contact_counter}... No contact detected.")
            if lose_contact_counter >= CONTACT_THRESHOLD:
                print("GRASP ----- Contact lost! Transitioning to approach phase.")
                # pb.setCollisionFilterPair(robot.robot_id, human.robot_id, -1, -1, enableCollision=0)
                phase = "approach"
                lose_contact_counter = 0
                contact_counter = 0
                tau_gripper = np.array(gripper_open_tau) * gripper_open_scale     # Open the gripper
        else:
            lose_contact_counter = max(0, lose_contact_counter-1)  
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
            F_ext_filt = np.zeros(3)  # initialize LPF state

    elif phase == "assist":
        human_role = human_role_assist
        human_k_lin, human_k_ang = 0.1*human_k_lin_init, 0.1*human_k_ang_init
        kuka_k_lin, kuka_k_ang = kuka_k_lin_init, kuka_k_ang_init
        tau_gripper = np.array(gripper_open_tau) * gripper_close_scale_assist     # Close the gripper firmly check the maximum force
        # # Robot desired pose in the assist phase
        # T_human_poc_to_ee = inv(T_poc) @ T_ee
        # robot_desired_pose = T_human_desired_pose @ inv(T_human_poc_to_ee)
        inv_human_poc_pos, inv_human_poc_quat = pb.invertTransform(human_poc_pose[:3], human_poc_pose[3:])
        human_poc_to_ee_pos, human_poc_to_ee_quat = pb.multiplyTransforms(
            inv_human_poc_pos, inv_human_poc_quat, human_ee_pos, human_ee_quat
        )
        human_ee_to_poc_pos, human_ee_to_poc_quat = pb.invertTransform(human_poc_to_ee_pos, human_poc_to_ee_quat)
        robot_desired_pos, robot_desired_quat = pb.multiplyTransforms(
            target_pose[:3], target_pose[3:], human_ee_to_poc_pos, human_ee_to_poc_quat
        )
        robot_desired_pose = list(robot_desired_pos) + list(robot_desired_quat)

        if not contacts:
            lose_contact_counter += 1
            F_ext, F_ext_point = None, None
            print(f"ASSIST ----- lose_contact_counter = {lose_contact_counter}... No contact detected.")
            if lose_contact_counter >= CONTACT_THRESHOLD:
                print("ASSIST ----- Contact lost! Transitioning to approach phase.")
                # pb.setCollisionFilterPair(robot.robot_id, human.robot_id, -1, -1, enableCollision=0)
                phase = "grasp"
                lose_contact_counter = 0
                human_k_lin, human_k_ang = human_k_lin_init, human_k_ang_init
        else:
            lose_contact_counter = max(0, lose_contact_counter-1)
            # Sum contact forces
            F_ext_raw = np.zeros(3)
            for c in contacts:
                normal_force = np.array(c[7]) * c[9]
                lat1 = np.array(c[11]) * c[10]
                lat2 = np.array(c[13]) * c[12]
                F_ext_c = normal_force + lat1 + lat2
                F_ext_raw += F_ext_c
                # draw_contact_Force(F_ext_c, c[5], length=0.03, color=[0, 0, 0], life_time=0.2)
            # Low-pass filter and clamp the external force used by controllers
            F_ext_filt = F_EXT_ALPHA * F_ext_filt + (1.0 - F_EXT_ALPHA) * F_ext_raw
            F_ext = np.clip(F_ext_filt, -CONTACT_FORCE_MAX, CONTACT_FORCE_MAX)
            F_ext_point = human_poc_pose[:3]
            # draw_contact_Force(-F_ext, F_ext_point, length=0.02, color=[0, 0, 0], life_time=0.2)
            print(f"ASSIST ----- External force (filt/clamped): {F_ext.round(2)}N, contacts: {[c[3] for c in contacts]}")

    else:
        print("Unknown phase! Switch to observing...")
        phase = "observe"


    # Human control (always active)
    tau_human = human.passive_DS(human_desired_pose, null_gain=human_null_gain, null_pos=human_null_pos, 
                                 damping_eigval=human_damping_eigval,
                                 k_lin=human_k_lin, k_ang=human_k_ang, role=human_role,
                                 F_ext=None if F_ext is not None else None, F_ext_point=F_ext_point
                                 )  
    
    if human_impairment_method == 'ROM':
        tau_rom = np.zeros_like(tau_human)
        q = np.array(human.get_joint_states()[0]) #* 180/pi  # Convert to degrees
        Gamma, Gamma_grad = human_rom.calc_Gamma_and_derivative(q[:4], condition='impaired')
        tau_rom[:4] = human_rom.enforce_ROM_tau(q[:4]) #* 1e2
        tau_human += tau_rom  # Add ROM torque to human control

    tau_robot = robot.passive_DS(robot_desired_pose, null_gain=kuka_null_gain, null_pos=kuka_null_pos, 
                                 damping_eigval=kuka_damping_eigval, 
                                 k_lin=kuka_k_lin, k_ang=kuka_k_ang,
                                 F_ext=F_ext if F_ext is not None else None, F_ext_point=F_ext_point
                                 )  

    apply_cmd_torque(human.robot_id, human.controllable_joints, tau_human)
    apply_cmd_torque(robot.robot_id, robot.arm_joints, tau_robot)
    apply_cmd_torque(robot.robot_id, robot.gripper_joints, tau_gripper+robot.get_gripper_gravity_trq())

    # Throttle verbose prints 
    if int(t / env.time_step) % 200 == 0:
        print(f"[t={t:.3f}] phase={phase} | human_err_pos={np.linalg.norm(human_target_err_pos):.4f}")

    prev_human_pos = human_ee_pos
    prev_human_quat = human_ee_quat

    pb.stepSimulation()
    time.sleep(env.time_step)


# Columns of pb.getContactPoints()
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
    # # pb.setCollisionFilterPair(robot.robot_id, human.robot_id, -1, -1, enableCollision=1)  # -1 = all links
    # for robot_link in range(robot.ee_index-3, robot.ee_index):
    #     for human_link in range(human.ee_index-5, human.ee_index):
    #         pb.setCollisionFilterPair(robot.robot_id, human.robot_id, 
    #                                     robot_link, human_link, 
    #                                     enableCollision=1,
    #                                     )

# Initially disable interaction forces between agents
# pb.changeDynamics(human.robot_id, -1,
#                 contactStiffness=1e4,  # Responsive force generation
#                 contactDamping=0.1,    # Minimal velocity damping
#                 lateralFriction=0.5,   # Realistic sliding
#                 contactProcessingThreshold=0,
#                 frictionAnchor=0)      # Prevent pre-contact friction effects
# pb.changeDynamics(human.robot_id, -1, frictionAnchor=1)  # frictionAnchor=1: Prevent sliding
# pb.changeDynamics(human.robot_id, -1, contactStiffness=0, contactDamping=0)

# pb.changeDynamics(cup, -1, contactStiffness=1e4, contactDamping=1e3, contactProcessingThreshold=0)  # Better collision resolution
# # pb.changeDynamics(table_cup, -1, lateralFriction=1, spinningFriction=0.5, rollingFriction=0.5, frictionAnchor=1) # frictionAnchor=1: Prevent sliding
# # pb.changeDynamics(table_cup, -1, contactStiffness=0.1, contactDamping=0.1, frictionAnchor=0)       # Prevent pre-contact friction effects

# pb.setCollisionFilterPair(robot.robot_id, robot.robot_id, -1, -1, enableCollision=0)  # Disable self-collision
# pb.setCollisionFilterPair(human.robot_id, human.robot_id, -1, -1, enableCollision=0)  # Disable self-collision

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
    # # # pb.setJointMotorControlArray(robot.robot_id, robot.gripper_joints,
    # # #                             controlMode=pb.POSITION_CONTROL, 
    # # #                             targetPositions=np.array(robot.gripper_upper_limits) * gripper_open_ratio,  
    # # #                             # positionGains=[0.05],    # Slow position control
    # # #                             positionGains=[1.0],     # Fast velocity control
    # # #                             # forces=[500],            # Apply force to close gripper
    # # #                             )
    # # pb.setJointMotorControlArray(robot.robot_id, [10],
    # #                             controlMode=pb.TORQUE_CONTROL,
    # #                             forces=[-100],
    # #                             positionGains=[0],  # Critical for torque mode
    # #                             velocityGains=[0]   # Disable implicit PD
    # #                             )