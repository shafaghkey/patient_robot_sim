from math import pi
import numpy as np

# class HRIConfig:
# Environment / object
cup_pos = [0.55, -0.15, 0.6]
cup_euler = [pi/2, 0, 0]
cup_handle_offset = [0.0, -0.055, 0.07]
cup_handle_euler = [pi/2, -pi/2, 0]      # For target orientation
cup_color = (1, 0.5, 1, 0.9)
cup_scale = (0.05, 0.05, 0.05)

# Robot (iiwa + gripper)
kuka_base = [0.1, -1.05, 0.4, 0, 0, 0.707, 0.707]
robot_name = 'iiwa14_gripper_umi'
robot_arm_joints = list(range(1, 8))
robot_gripper_joints = [10, 12, 14, 15, 17, 19]
robot_ee_index = 8
robot_q_init = [-0.5, 0.5, 0, -pi/2, 0, 0, -pi/2]  #[-0.5, 0.5, 0, -pi/2, 0, 0, pi]
kuka_null_pos = None #[0, pi/4, 0, -1.5, 0, pi/4, pi/2]
kuka_null_gain = None #[5., 20., 30., 50., 10., 20., 1.]   #[1., 60, 10., 40, 5., 1., 1.]
kuka_damping_eigval = [100.0, 100.0, 100.0, 40., 40., 40.]   #[100.0, 200.0, 200.0, 10., 10., 10.]
kuka_k_lin_init = 5
kuka_k_ang_init = 2
# Robot joint pos limits: [[-2.97 -2.09 -2.97 -2.09 -2.97 -2.09 -3.05]
#                          [ 2.97  2.09  2.97  2.09  2.97  2.09  3.05]]
# Robot joint vel limits: [10. 10. 10. 10. 10. 10. 10.]

# Human
human_name = 'human_gazebo_shkey'
human_controllable_joints = list(range(6, 14))
# Optional home configuration edits (index:value)
human_home_overrides = {
    23: -1.3,   # Left shoulder rotx
    49: -pi/2,  # Right Hip
    68: -pi/2,  # Left Hip
    51:  pi/2,  # Right Knee
    70:  pi/2,  # Left Knee
}
human_init_states = {  # relative to controllable_joints ordering (example)
    1: -1.3,
}
human_null_pos = None
human_null_gain = None
human_damping_eigval = [200.0, 100.0, 100.0, 2, 2, 2]
human_k_lin_init = 30
human_k_ang_init = 4
human_role_observe = 1.0
human_role_grasp = 0.5
human_role_assist = 0.05

# Phase thresholds (iterations @ 1000Hz)
STUCK_THRESHOLD = 40
REACH_THRESHOLD = 20
CONTACT_THRESHOLD = 20
TASK_THRESHOLD = 20

# Tunables for stable physical interaction
CONTACT_FORCE_MAX = 25.0     # N, clamp contact force used by controllers
F_EXT_ALPHA = 0.9            # LPF for contact force (0=no filter, 0.9=strong smoothing)
HUMAN_TAU_LIMIT = 25.0       # Nm, per-joint clamp for human

# Impairment method: 'mid_pose' | 'ROM' | 'joint_limits'
human_impairment_method = 'mid_pose'

# Gripper torques
gripper_open_tau = [1.25, 0.2, 0.1, 1.25, 0.2, 0.1]
gripper_close_scale_grasp = 10
gripper_close_scale_assist = 10
gripper_open_scale = -10

# IIWA startup defaults used by the package's Gazebo helper.
IIWA_JOINT_NAMES = ['iiwa_joint_1', 'iiwa_joint_2', 'iiwa_joint_3', 'iiwa_joint_4', 'iiwa_joint_5', 'iiwa_joint_6', 'iiwa_joint_7']
IIWA_INIT_JOINTS = [0.0, 1.13446, 0.0, -1.39626, 0.0, -0.959931, 0.0]
IIWA_EE_INDEX = 9


