#!/usr/bin/env python3
"""human_sim_node

PyBullet-based human patient simulator. Runs the Human controller in a ROS node
to simulate impaired-arm dynamics for HRI experiments. Used in
multi_agent_sim_test.launch.

Publishes:
  /human/joint_states  (JointState) — simulated human joint positions and velocities
"""
import rospy
import pybullet as pb
import numpy as np
import tf
import threading
from sensor_msgs.msg import JointState

import os, sys
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
from config.hri_config import *
from controllers.human_controller_role import Human
from utils.pb_vis_GUI_utils import *
from utils.transforms import quat_diff
from geometry_msgs.msg import Pose
from utils.ros_msg_utils import create_pose_msg

MOCAP = False

def get_poc_pose(robot, forearm_index=10, dist_from_forearm=0.0, forearm_dim=0.0):
    """
    Calculate the point of contact (PoC) and its orientation between two links.

    Args:
        robot: The robot object.
        forearm_index: Index of the forearm link.
        dist_from_forearm: Distance from the forearm to the point of contact (default: 0.12).
        forearm_dim: Dimension of the forearm (default: 0.05).
        theta: Rotation angle around the y-axis (default: -pi/2).

    Returns:
        tuple: (poc_pos, poc_quat), where poc_pos is the position of the PoC and
               poc_quat is the quaternion representing its orientation.
    """
    # Get the states of the forearm and wrist links
    forearm_state = pb.getLinkState(robot, forearm_index)  # Link=RightForeArm
    forearm_pos, forearm_quat = forearm_state[0], forearm_state[1]

    # Calculate the point of contact pose (above forearm: in forearm y-axis direction)
    forearm_poc_trans = np.array([0, forearm_dim, dist_from_forearm])
    # wrist_poc_quat = [0, 0, 0, 1] # Identity quaternion
    forearm_poc_quat = pb.getQuaternionFromEuler([0, -pi/2, pi/2])
    poc_pos, poc_quat = pb.multiplyTransforms(forearm_pos, forearm_quat, forearm_poc_trans, forearm_poc_quat)

    return np.concatenate([poc_pos, poc_quat])   #list(poc_pos) + list(poc_quat)

class HumanSim:
    def __init__(self):
        rospy.init_node('human_sim', anonymous=True)
        self.rate = rospy.Rate(100)  # 100 Hz
        self.task_done_counter = 0
        self.HUMAN_TAU_LIMIT = 25.0  # Nm clamp


        # Params
        self.gui = rospy.get_param("~pybullet_gui", False)
        self.dt = rospy.get_param("~time_step", 1e-3)
        self.human_urdf = rospy.get_param("~human_urdf", human_name)

        # PyBullet init
        pb.connect(pb.GUI if self.gui else pb.DIRECT)
        pb.resetSimulation()
        pb.setGravity(0, 0, -9.81)
        pb.setTimeStep(self.dt)

        # Publishers
        self.human_target_pose_pub = rospy.Publisher('/human/desired_pose', Pose, queue_size=10)
        self.human_desired_joint_state_pub = rospy.Publisher('/human/desired_joint_states', JointState, queue_size=10)
        self.human_joint_state_pub = rospy.Publisher('/human/joint_states', JointState, queue_size=10)

        # TF broadcaster / listener
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        # Human setup
        self.human_base_pose = [0, 0, 0.5, 0, 0, 0, 1]
        try:
            self.tf_listener.waitForTransform('world', 'Pelvis', rospy.Time(), rospy.Duration(2.0))
            human_base_pos, human_base_quat = self.tf_listener.lookupTransform('world', 'Pelvis', rospy.Time(0))
            self.human_base_pose = list(human_base_pos) + list(human_base_quat)
            rospy.loginfo("Got TF world->Pelvis: %s, %s", str(human_base_pos), str(human_base_quat))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("No TF world->Pelvis; using default human_base_pose: %s", str(e))
            # keep default instead of returning

        # Cup pose from config
        # Cup pose from config (compose relative cup pose with human base)
        base_pos, base_quat = tuple(self.human_base_pose[:3]), tuple(self.human_base_pose[3:])
        cup_local_quat = pb.getQuaternionFromEuler(cup_euler)
        cup_world_pos, cup_world_quat = pb.multiplyTransforms(base_pos, base_quat,
                                                              tuple(cup_pos), cup_local_quat)
        self.cup_pose = list(cup_world_pos) + list(cup_world_quat)
        
        # Cup pose subscription (optional mocap or tf)
        self._cup_lock = threading.Lock()
        self._natnet_ready = False
        if MOCAP:
            rospy.Subscriber('/natnet/cup/pose', Pose, self.cup_pose_callback, queue_size=1)
        else:
            # Try TF if available; otherwise fall back to configured pose
            try:
                pos, quat = self.tf_listener.lookupTransform('world', 'cup', rospy.Time(0))
                self.cup_pose = list(pos) + list(quat)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("No TF world->cup; using configured cup_pose.")

        # Desired handle pose (world frame)
        self.human_desired_pose = list(np.array(self.cup_pose[:3]) + np.array(cup_handle_offset)) + \
                                  list(pb.getQuaternionFromEuler(cup_handle_euler))

        self.human = Human(robot_name=self.human_urdf, controllable_joints=human_controllable_joints)
        self.human.load(base_pose=self.human_base_pose)

        home_config = [0.0] * pb.getNumJoints(self.human.robot_id)
        for idx, val in human_home_overrides.items():
            if idx < len(home_config):
                home_config[idx] = val
        self.human.set_home_configuration(home_config)

        human_q_init = [0.0] * len(self.human.controllable_joints)
        for rel_idx, val in human_init_states.items():
            if rel_idx < len(human_q_init):
                human_q_init[rel_idx] = val
        self.human.set_joint_positions(human_q_init)

        
        # Impairment
        if human_impairment_method == 'mid_pose':
            # Shift target a bit to simulate impairment
            hpos = np.array(self.human_desired_pose[:3])
            self.human_desired_pose[:3] = (hpos + np.array([-0.1, 0.0, -0.05])).tolist()
        elif human_impairment_method == 'joint_limits':
            rospy.logwarn("Joint limit impairment via changeDynamics is not supported; implement in controller instead.")
        elif human_impairment_method != 'ROM':
            rospy.logwarn("Unknown impairment method; implementing as no-op.")
        else:
            raise ValueError(f"Unknown impairment method: {human_impairment_method}")

        # Dynamics/friction
        for link_idx in range(pb.getNumJoints(self.human.robot_id)+1):
            pb.changeDynamics(self.human.robot_id, link_idx,
                              linearDamping=0.0, angularDamping=0.0, jointDamping=0.0,
                              restitution=0.0, lateralFriction=0.6, spinningFriction=0.1, rollingFriction=0.1)
            pb.changeDynamics(self.human.robot_id, link_idx, maxJointVelocity=200)

        # Disable default motors (enable torque control)
        pb.setJointMotorControlArray(self.human.robot_id, self.human.controllable_joints,
                                     controlMode=pb.VELOCITY_CONTROL, forces=[0.0]*len(self.human.controllable_joints))

    def cup_pose_callback(self, msg):
        cup_pose_mocap = np.array([msg.position.x, msg.position.y, msg.position.z,
                             msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.human_desired_pose = cup_pose_mocap[:3] + np.array(cup_handle_offset) + list(pb.getQuaternionFromEuler(cup_handle_euler))
        self._natnet_ready = True
            # Internal state

    def main(self):
        if MOCAP:
            rospy.loginfo("Waiting for /natnet/cup/pose...")
            while not rospy.is_shutdown():
                with self._cup_lock:
                    ready = self._natnet_ready
                if ready:
                    rospy.loginfo("/natnet/cup/pose received.")
                    break
                self.rate.sleep()

        while not rospy.is_shutdown():
            human_ee_state = pb.getLinkState(self.human.robot_id, self.human.ee_index)  # (pos, quat) -> indices 0,1
            human_ee_pos, human_ee_quat = list(human_ee_state[0]), list(human_ee_state[1])
            human_ee_pose = human_ee_pos + human_ee_quat
            human_poc_pose = get_poc_pose(self.human.robot_id)  # Expect [x,y,z,qx,qy,qz,qw]
            human_target_err_pos = np.array(self.human_desired_pose[:3]) - np.array(human_ee_pos)
            human_target_err_quat_vec = quat_diff(self.human_desired_pose[3:], human_ee_quat)
            human_target_err = list(human_target_err_pos) + list(human_target_err_quat_vec)

            # Task done check
            if np.linalg.norm(human_target_err_pos) < 1e-3 and np.linalg.norm(human_target_err_quat_vec) < 1e-2:
                self.task_done_counter += 1
                if self.task_done_counter >= TASK_THRESHOLD:
                    rospy.loginfo("Task completed. Pausing sim.")
                    pb.setTimeStep(0)
            else:
                self.task_done_counter = max(0, self.task_done_counter - 1)

            human_k_lin, human_k_ang = human_k_lin_init, human_k_ang_init
            human_role = human_role_observe
            F_ext = None
            F_ext_point = human_poc_pose[:3]  # Apply external force at the PoC

            tau_human = self.human.passive_DS(self.human_desired_pose,
                                              null_gain=human_null_gain, null_pos=human_null_pos,
                                              damping_eigval=human_damping_eigval,
                                              k_lin=human_k_lin, k_ang=human_k_ang, role=human_role,
                                              F_ext=F_ext, F_ext_point=F_ext_point)
            tau_human = np.clip(tau_human, -self.HUMAN_TAU_LIMIT, self.HUMAN_TAU_LIMIT)

            pb.setJointMotorControlArray(self.human.robot_id, self.human.controllable_joints,
                                         pb.TORQUE_CONTROL, forces=tau_human.tolist(),
                                         positionGains=[0.0]*len(self.human.controllable_joints),
                                         velocityGains=[0.0]*len(self.human.controllable_joints))

           # Publish joint states (all joints for RViz)
            nj = pb.getNumJoints(self.human.robot_id)
            q_states = pb.getJointStates(self.human.robot_id, list(range(nj)))
            joint_infos = [pb.getJointInfo(self.human.robot_id, i) for i in range(nj)]
            # q_states = [j for j, i in zip(q_states, joint_infos) if i[3] > -1]
            joint_names = [info[1].decode('utf-8') for info in joint_infos]
            q = [s[0] for s in q_states]
            dq = [s[1] for s in q_states]
            js_msg = JointState()
            js_msg.header.stamp = rospy.Time.now()
            js_msg.name = joint_names
            js_msg.header.frame_id = "human"
            js_msg.position = list(q)
            js_msg.velocity = list(dq)
            self.human_joint_state_pub.publish(js_msg)
            # rospy.logwarn_throttle(5, f"Human joint pos: {len(q)},  Human joint  name: {len(joint_names)}")

            # Publish desired pose
            des_pose_msg = create_pose_msg(self.human_desired_pose[:3], self.human_desired_pose[3:])
            self.human_target_pose_pub.publish(des_pose_msg)

            # TF for target
            self.tf_broadcaster.sendTransform(self.cup_pose[:3],
                                              self.cup_pose[3:],
                                              rospy.Time.now(),
                                              "cup_handle",
                                              "world")
             # Show frames in pb
            draw_pose_frame(human_ee_pose, length=0.1, lineWidth=2, life_time=0.2)
            draw_pose_frame(self.cup_pose, length=0.1, lineWidth=3, life_time=0.2)

            # Step sim
            pb.stepSimulation()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = HumanSim()
        node.main()
    except rospy.ROSInterruptException:
        pass