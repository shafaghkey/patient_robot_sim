#!/usr/bin/env python3
"""human_publisher

Converts NatNet/OptiTrack skeleton poses into ROS JointState messages for the
human_gazebo_shkey.urdf model. Used with human_robot_sim.launch.

Subscriptions:
  /natnet_ros/Skeleton_<id>/pose  (PoseArray) — raw skeleton bone poses from NatNet

Publishes:
  /human/joint_states  (JointState) — joint angles derived from 51-bone skeleton
"""

import numpy as np
import rospy
import sensor_msgs.msg
from geometry_msgs.msg import PoseArray
from math import pi
from tf.transformations import euler_from_quaternion
import threading

fixed_frame_name = "world"  # This should match the root link of your human URDF model (e.g., "Pelvis" or "base_link")


class SKL:
    def __init__(self, dt) -> None:
        self.dt = dt

        skl_bones_num = 51
        self.bone_pose = np.zeros((skl_bones_num, 7))
        self.bone_pos = np.zeros((skl_bones_num, 3))
        self.bone_rot = np.zeros((skl_bones_num, 3))

        # Protect joints_list from concurrent access (callback thread vs publish loop)
        self._joints_lock = threading.Lock()

        # Will be overwritten in callback when mocap is present.
        self.joints_list = [0, 0, 0, 0, 0,                #Ab, Chest
                            0, 0, 0, 0, 0,             #Neck, Head
                            0, 0, -1.3, 0,            #Right shoulder/upperarm
                            0, 0, 0, 0,            #Right forearm/hand
                            0, 0, -1.3, 0,            # Left shoulder/upperarm
                            0, 0, 0, 0]            # Left forearm/hand
        
        # MOCAP = rospy.get_param('/mocap', True)
        # if MOCAP:
        #     # NatNet skeleton pose stream
        self.natnet_skl_sub = rospy.Subscriber('/natnet_ros/fullbody/pose', PoseArray, self.skl_callback, tcp_nodelay=True, queue_size=1)
        # Joint states output for robot_state_publisher
        self.skl_joint_state_pub = rospy.Publisher('/human/joint_states', sensor_msgs.msg.JointState, queue_size=10)

    def skl_callback(self,msg):
        '''
        SKL bone index mapping in /natnet_ros/fullbody/pose:
        0 hip
        1 abdomen
        2 chest
        22 neck
        23 head
        3 r_shoulder
        4 r_u_arm
        5 r_f_arm
        6 r_hand
        24 l_shoulder
        25 l_u_arm
        26 l_f_arm
        27 l_hand
        

        '''
        for idx, bone in enumerate(msg.poses):
            self.bone_pose[idx, 0] = bone.position.x
            self.bone_pose[idx, 1] = bone.position.y
            self.bone_pose[idx, 2] = bone.position.z
            self.bone_pose[idx, 3] = bone.orientation.x
            self.bone_pose[idx, 4] = bone.orientation.y
            self.bone_pose[idx, 5] = bone.orientation.z
            self.bone_pose[idx, 6] = bone.orientation.w

            ang1, ang2, ang3 = euler_from_quaternion(self.bone_pose[idx, 3:], axes='rxyz')

            self.bone_pos[idx, 0] = bone.position.x
            self.bone_pos[idx, 1] = bone.position.y
            self.bone_pos[idx, 2] = bone.position.z

            # Preserve your original axis mapping
            self.bone_rot[idx, 0] = -ang2
            self.bone_rot[idx, 1] = ang1
            self.bone_rot[idx, 2] = ang3


        # # Preserve your original joint construction
        # # joints = [0.0]
        # joints = []
        # joints.append(self.bone_rot[0, :2])    # Hip
        # # joints.append(self.bone_rot[1, :2])    # Ab
        # joints.append(self.bone_rot[2, :])    # Chest
        # joints.append(self.bone_rot[22, :])    # Neck
        # joints.append(self.bone_rot[23, :2])    # Head
        # joints.append(self.bone_rot[3, 0])   # Right Arm (note: your original used 24 here)
        # joints.append(self.bone_rot[4, :])   # upper_arm
        # joints.append(self.bone_rot[5, :2])   # forearm
        # joints.append(self.bone_rot[6, :2])   # hand
        # joints.append(self.bone_rot[24, 0])    # Left Arm  (note: your original used 5 here)
        # joints.append(self.bone_rot[25, :])    # upper_arm
        # joints.append(self.bone_rot[26, :2])    # forearm
        # joints.append(self.bone_rot[27, :12])    # hand

        # # Flatten
        # self.joints_list = []
        # for element in joints:
        #     if isinstance(element, np.ndarray):
        #         self.joints_list.extend(element.tolist())
        #     else:
        #         self.joints_list.append(float(element))
        

        new_joints_list = [
            self.bone_rot[1, 0], self.bone_rot[1, 1],                      # Pelvis to LowerTrunk     
            self.bone_rot[2, 0], self.bone_rot[2, 1], self.bone_rot[2, 2], # LowerTrunk to UpperTrunk
            self.bone_rot[22, 0], self.bone_rot[22, 1], self.bone_rot[22, 2], # UpperTrunk to Neck
            self.bone_rot[23, 0], self.bone_rot[23, 1],                                # Neck to Head
            
            self.bone_rot[3, 0],                                         # UpperTrunk to RightShoulder
            self.bone_rot[4, 0], self.bone_rot[4, 1], self.bone_rot[4, 2],                 # Right shoulder chain
            self.bone_rot[5, 1], self.bone_rot[5, 0],            # Right forearm 
            self.bone_rot[6, 0], self.bone_rot[6, 1],                        # Right hand 
            
            self.bone_rot[24, 0],                                         # UpperTrunk to LeftShoulder
            self.bone_rot[25, 0], self.bone_rot[25, 1], self.bone_rot[25, 2],                 # Left shoulder chain
            self.bone_rot[26, 1], self.bone_rot[26, 0],            # Left forearm 
            self.bone_rot[27, 0], self.bone_rot[27, 1],                        # Left hand
        ]

        # Atomic swap under lock so publish loop never observes partial updates.
        with self._joints_lock:
            self.joints_list = new_joints_list

        rospy.logdebug_throttle(1.0, f"Updated joints_list (len={len(new_joints_list)})")


    def publish_joint_states(self):
        js = sensor_msgs.msg.JointState()
        js.header.stamp = rospy.Time.now()
        js.header.frame_id = fixed_frame_name

        # Joint names aligned to human_gazebo_shkey.urdf
        # Key change: jC7RightShoulder_rotx and jC7LeftShoulder_rotx
        
        js.name = [
            'jL5S1_rotx', 'jL5S1_roty',                                    # Pelvis to LowerTrunk
            'jT9T8_rotx', 'jT9T8_roty', 'jT9T8_rotz',                      # LowerTrunk to UpperTrunk
            'jT1C7_rotx', 'jT1C7_roty', 'jT1C7_rotz',                      # UpperTrunk to Neck
            'jC1Head_rotx', 'jC1Head_roty',                                # Neck to Head

            'jC7RightShoulder_rotx', 
            'jRightShoulder_rotz',                                         # UpperTrunk to RightShoulder
            'jRightShoulder_rotx', 'jRightShoulder_roty',                 # Right shoulder chain
            'jRightElbow_rotz', 'jRightElbow_roty',
            'jRightWrist_rotx', 'jRightWrist_rotz',                        # Right elbow/wrist

            'jC7LeftShoulder_rotx', 
            'jLeftShoulder_rotz',                                         # UpperTrunk to LeftShoulder
            'jLeftShoulder_rotx', 'jLeftShoulder_roty',                   # Left shoulder chain
            'jLeftElbow_rotz', 'jLeftElbow_roty',
            'jLeftWrist_rotx', 'jLeftWrist_rotz'                           # Left elbow/wrist
        ]

        # Positions from NatNet mapping
        with self._joints_lock:
            pos = list(self.joints_list)

        # robot_state_publisher will reject JointState if name/position sizes don't match
        # or if positions contain NaNs/Infs.
        if len(pos) != len(js.name):
            rospy.logwarn_throttle(2.0, f"Invalid JointState: positions({len(pos)}) != names({len(js.name)}). Skipping publish.")
            return

        if not np.isfinite(np.array(pos, dtype=float)).all():
            rospy.logwarn_throttle(2.0, "Invalid JointState: non-finite joint position. Skipping publish.")
            return

        js.position = pos

        js.velocity = []
        js.effort = []

        self.skl_joint_state_pub.publish(js)
        # roslog for debugging: length of joints_list and joint names should match
        rospy.logdebug(f"Published joint states: {len(self.joints_list)} joints, names: {len(js.name)}")


def main():
    rospy.init_node('human_joint_publisher')
    freq=10
    r = rospy.Rate(freq)
    skl = SKL(1/freq)

    while not rospy.is_shutdown():
        skl.publish_joint_states()
        r.sleep()
    rospy.spin()
         

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

