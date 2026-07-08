#!/usr/bin/env python3
"""set_initial_joints

One-shot Gazebo startup helper. Waits for the iiwa model to spawn, then calls
/gazebo/set_model_configuration to apply IIWA_INIT_JOINTS from
config/hri_config.py. Exits after the service call completes.
Used in this package's Gazebo launch flow.

ROS params:
  ~model_name        (str, default 'iiwa')        — Gazebo model name
  ~urdf_param_name   (str, default 'robot_description') — param holding the URDF
"""
import rospy
from gazebo_msgs.srv import SetModelConfiguration, SetModelConfigurationRequest, GetModelState
import os, sys
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
from config.hri_config import IIWA_JOINT_NAMES, IIWA_INIT_JOINTS

def wait_for_model(model_name, timeout=30.0):
    rospy.wait_for_service('/gazebo/get_model_state', timeout=timeout)
    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    rate = rospy.Rate(5)
    start = rospy.Time.now()
    while not rospy.is_shutdown():
        try:
            resp = get_state(model_name, '')
            if resp.success:
                return True
        except rospy.ServiceException:
            pass
        if (rospy.Time.now() - start).to_sec() > timeout:
            return False
        rate.sleep()
    return False

def main():
    rospy.init_node('set_initial_joints', anonymous=True)
    model_name = rospy.get_param('~model_name', 'iiwa')
    urdf_param_name = rospy.get_param('~urdf_param_name', 'robot_description')
    joint_names = rospy.get_param('~joint_names', IIWA_JOINT_NAMES)
    joint_positions = rospy.get_param('~joint_positions', IIWA_INIT_JOINTS)
    delay_sec = rospy.get_param('~delay_sec', 1.0)

    if not joint_names or not joint_positions or len(joint_names) != len(joint_positions):
        rospy.logerr('joint_names and joint_positions must be non-empty and of equal length')
        return

    rospy.sleep(delay_sec)

    if not wait_for_model(model_name, timeout=30.0):
        rospy.logerr('Model %s not found in Gazebo', model_name)
        return

    rospy.wait_for_service('/gazebo/set_model_configuration', timeout=30.0)
    set_cfg = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)

    req = SetModelConfigurationRequest()
    req.model_name = model_name
    req.urdf_param_name = urdf_param_name
    req.joint_names = joint_names
    req.joint_positions = joint_positions

    try:
        resp = set_cfg(req)
        if resp.success:
            rospy.loginfo('Initial joints set for %s', model_name)
        else:
            rospy.logerr('Failed to set joints: %s', resp.status_message)
    except Exception as e:
        rospy.logerr('Service call failed: %s', e)

if __name__ == '__main__':
    main()