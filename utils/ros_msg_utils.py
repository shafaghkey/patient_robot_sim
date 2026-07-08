"""ros_msg_utils

ROS message factory functions and RViz marker helpers.

Provides:
  reorder_quat_xyzw_to_wxyz / reorder_quat_wxyz_to_xyzw  — quaternion convention swap
  pose_msg_to_position_quaternion / pose_msg_to_xyz_quat  — Pose message decoders
  pos_quat_to_pose_msg / create_pose_msg                  — Pose message constructors
  create_twist_stamped_msg                                 — TwistStamped factory
  create_joint_state_msg                                   — JointState factory
  create_wrench_msg                                        — WrenchStamped factory
  create_marker_arrow_msg                                  — RViz arrow Marker factory
  publish_workspace_sphere_marker                          — RViz sphere Marker publisher
"""
import os
import sys
from typing import Any, Optional, Sequence

import numpy as np
import threading
import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from utils.transforms import quat_normalized


# ─── Quaternion convention helpers ───────────────────────────────────────────

def reorder_quat_xyzw_to_wxyz(quat_xyzw: Sequence[float]) -> np.ndarray:
    """Reorder quaternion from (x,y,z,w) to (w,x,y,z)."""
    q = np.asarray(quat_xyzw, dtype=float)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def reorder_quat_wxyz_to_xyzw(quat_wxyz: Sequence[float]) -> np.ndarray:
    """Reorder quaternion from (w,x,y,z) to (x,y,z,w)."""
    q = np.asarray(quat_wxyz, dtype=float)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


# ─── Pose message decoders ────────────────────────────────────────────────────

def pose_msg_to_xyz_quat(p: Pose):
    """Decode Pose → ((x,y,z), (qx,qy,qz,qw)) as plain tuples (no normalization)."""
    return ((p.position.x, p.position.y, p.position.z),
            (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w))


def pose_msg_to_position_quaternion(pose_msg: Any, reorder: bool = False):
    """Decode geometry_msgs/Pose → (pos (3,), quat (4,)) numpy arrays.

    Args:
        reorder: If True, return quaternion in (w,x,y,z) instead of (x,y,z,w).
    """
    position    = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z],
                           dtype=np.float64)
    orientation = quat_normalized([pose_msg.orientation.x, pose_msg.orientation.y,
                                   pose_msg.orientation.z, pose_msg.orientation.w])
    if reorder:
        orientation = reorder_quat_xyzw_to_wxyz(orientation)
    return position, orientation


# ─── Pose message constructors ────────────────────────────────────────────────

def pos_quat_to_pose_msg(pos, quat) -> Pose:
    """(pos (3,), quat (x,y,z,w)) → geometry_msgs/Pose."""
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(pos[0]), float(pos[1]), float(pos[2])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (
        float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    return p


def create_pose_msg(position, orientation, reorder: bool = False) -> Pose:
    """Build geometry_msgs/Pose from position and quaternion arrays.

    Args:
        reorder: If True, treat orientation as (w,x,y,z) and reorder to (x,y,z,w).
    """
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (
        float(position[0]), float(position[1]), float(position[2]))
    if reorder:
        pose.orientation.x = float(orientation[1])
        pose.orientation.y = float(orientation[2])
        pose.orientation.z = float(orientation[3])
        pose.orientation.w = float(orientation[0])
    else:
        pose.orientation.x = float(orientation[0])
        pose.orientation.y = float(orientation[1])
        pose.orientation.z = float(orientation[2])
        pose.orientation.w = float(orientation[3])
    return pose


# ─── Other message factories ─────────────────────────────────────────────────

def create_twist_stamped_msg(linear, angular, stamp=None, frame_id: str = "world"):
    """Build geometry_msgs/TwistStamped."""
    from geometry_msgs.msg import TwistStamped
    msg = TwistStamped()
    msg.header.stamp    = stamp if stamp is not None else rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.twist.linear.x,  msg.twist.linear.y,  msg.twist.linear.z  = (
        float(linear[0]),  float(linear[1]),  float(linear[2]))
    msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = (
        float(angular[0]), float(angular[1]), float(angular[2]))
    return msg


def create_joint_state_msg(positions=None, velocities=None, efforts=None,
                           names=None, stamp=None, frame_id: str = "") -> JointState:
    """Build sensor_msgs/JointState."""
    js = JointState()
    js.position  = list(positions)  if positions  is not None else []
    js.velocity  = list(velocities) if velocities is not None else []
    js.effort    = list(efforts)    if efforts    is not None else []
    js.header.stamp    = stamp if stamp is not None else rospy.Time.now()
    js.header.frame_id = frame_id
    js.name      = list(names) if names is not None else []
    return js


def create_wrench_msg(wrench, stamp=None, frame_id: str = ""):
    """Build geometry_msgs/WrenchStamped from a (6,) array [fx,fy,fz,tx,ty,tz]."""
    from geometry_msgs.msg import WrenchStamped
    ws = WrenchStamped()
    ws.header.stamp    = stamp if stamp is not None else rospy.Time.now()
    ws.header.frame_id = frame_id
    ws.wrench.force.x,  ws.wrench.force.y,  ws.wrench.force.z  = (
        float(wrench[0]), float(wrench[1]), float(wrench[2]))
    ws.wrench.torque.x, ws.wrench.torque.y, ws.wrench.torque.z = (
        float(wrench[3]), float(wrench[4]), float(wrench[5]))
    return ws


# ─── RViz marker helpers ─────────────────────────────────────────────────────

def create_marker_arrow_msg(origin_xyz, force_xyz, scale: float = 0.25,
                            color=None, stamp=None):
    """Build visualization_msgs/Marker arrow from origin in the direction of force_xyz."""
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import Point, Vector3
    if color is None:
        color = [1.0, 0.2, 0.2, 1.0]
    p1 = [origin_xyz[i] + force_xyz[i] * scale for i in range(3)]
    m = Marker()
    m.header.frame_id    = "world"
    m.header.stamp       = stamp if stamp is not None else rospy.Time.now()
    m.id                 = 0
    m.type               = Marker.ARROW
    m.action             = Marker.ADD
    m.pose.orientation.w = 1.0
    m.points             = [Point(*origin_xyz), Point(*p1)]
    m.scale              = Vector3(0.01, 0.025, 0.0)
    m.color.r, m.color.g, m.color.b, m.color.a = color
    return m


def publish_workspace_sphere_marker(pub, center, threshold: float,
                                    frame_id: str = "world",
                                    color=(0.1, 0.8, 0.2, 0.2)) -> None:
    """Publish a sphere Marker representing the reachability workspace boundary."""
    from visualization_msgs.msg import Marker
    m = Marker()
    m.header.frame_id           = frame_id
    m.header.stamp              = rospy.Time.now()
    m.ns, m.id                  = "workspace", 0
    m.type, m.action            = Marker.SPHERE, Marker.ADD
    m.pose.position.x, m.pose.position.y, m.pose.position.z = (
        float(center[0]), float(center[1]), float(center[2]))
    m.pose.orientation.w        = 1.0
    d = 2.0 * float(threshold)
    m.scale.x = m.scale.y = m.scale.z = d
    m.color.r, m.color.g, m.color.b, m.color.a = color
    m.lifetime                  = rospy.Duration(0)
    try:
        pub.publish(m)
    except Exception:
        pass


def init_shared_buffer():
    """Legacy helper: returns (lock, have_data, pos (3,), quat (4,))."""
    return (threading.Lock(), False,
            np.zeros(3, dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))