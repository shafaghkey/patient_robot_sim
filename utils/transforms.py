"""transforms

Pure SE(3) / quaternion math. No ROS imports — safe to use in training
scripts, offline tools, and tests outside a ROS environment.

"""
import os
import sys
import math
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

# as_matrix() was added in scipy 1.4.0; fall back to as_dcm() for older installs
# (Ubuntu system scipy packaged with ROS Noetic is 1.3.3).
def _as_mat(r: R_scipy) -> np.ndarray:
    try:
        return r.as_matrix()
    except AttributeError:
        return r.as_dcm()  # type: ignore[attr-defined]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def pos_dist(a, b) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)


def lateral_dist_to_axis(point, axis_origin, axis_quat_xyzw) -> float:
    """Perpendicular distance from `point` to the line through `axis_origin` along the z-axis of `axis_quat_xyzw`."""
    z_hat = quat_to_R(axis_quat_xyzw)[:, 2]
    delta = np.asarray(point, dtype=float) - np.asarray(axis_origin, dtype=float)
    return float(np.linalg.norm(delta - np.dot(delta, z_hat) * z_hat))


def quat_normalized(quat: Sequence[float]) -> np.ndarray:
    """Return a normalized copy of the input quaternion."""
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,):
        raise ValueError("Input quaternion must be a 4-element sequence")

    # # Heuristic: some sources may publish (w,x,y,z). Choose the ordering whose
    # # scalar component (w) has larger absolute value — this tends to pick the
    # # convention where the quaternion is nearer the identity when frames align.
    # try:
    #     # candidate assuming input is (x,y,z,w)
    #     q_xyzw = q.copy()
    #     s_xyzw = abs(q_xyzw[3])
    # except Exception:
    #     s_xyzw = -1.0
    # try:
    #     # candidate assuming input is (w,x,y,z) -> convert to (x,y,z,w)
    #     q_wxyz_to_xyzw = reorder_quat_wxyz_to_xyzw(q)
    #     s_wxyz = abs(q_wxyz_to_xyzw[3])
    # except Exception:
    #     s_wxyz = -1.0

    # if s_wxyz > s_xyzw:
    #     q = q_wxyz_to_xyzw

    norm = np.linalg.norm(q)
    if norm > 1e-12:
        return (q / norm)
    else:
        return np.array([0, 0, 0, 1], dtype=np.float64)
    
def reorder_quat_xyzw_to_wxyz(quat_xyzw: Sequence[float]) -> np.ndarray:
    """Reorder quaternion from (x, y, z, w) to (w, x, y, z) format."""
    q = np.asarray(quat_xyzw, dtype=float)
    if q.shape != (4,):
        raise ValueError("Input quaternion must be a 4-element sequence")
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)

def reorder_quat_wxyz_to_xyzw(quat_wxyz: Sequence[float]) -> np.ndarray:
    """Reorder quaternion from (w, x, y, z) to (x, y, z, w) format."""
    q = np.asarray(quat_wxyz, dtype=float)
    if q.shape != (4,):
        raise ValueError("Input quaternion must be a 4-element sequence")
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def quat_to_R(quat_xyzw) -> np.ndarray:
    """Quaternion (x,y,z,w) → 3×3 rotation matrix."""
    x, y, z, w = quat_xyzw
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1.0 - 2.0*(yy+zz),  2.0*(xy-wz),       2.0*(xz+wy)      ],
        [2.0*(xy+wz),         1.0 - 2.0*(xx+zz),  2.0*(yz-wx)      ],
        [2.0*(xz-wy),         2.0*(yz+wx),         1.0 - 2.0*(xx+yy)],
    ], dtype=np.float64)


def quat_diff(q_target, q_current) -> np.ndarray:
    """Small-angle 3-D error vector (axis-angle) between two (x,y,z,w) quaternions."""
    qt = np.array(q_target,  dtype=np.float64)
    qc = np.array(q_current, dtype=np.float64)
    qt /= np.linalg.norm(qt) + 1e-12
    qc /= np.linalg.norm(qc) + 1e-12
    qc_conj = np.array([-qc[0], -qc[1], -qc[2], qc[3]])
    r = np.array([
        qt[3]*qc_conj[0] + qt[0]*qc_conj[3] + qt[1]*qc_conj[2] - qt[2]*qc_conj[1],
        qt[3]*qc_conj[1] - qt[0]*qc_conj[2] + qt[1]*qc_conj[3] + qt[2]*qc_conj[0],
        qt[3]*qc_conj[2] + qt[0]*qc_conj[1] - qt[1]*qc_conj[0] + qt[2]*qc_conj[3],
        qt[3]*qc_conj[3] - qt[0]*qc_conj[0] - qt[1]*qc_conj[1] - qt[2]*qc_conj[2],
    ])
    r /= np.linalg.norm(r) + 1e-12
    if r[3] < 0:   # canonical positive-w form → shortest-path rotation
        r = -r
    angle = 2.0 * np.arccos(np.clip(r[3], -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    return angle * r[:3] / (np.sin(angle / 2.0) + 1e-12)


def quat_diff_rad(q_target_xyzw, q_current_xyzw) -> float:
    """Angular distance in radians ∈ [0, π]; robust to q vs −q sign ambiguity."""
    qt = quat_normalized(q_target_xyzw)
    qc = quat_normalized(q_current_xyzw)
    dot = float(np.clip(abs(np.dot(qt, qc)), -1.0, 1.0))
    return float(2.0 * math.acos(dot))


def multiply_transforms(pos1, quat1, pos2, quat2):
    """Compose T = T1 * T2. Quaternions in (x,y,z,w). Returns (pos_list, quat_list)."""
    r1 = R_scipy.from_quat(np.asarray(quat1, dtype=float))
    new_pos  = np.asarray(pos1, dtype=float) + _as_mat(r1) @ np.asarray(pos2, dtype=float)
    new_quat = (r1 * R_scipy.from_quat(np.asarray(quat2, dtype=float))).as_quat()
    return new_pos.tolist(), new_quat.tolist()


def invert_transform(pos, quat):
    """Invert rigid transform (pos, quat). Quaternion in (x,y,z,w)."""
    r     = R_scipy.from_quat(np.asarray(quat, dtype=float))
    r_inv = r.inv()
    return (-(_as_mat(r_inv) @ np.asarray(pos, dtype=float))).tolist(), r_inv.as_quat().tolist()


def transform_point(pos: np.ndarray, quat: np.ndarray, local_pt: np.ndarray) -> np.ndarray:
    """Apply rigid transform (pos, quat) to a point expressed in the local frame."""
    return np.asarray(pos, dtype=float) + _as_mat(R_scipy.from_quat(quat)) @ np.asarray(local_pt, dtype=float)


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → quaternion (x,y,z,w)."""
    return R_scipy.from_matrix(m).as_quat()

