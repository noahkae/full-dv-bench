"""
ROS message <-> numpy conversions with no dependency on sensor_msgs_py or
cv_bridge (both are awkward on Galactic/WSL). Pure functions on duck-typed
messages, so they are unit-tested without ROS (tests/test_ros_convert.py).
"""
from __future__ import annotations

import math

import numpy as np

_PF_DTYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16, 5: np.int32, 6: np.uint32,
              7: np.float32, 8: np.float64}


def cloud_to_xyz(msg):
    """sensor_msgs/PointCloud2 -> (N, 3) float64 (x, y, z), any point layout."""
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3))
    fields = {f.name: f for f in msg.fields}
    names, formats, offsets = [], [], []
    for k in ('x', 'y', 'z'):
        f = fields[k]
        names.append(k)
        formats.append(np.dtype(_PF_DTYPES[f.datatype]).newbyteorder('>' if msg.is_bigendian else '<'))
        offsets.append(f.offset)
    dt = np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': msg.point_step})
    raw = np.frombuffer(bytes(msg.data), dtype=dt, count=n)
    xyz = np.column_stack([raw['x'], raw['y'], raw['z']]).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def image_to_bgr(msg):
    """sensor_msgs/Image (bgr8 / rgb8 / 32FC1) -> numpy (BGR uint8, or float32 depth)."""
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding in ('bgr8', 'rgb8'):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return img[..., ::-1].copy() if msg.encoding == 'rgb8' else img
    if msg.encoding == '32FC1':
        return buf.view(np.float32).reshape(msg.height, msg.step // 4)[:, :msg.width]
    raise ValueError('unsupported image encoding ' + msg.encoding)


def intrinsics_from_info(info):
    k = info.k if hasattr(info, 'k') else info.K
    return float(k[0]), float(k[4]), float(k[2]), float(k[5])


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class GeoToEnu(object):
    """NavSatFix lat/lon -> local ENU metres, anchored at the first fix."""

    R = 6378137.0

    def __init__(self):
        self.origin = None

    def __call__(self, lat, lon):
        if self.origin is None:
            self.origin = (lat, lon)
        lat0, lon0 = self.origin
        x = math.radians(lon - lon0) * self.R * math.cos(math.radians(lat0))
        y = math.radians(lat - lat0) * self.R
        return x, y


def sensor_to_car(points, mount):
    """Points in a sensor frame -> car frame. mount = (x, y, z, roll, pitch, yaw) radians."""
    x, y, z, roll, pitch, yaw = mount
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                  [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                  [-sp, cp * sr, cp * cr]])
    return points @ R.T + np.array([x, y, z])
