"""
Frame helpers. Everything in fsds_lite is ENU, matching FSDS:
x forward (at spawn), y left, z up, yaw counter-clockwise from +x.
Global positions are relative to the car's spawn point.
"""
import math

EARTH_RADIUS = 6378137.0

# FSDS / AirSim default home point. Only used to turn ENU into lat/lon.
DEFAULT_HOME = (47.641468, -122.140165, 122.0)


def yaw_to_quat(yaw):
    """(x, y, z, w) quaternion for a pure rotation about +z."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quat_to_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rpy_to_matrix(roll, pitch, yaw):
    """Rotation matrix (row-major nested lists) for intrinsic Z-Y-X (yaw, pitch, roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def rpy_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def enu_to_geodetic(x, y, z, home=DEFAULT_HOME):
    """Flat-earth approximation - plenty for a few-hundred-metre track."""
    lat0, lon0, alt0 = home
    lat = lat0 + math.degrees(y / EARTH_RADIUS)
    lon = lon0 + math.degrees(x / (EARTH_RADIUS * math.cos(math.radians(lat0))))
    return lat, lon, alt0 + z


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi
