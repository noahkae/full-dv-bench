"""
Sensor models and the FSDS settings.json loader.

Sensors are configured from an FSDS/AirSim settings.json (the same file the
workbench's hardcoded offsets are meant to match), so the sensor layout is
defined in one place.

LiDAR is exact analytic ray casting against a flat ground plane (z = 0) and
right circular cones - no mesh, no engine. Semantics follow FSDS docs/lidar.md:
  - lasers spread evenly from VerticalFOVLower..VerticalFOVUpper
  - HorizontalFOVStart/End are degrees clockwise from forward (top view)
  - PointsPerScan firings per scan, split evenly across lasers
  - 100 m range, misses are dropped, points are in the sensor frame
  - a whole scan is captured at one instant (no motion distortion)
"""
import json
import math
from dataclasses import dataclass

import numpy as np

from .geometry import rpy_to_matrix

SENSOR_IMU = 2
SENSOR_GPS = 3
SENSOR_LIDAR = 6
SENSOR_GSS = 7
IMAGE_SCENE, IMAGE_DEPTH_PLANAR, IMAGE_DEPTH_PERSPECTIVE = 0, 1, 2


@dataclass
class LidarConfig:
    name: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    n_lasers: int = 16
    points_per_scan: int = 2048
    vfov_upper: float = 10.0
    vfov_lower: float = -10.0
    hfov_start: float = -90.0
    hfov_end: float = 90.0
    rotations_per_second: float = 10.0
    max_range: float = 100.0
    range_noise_std: float = 0.0


@dataclass
class ImuConfig:
    """AirSim's ImuSimple model with its MPU-6000 defaults (what FSDS uses):
    white noise (angle/velocity random walk) plus a bias that slowly random
    walks, with stability `*_bias_stability` and correlation time `*_tau`."""
    name: str = 'Imu'
    rate_hz: float = 250.0
    noise: bool = True
    gyro_arw: float = 0.30 / math.sqrt(3600.0) * math.pi / 180.0       # rad/sqrt(s)
    gyro_tau: float = 500.0                                            # s
    gyro_bias_stability: float = 4.6 / 3600.0 * math.pi / 180.0        # rad/s
    accel_vrw: float = 0.24 * 9.80665 / 1.0e3                         # m/s/sqrt(s)
    accel_tau: float = 800.0
    accel_bias_stability: float = 36.0e-6 * 9.80665                    # m/s^2


@dataclass
class GpsConfig:
    name: str = 'Gps'
    rate_hz: float = 10.0
    eph: float = 0.04          # horizontal std (m), FSDS default
    epv: float = 0.04
    noise_min_speed: float = 0.1  # FSDS only adds noise above this speed


@dataclass
class GssConfig:
    name: str = 'GSS'
    rate_hz: float = 100.0


@dataclass
class CameraConfig:
    name: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    width: int = 785
    height: int = 785
    fov_deg: float = 90.0     # horizontal, like AirSim
    rate_hz: float = 30.0     # FSDS sets this in the bridge launch file, not settings.json
    image_type: int = 0       # AirSim ImageType: 0 Scene, 1 DepthPlanar, 2 DepthPerspective
    depth_noise: bool = True  # FSDS bridge: 7x7 blur + coarser depth with range

    @property
    def is_depth(self):
        return self.image_type in (IMAGE_DEPTH_PLANAR, IMAGE_DEPTH_PERSPECTIVE)

    def intrinsics(self):
        """(fx, fy, cx, cy) for an ideal pinhole with square pixels."""
        fx = self.width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
        return fx, fx, self.width / 2.0, self.height / 2.0


@dataclass
class SensorSuite:
    vehicle_name: str
    lidars: list
    imu: object = None
    gps: object = None
    gss: object = None
    cameras: list = None


def load_settings(path):
    """Parse the first vehicle in an FSDS settings.json into a SensorSuite."""
    with open(path) as f:
        settings = json.load(f)
    vehicles = settings.get('Vehicles') or {}
    if not vehicles:
        raise ValueError('{} has no "Vehicles" section'.format(path))
    vehicle_name, vehicle = next(iter(vehicles.items()))
    lidars, imu, gps, gss = [], None, None, None
    for name, s in (vehicle.get('Sensors') or {}).items():
        if not s.get('Enabled', True):
            continue
        kind = s.get('SensorType')
        if kind == SENSOR_LIDAR:
            lidars.append(LidarConfig(
                name=name,
                x=s.get('X', 0.0), y=s.get('Y', 0.0), z=s.get('Z', 0.0),
                roll=s.get('Roll', 0.0), pitch=s.get('Pitch', 0.0), yaw=s.get('Yaw', 0.0),
                n_lasers=int(s.get('NumberOfLasers', 16)),
                points_per_scan=int(s.get('PointsPerScan', 2048)),
                vfov_upper=s.get('VerticalFOVUpper', 10.0),
                vfov_lower=s.get('VerticalFOVLower', -10.0),
                hfov_start=s.get('HorizontalFOVStart', -90.0),
                hfov_end=s.get('HorizontalFOVEnd', 90.0),
                rotations_per_second=s.get('RotationsPerSecond', 10.0),
            ))
        elif kind == SENSOR_IMU:
            imu = ImuConfig(name=name)
        elif kind == SENSOR_GPS:
            gps = GpsConfig(name=name)
        elif kind == SENSOR_GSS:
            gss = GssConfig(name=name)
    cameras = []
    for name, c in (vehicle.get('Cameras') or {}).items():
        caps = c.get('CaptureSettings', []) or [{}]
        cap = caps[0]
        image_type = int(cap.get('ImageType', 0))
        if image_type not in (IMAGE_SCENE, IMAGE_DEPTH_PLANAR, IMAGE_DEPTH_PERSPECTIVE):
            raise ValueError('camera {}: ImageType {} not supported (0 Scene, 1 DepthPlanar, '
                             '2 DepthPerspective)'.format(name, image_type))
        cameras.append(CameraConfig(
            image_type=image_type,
            name=name,
            x=c.get('X', 0.0), y=c.get('Y', 0.0), z=c.get('Z', 0.0),
            roll=c.get('Roll', 0.0), pitch=c.get('Pitch', 0.0), yaw=c.get('Yaw', 0.0),
            width=int(cap.get('Width', 785)), height=int(cap.get('Height', 785)),
            fov_deg=cap.get('FOV_Degrees', 90.0),
        ))
    return SensorSuite(vehicle_name, lidars, imu, gps, gss, cameras)


class Lidar(object):
    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        n_az = max(1, cfg.points_per_scan // max(1, cfg.n_lasers))
        # FOV angles are clockwise from forward; ENU azimuth is counter-clockwise.
        az = -np.radians(np.linspace(cfg.hfov_start, cfg.hfov_end, n_az, endpoint=n_az == 1))
        if cfg.n_lasers > 1:
            el = np.radians(np.linspace(cfg.vfov_upper, cfg.vfov_lower, cfg.n_lasers))
        else:
            el = np.radians([cfg.vfov_upper])
        el_g, az_g = np.meshgrid(el, az, indexing='ij')
        dirs = np.stack([np.cos(el_g) * np.cos(az_g),
                         np.cos(el_g) * np.sin(az_g),
                         np.sin(el_g)], axis=-1).reshape(-1, 3)
        self.dirs_sensor = dirs                    # (M, 3); ray index = laser * n_az + column
        mount = np.array(rpy_to_matrix(math.radians(cfg.roll), math.radians(cfg.pitch),
                                       math.radians(cfg.yaw)))
        self.dirs_car = dirs @ mount.T
        self.offset = np.array([cfg.x, cfg.y, cfg.z], dtype=np.float64)

        # Azimuth-column culling: rays lie on a regular azimuth grid, so a cone
        # can only be hit by the few columns its angular width covers. Only
        # valid when the mount is level (yaw-only), otherwise brute force.
        self.n_az = n_az
        self.n_lasers = len(el)
        self.az_step = math.radians(cfg.hfov_end - cfg.hfov_start) / n_az
        self.full_circle = (cfg.hfov_end - cfg.hfov_start) >= 360.0
        self.level_mount = abs(cfg.roll) < 1e-9 and abs(cfg.pitch) < 1e-9
        self.max_tan_el = float(np.max(np.abs(np.tan(el))))

    def scan(self, car_x, car_y, car_yaw, cone_xy, cone_r, cone_h, car_roll=0.0, car_pitch=0.0,
             boxes=None, cylinders=None):
        """Return an (N, 3) float32 array of hits in the sensor frame.

        cone_*: upright cones (track cones plus any spare-cone stacks).
        boxes: (K, 6) x, y, yaw, half_x, half_y, height - barriers, equipment.
        cylinders: (K, 4) x, y, radius, height - tyre stacks, poles.

        The sensor moves with the sprung body, so body roll/pitch tilt every
        ray: braking dips the ground ring towards the car, cornering skews it.
        """
        rot = np.array(rpy_to_matrix(car_roll, car_pitch, car_yaw))
        origin = np.array([car_x, car_y, 0.0]) + rot @ self.offset
        self._tilt = abs(car_roll) + abs(car_pitch)
        d = self.dirs_car @ rot.T                                   # world-frame dirs (M, 3)
        rng_max = self.cfg.max_range
        t_best = np.full(len(d), np.inf)

        # ground plane z = 0
        down = d[:, 2] < -1e-9
        t_best[down] = -origin[2] / d[down, 2]

        if len(cone_xy):
            self._cast(t_best, origin, d, car_yaw, np.asarray(cone_xy), np.asarray(cone_r),
                       lambda rays, i, xy: self._hit_pairs(origin, d[rays], xy[i], cone_r[i], cone_h[i]))
        if boxes is not None and len(boxes):
            self._cast(t_best, origin, d, car_yaw, boxes[:, :2], np.hypot(boxes[:, 3], boxes[:, 4]),
                       lambda rays, i, xy: self._hit_boxes(origin, d[rays], boxes[i]))
        if cylinders is not None and len(cylinders):
            self._cast(t_best, origin, d, car_yaw, cylinders[:, :2], cylinders[:, 2],
                       lambda rays, i, xy: self._hit_cylinders(origin, d[rays], cylinders[i]))

        hit = t_best <= rng_max
        t = t_best[hit]
        if self.cfg.range_noise_std > 0.0:
            t = t + self.rng.normal(0.0, self.cfg.range_noise_std, size=t.shape)
        return (self.dirs_sensor[hit] * t[:, None]).astype(np.float32)

    def _cast(self, t_best, origin, d, car_yaw, xy, bound_r, hit):
        """Cull objects to candidate (ray, object) pairs, intersect, keep nearest hits."""
        rel = xy - origin[:2]
        dist = np.hypot(rel[:, 0], rel[:, 1])
        near = np.nonzero(dist < self.cfg.max_range + bound_r + 1.0)[0]
        if not len(near):
            return
        if self.level_mount and self.az_step > 0:
            rays, local = self._candidates(rel[near], dist[near], bound_r[near], car_yaw)
        else:
            rays = np.repeat(np.arange(len(d)), len(near))
            local = np.tile(np.arange(len(near)), len(d))
        if len(rays):
            t = hit(rays, near[local], xy)
            np.minimum.at(t_best, rays, t)

    @staticmethod
    def _hit_boxes(o, d, b):
        """Ray vs yaw-rotated box standing on the ground (slab method)."""
        c, s = np.cos(b[:, 2]), np.sin(b[:, 2])
        rx, ry = o[0] - b[:, 0], o[1] - b[:, 1]
        ox, oy = c * rx + s * ry, -s * rx + c * ry               # origin in the box frame
        dx, dy = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        dz = d[:, 2]
        t_lo = np.full(len(b), -np.inf)
        t_hi = np.full(len(b), np.inf)
        for org, dirn, lo, hi in ((ox, dx, -b[:, 3], b[:, 3]), (oy, dy, -b[:, 4], b[:, 4]),
                                  (np.full(len(b), o[2]), dz, np.zeros(len(b)), b[:, 5])):
            par = np.abs(dirn) < 1e-12
            inv = 1.0 / np.where(par, 1.0, dirn)
            t1, t2 = (lo - org) * inv, (hi - org) * inv
            near_t, far_t = np.minimum(t1, t2), np.maximum(t1, t2)
            outside = par & ((org < lo) | (org > hi))
            near_t = np.where(par, np.where(outside, np.inf, -np.inf), near_t)
            far_t = np.where(par, np.where(outside, -np.inf, np.inf), far_t)
            t_lo, t_hi = np.maximum(t_lo, near_t), np.minimum(t_hi, far_t)
        ok = (t_hi >= t_lo) & (t_lo > 1e-6)
        return np.where(ok, t_lo, np.inf)

    @staticmethod
    def _hit_cylinders(o, d, cyl):
        """Ray vs vertical cylinder standing on the ground (side + top cap)."""
        ox, oy = o[0] - cyl[:, 0], o[1] - cyl[:, 1]
        r, h = cyl[:, 2], cyl[:, 3]
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        cc = ox * ox + oy * oy - r * r
        disc = b * b - 4.0 * a * cc
        ok = (disc >= 0.0) & (a > 1e-12)
        t_side = (-b - np.sqrt(np.where(ok, disc, 0.0))) / (2.0 * np.where(ok, a, 1.0))
        z = o[2] + t_side * dz
        t_side = np.where(ok & (t_side > 1e-6) & (z >= 0.0) & (z <= h), t_side, np.inf)
        down = dz < -1e-9
        t_top = (h - o[2]) / np.where(down, dz, -1.0)
        px, py = ox + t_top * dx, oy + t_top * dy
        t_top = np.where(down & (t_top > 1e-6) & (px * px + py * py <= r * r), t_top, np.inf)
        return np.minimum(t_side, t_top)

    def _candidates(self, rel, dist, radius, car_yaw):
        """(ray index, cone index) pairs whose azimuth column can touch the cone."""
        # cone bearing in the sensor frame, as a clockwise FOV angle like the config
        sensor_yaw = car_yaw + math.radians(self.cfg.yaw)
        theta = -(np.arctan2(rel[:, 1], rel[:, 0]) - sensor_yaw)
        start = math.radians(self.cfg.hfov_start)
        theta = np.mod(theta - start, 2.0 * math.pi)              # 0 .. 2pi from FOV start
        # body tilt shifts a ray's world azimuth by up to ~tilt * tan(elevation)
        margin = self.az_step + 1.5 * self._tilt * self.max_tan_el
        half = np.arcsin(np.clip(radius / np.maximum(dist, 1e-6), 0.0, 1.0)) + margin
        k0 = np.floor((theta - half) / self.az_step).astype(np.int64)
        k1 = np.ceil((theta + half) / self.az_step).astype(np.int64)
        close = dist < radius + 0.5                                # pathological: test all columns
        k0[close], k1[close] = 0, self.n_az - 1
        width = k1 - k0 + 1
        cone_idx = np.repeat(np.arange(len(rel)), width)
        cols = np.repeat(k0, width) + (np.arange(width.sum()) - np.repeat(np.cumsum(width) - width, width))
        if self.full_circle:
            cols = np.mod(cols, self.n_az)
        else:
            keep = (cols >= 0) & (cols < self.n_az)
            cols, cone_idx = cols[keep], cone_idx[keep]
        # every laser in each column
        lasers = np.arange(self.n_lasers) * self.n_az
        rays = (lasers[None, :] + cols[:, None]).reshape(-1)
        return rays, np.repeat(cone_idx, self.n_lasers)

    @staticmethod
    def _hit_pairs(o, d, cxy, cr, ch):
        """Nearest positive hit distance for each (ray, upright cone) pair; inf on a miss.

        Cone surface: |p_xy - c| = (r/h) * (h - p_z), 0 <= p_z <= h, with p = o + t d.
        """
        k2 = (cr / ch) ** 2
        ox = o[0] - cxy[:, 0]
        oy = o[1] - cxy[:, 1]
        h = ch - o[2]                                             # apex height above origin
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        a = dx * dx + dy * dy - k2 * dz * dz
        b = 2.0 * (ox * dx + oy * dy + k2 * h * dz)
        cc = ox * ox + oy * oy - k2 * h * h
        disc = b * b - 4.0 * a * cc
        ok = (disc >= 0.0) & (np.abs(a) > 1e-12)
        sq = np.sqrt(np.where(ok, disc, 0.0))
        a_safe = np.where(ok, a, 1.0)
        best = np.full(a.shape, np.inf)
        for sign in (-1.0, 1.0):
            t = (-b + sign * sq) / (2.0 * a_safe)
            z = o[2] + t * dz
            valid = ok & (t > 1e-6) & (z >= 0.0) & (z <= ch)
            best = np.where(valid & (t < best), t, best)
        return best
