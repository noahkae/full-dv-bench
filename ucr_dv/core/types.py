"""
Data contracts shared by every bench and every algorithm.

Frames and conventions (same as fsds_lite / ROS REP 103):
    world  ENU, metres, relative to the car's spawn (x forward at spawn, y left)
    car    x forward, y left, z up, origin at the car's reference point
    angles radians, counter-clockwise positive; yaw 0 = +x

Cones are plain numpy arrays so algorithms stay fast and easy to write:
    (N, 4) float64 columns [x, y, color, confidence]
Colour ids keep the ucr-dv-timeless numbering (so old student code works):
    YELLOW 0 (right side), BLUE 1 (left side), ORANGE 2, BIG_ORANGE 3, UNKNOWN 4

Steering in a ControlCommand is the ROAD-WHEEL angle in radians, positive =
LEFT (ENU). Each backend converts to its own convention (fsds_lite: -1..1,
positive right; timeless: steering-wheel degrees; Assetto Corsa: -1..1 of lock).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

YELLOW, BLUE, ORANGE, BIG_ORANGE, UNKNOWN = 0, 1, 2, 3, 4
COLOR_NAMES = {YELLOW: 'yellow', BLUE: 'blue', ORANGE: 'orange', BIG_ORANGE: 'big_orange',
               UNKNOWN: 'unknown'}
COLOR_IDS = {v: k for k, v in COLOR_NAMES.items()}
N_COLORS = 5
# display colours (RGB) used by every view
COLOR_RGB = {YELLOW: (255, 210, 0), BLUE: (40, 110, 255), ORANGE: (255, 130, 0),
             BIG_ORANGE: (255, 70, 0), UNKNOWN: (190, 190, 190)}


def empty_cones() -> np.ndarray:
    return np.zeros((0, 4))


def make_cones(xy, color=UNKNOWN, conf=1.0) -> np.ndarray:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    out = np.empty((len(xy), 4))
    out[:, :2] = xy
    out[:, 2] = color
    out[:, 3] = conf
    return out


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw])


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def to_car(points_xy: np.ndarray, pose: Pose2D) -> np.ndarray:
    """World (N, 2) -> car frame. Returns a new array."""
    p = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    dx, dy = p[:, 0] - pose.x, p[:, 1] - pose.y
    return np.column_stack([c * dx + s * dy, -s * dx + c * dy])


def to_world(points_xy: np.ndarray, pose: Pose2D) -> np.ndarray:
    """Car frame (N, 2) -> world. Returns a new array."""
    p = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return np.column_stack([pose.x + c * p[:, 0] - s * p[:, 1],
                            pose.y + s * p[:, 0] + c * p[:, 1]])


# ---- sensor frames (what a backend hands the stack) ----------------------

@dataclass
class ConeDetections:
    """Detection-level sensor output: cones already found, in the CAR frame.

    Produced by the fast synthetic sensor models (timeless, Assetto Corsa, or
    the 3D bench in 'detection' fidelity). `cov` is optional per-cone 2x2
    position covariance (N, 2, 2) in the car frame.
    """
    sensor: str
    stamp: float
    cones: np.ndarray
    cov: Optional[np.ndarray] = None
    kind: str = 'detections'


@dataclass
class PointCloud:
    """Raw LiDAR scan, already transformed into the CAR frame (N, 3)."""
    sensor: str
    stamp: float
    points: np.ndarray
    mount: tuple = (0.0, 0.0, 0.0)      # sensor origin in the car frame
    kind: str = 'pointcloud'


@dataclass
class CameraImage:
    """Raw camera frame. `image` is BGR uint8 (H, W, 3), or float32 depth
    metres (H, W) when `is_depth`. `mount` is (x, y, z, roll, pitch, yaw) of
    the camera in the car frame (radians); intrinsics are (fx, fy, cx, cy)."""
    sensor: str
    stamp: float
    image: np.ndarray
    intrinsics: tuple
    mount: tuple
    is_depth: bool = False
    depth: Optional[np.ndarray] = None   # aligned depth, if the rig has a depth/stereo pair
    kind: str = 'image'


@dataclass
class EgoMeasurements:
    """Proprioceptive sensors. None = not fitted on this rig / bench."""
    wheel_speed: Optional[float] = None      # m/s, noisy
    gss_velocity: Optional[tuple] = None     # (vx, vy) body frame, ground-speed sensor
    yaw_rate: Optional[float] = None         # rad/s, gyro z
    accel: Optional[tuple] = None            # (ax, ay) body frame
    gnss_xy: Optional[tuple] = None          # world ENU metres (already converted)
    steering: Optional[float] = None         # measured road-wheel angle, rad, +left


@dataclass
class GroundTruth:
    """Simulator truth. Used for metrics, for the GT driver in the perception
    bench, and by algorithms ONLY when the profile allows it
    (estimation.odometry = 'ground_truth')."""
    pose: Pose2D
    speed: float
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0
    steering: float = 0.0                    # road-wheel angle, rad, +left
    cones_world: Optional[np.ndarray] = None  # (N, 4) whole track, world frame
    cones_hit: Optional[np.ndarray] = None    # (N,) bool


@dataclass
class Observation:
    t: float                                  # sim time, s
    frames: Dict[str, object] = field(default_factory=dict)   # sensor name -> frame
    ego: EgoMeasurements = field(default_factory=EgoMeasurements)
    truth: Optional[GroundTruth] = None


@dataclass
class ControlCommand:
    steering: float = 0.0          # road-wheel angle, rad, +LEFT
    throttle: float = 0.0          # 0..1
    brake: float = 0.0             # 0..1
    target_speed: float = 0.0      # m/s, informational (logged, used by speed-mode backends)
    finished: bool = False         # the stack says the mission is over (AS finished)


@dataclass
class StackOutput:
    """Everything one pipeline tick produced, for views, logs and metrics."""
    command: ControlCommand
    cones_by_sensor: Dict[str, np.ndarray] = field(default_factory=dict)  # car frame
    cones: np.ndarray = field(default_factory=empty_cones)                 # fused, car frame
    pose: Pose2D = field(default_factory=Pose2D)                           # estimated
    map_world: np.ndarray = field(default_factory=empty_cones)             # cone map, world
    path: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))     # car frame
    speed_profile: np.ndarray = field(default_factory=lambda: np.zeros(0))
    target: Optional[tuple] = None                                         # lookahead, car frame
    timings_ms: Dict[str, float] = field(default_factory=dict)
    debug: Dict[str, object] = field(default_factory=dict)
    lap: int = 0
