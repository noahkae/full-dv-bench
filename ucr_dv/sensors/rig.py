"""
Sensor rig: which sensors the car carries, where, and how they behave.

A rig is a list of SensorSpec. Mounts are in the CAR frame (x forward, y left,
z up; metres) with roll/pitch/yaw in DEGREES (yaw counter-clockwise, pitch
positive = looking down). Perception sensors (lidar, camera, stereo) also
carry the algorithm that turns their raw data into cones.

Fidelity, per sensor:
    'detection'  fast synthetic model: ground-truth cones -> what this sensor
                 + algorithm would plausibly report (range, FOV, resolution,
                 occlusion, noise, misses, colour errors). Works on every bench.
    'raw'        real data from the simulator (ray-cast point clouds, rendered
                 images) through the real algorithm. 3D bench only.
    'auto'       raw where the bench supports it, detection otherwise.
"""
from __future__ import annotations

import copy
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .. import paths

PERCEPTION_TYPES = ('lidar', 'camera', 'stereo')
EGO_TYPES = ('wheel_speed', 'imu', 'gnss', 'gss')
SENSOR_TYPES = PERCEPTION_TYPES + EGO_TYPES + ('timeless_classic',)

# Default parameters per sensor type. Every key is editable in the rig editor.
TYPE_DEFAULTS = {
    'lidar': {'channels': 16, 'vfov_upper': 15.0, 'vfov_lower': -15.0, 'hfov': 360.0,
              'h_res_deg': 0.2, 'rate_hz': 10.0, 'max_range': 100.0, 'range_noise': 0.02,
              'false_positive_rate': 0.05},
    'camera': {'width': 1280, 'height': 720, 'hfov': 90.0, 'rate_hz': 30.0, 'min_box_px': 12.0,
               'pixel_noise': 1.5, 'color_accuracy': 0.98, 'max_range': 40.0,
               'false_positive_rate': 0.05, 'render_width': 640},
    'stereo': {'width': 1280, 'height': 720, 'hfov': 110.0, 'rate_hz': 30.0, 'baseline': 0.12,
               'min_box_px': 12.0, 'pixel_noise': 1.5, 'disparity_noise': 0.25,
               'color_accuracy': 0.98, 'max_range': 25.0, 'false_positive_rate': 0.05,
               'render_width': 640},
    'wheel_speed': {'noise': 0.05, 'rate_hz': 100.0},
    'imu': {'gyro_noise': 0.002, 'accel_noise': 0.05, 'gyro_bias': 0.001, 'rate_hz': 250.0},
    'gnss': {'noise': 0.02, 'rate_hz': 10.0},
    'gss': {'noise': 0.02, 'rate_hz': 100.0},
    # the original ucr-dv-timeless detector model, kept for continuity
    'timeless_classic': {'fov': 110.0, 'max_dist': 50.0, 'box_forward': 20.0, 'box_back': -1.0,
                         'box_half_width': 10.0, 'sigma_x': 0.2, 'skew_x': 1.0, 'sigma_y': 0.05,
                         'false_negative': 0.03, 'false_negative_big': 0.10,
                         'spurious_prob': 0.001, 'class_flip': 0.003},
}

DEFAULT_ALGORITHM = {
    'lidar': {'name': 'lidar_grid', 'params': {}},
    'camera': {'name': 'camera_color_blob', 'params': {}},
    'stereo': {'name': 'camera_color_blob', 'params': {}},
}


@dataclass
class Mount:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def rad(self):
        return (self.x, self.y, self.z, math.radians(self.roll), math.radians(self.pitch),
                math.radians(self.yaw))


@dataclass
class SensorSpec:
    name: str
    type: str
    mount: Mount = field(default_factory=Mount)
    params: Dict[str, float] = field(default_factory=dict)
    algorithm: Optional[Dict] = None        # {'name': ..., 'params': {...}} for perception sensors
    fidelity: str = 'auto'
    enabled: bool = True
    preset: str = ''

    def p(self, key):
        if key in self.params:
            return self.params[key]
        return TYPE_DEFAULTS.get(self.type, {}).get(key)

    @property
    def is_perception(self):
        return self.type in PERCEPTION_TYPES or self.type == 'timeless_classic'

    def full_params(self):
        out = dict(TYPE_DEFAULTS.get(self.type, {}))
        out.update(self.params)
        return out

    def focal_px(self):
        return self.p('width') / (2.0 * math.tan(math.radians(self.p('hfov')) / 2.0))

    def vfov(self):
        """Vertical FOV (deg) of a camera from its aspect ratio."""
        return math.degrees(2.0 * math.atan(math.tan(math.radians(self.p('hfov')) / 2.0)
                                            * self.p('height') / float(self.p('width'))))

    def to_dict(self):
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d['mount'] = Mount(**d.get('mount', {}))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Rig:
    name: str = 'rig'
    sensors: List[SensorSpec] = field(default_factory=list)

    def perception(self):
        return [s for s in self.sensors if s.enabled and s.is_perception]

    def ego(self, kind):
        for s in self.sensors:
            if s.enabled and s.type == kind:
                return s
        return None

    def get(self, name):
        for s in self.sensors:
            if s.name == name:
                return s
        raise KeyError(name)

    def unique_name(self, base):
        names = {s.name for s in self.sensors}
        if base not in names:
            return base
        k = 2
        while '{}{}'.format(base, k) in names:
            k += 1
        return '{}{}'.format(base, k)

    def to_dict(self):
        return {'name': self.name, 'sensors': [s.to_dict() for s in self.sensors]}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get('name', 'rig'), [SensorSpec.from_dict(s) for s in d.get('sensors', [])])

    def copy(self):
        return Rig.from_dict(copy.deepcopy(self.to_dict()))


# ---- presets from fsd-research (see docs/RESEARCH_NOTES.md) ----------------
# Specs are datasheet-level approximations; mounts are typical FS placements.

PRESETS = {
    'Hesai Pandar40P (KIT 2021)': ('lidar', {'channels': 40, 'vfov_upper': 7.0, 'vfov_lower': -16.0,
                                             'hfov': 360.0, 'h_res_deg': 0.2, 'rate_hz': 10.0,
                                             'max_range': 200.0, 'range_noise': 0.02},
                                   Mount(x=1.0, z=0.35)),
    'Velodyne VLP-16 (IIT Bombay)': ('lidar', {'channels': 16, 'vfov_upper': 15.0, 'vfov_lower': -15.0,
                                               'hfov': 360.0, 'h_res_deg': 0.2, 'rate_hz': 10.0,
                                               'max_range': 100.0, 'range_noise': 0.03},
                                     Mount(x=1.0, z=0.35)),
    'Robosense Helios 16P (Heriot-Watt)': ('lidar', {'channels': 16, 'vfov_upper': 15.0, 'vfov_lower': -15.0,
                                                     'hfov': 360.0, 'h_res_deg': 0.2, 'rate_hz': 10.0,
                                                     'max_range': 150.0, 'range_noise': 0.02},
                                           Mount(x=1.0, z=0.35)),
    'Ouster OS1-64 (generic 64 ch)': ('lidar', {'channels': 64, 'vfov_upper': 22.5, 'vfov_lower': -22.5,
                                                'hfov': 360.0, 'h_res_deg': 0.35, 'rate_hz': 10.0,
                                                'max_range': 120.0, 'range_noise': 0.03},
                                      Mount(x=1.0, z=0.35)),
    'AMZ front-wing 32 ch (AMZ 2019)': ('lidar', {'channels': 32, 'vfov_upper': 10.0, 'vfov_lower': -15.0,
                                                  'hfov': 180.0, 'h_res_deg': 0.2, 'rate_hz': 10.0,
                                                  'max_range': 100.0, 'range_noise': 0.02},
                                        Mount(x=1.2, z=0.15)),
    'FSDS default 4-ch LiDAR': ('lidar', {'channels': 4, 'vfov_upper': 5.0, 'vfov_lower': -5.0,
                                          'hfov': 180.0, 'h_res_deg': 180.0 / 1024, 'rate_hz': 10.0,
                                          'max_range': 100.0, 'range_noise': 0.0},
                                Mount(x=0.45, z=0.55)),
    'ZED 2i stereo (IIT/HW/Fontan)': ('stereo', {'width': 1280, 'height': 720, 'hfov': 110.0,
                                                 'rate_hz': 30.0, 'baseline': 0.12, 'max_range': 20.0},
                                      Mount(x=-0.2, z=1.1, pitch=5.0)),
    'Mono 12 mm global shutter (AMZ far)': ('camera', {'width': 1600, 'height': 1200, 'hfov': 54.5,
                                                       'rate_hz': 30.0, 'max_range': 40.0},
                                            Mount(x=-0.2, z=1.1, pitch=2.0)),
    'Mono wide 1080p': ('camera', {'width': 1920, 'height': 1080, 'hfov': 90.0, 'rate_hz': 30.0,
                                   'max_range': 30.0},
                        Mount(x=-0.2, z=1.1, pitch=4.0)),
    'Wheel speed sensors': ('wheel_speed', {}, Mount()),
    'IMU (MPU-6000 class)': ('imu', {}, Mount()),
    'GNSS RTK (u-blox F9P)': ('gnss', {'noise': 0.02}, Mount(z=1.2)),
    'Optical ground-speed sensor': ('gss', {}, Mount(x=-0.5, z=0.3)),
    'Timeless classic detector': ('timeless_classic', {}, Mount()),
}

NAME_STEMS = {'lidar': 'lidar', 'camera': 'cam', 'stereo': 'stereo', 'wheel_speed': 'wheels',
              'imu': 'imu', 'gnss': 'gnss', 'gss': 'gss', 'timeless_classic': 'detector'}


def from_preset(preset: str, name: Optional[str] = None) -> SensorSpec:
    kind, params, mount = PRESETS[preset]
    algo = copy.deepcopy(DEFAULT_ALGORITHM.get(kind))
    return SensorSpec(name or NAME_STEMS[kind], kind, copy.deepcopy(mount), dict(params), algo,
                      preset=preset)


def default_rig() -> Rig:
    """One 16-channel LiDAR + wheel speed + IMU: the measured-robust baseline."""
    lid = from_preset('Velodyne VLP-16 (IIT Bombay)', 'lidar')
    return Rig('lidar_baseline', [lid, from_preset('Wheel speed sensors', 'wheels'),
                                  from_preset('IMU (MPU-6000 class)', 'imu')])


def lidar_camera_rig() -> Rig:
    lid = from_preset('Velodyne VLP-16 (IIT Bombay)', 'lidar')
    cam = from_preset('Mono wide 1080p', 'cam')
    return Rig('lidar_camera', [lid, cam, from_preset('Wheel speed sensors', 'wheels'),
                                from_preset('IMU (MPU-6000 class)', 'imu'),
                                from_preset('GNSS RTK (u-blox F9P)', 'gnss')])


def list_rigs():
    if not os.path.isdir(paths.RIGS):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(paths.RIGS) if f.endswith('.json'))


def load_rig(name_or_path) -> Rig:
    path = name_or_path if os.path.isfile(name_or_path) else os.path.join(paths.RIGS, name_or_path + '.json')
    with open(path) as f:
        return Rig.from_dict(json.load(f))


def save_rig(rig: Rig, path=None):
    path = path or os.path.join(paths.RIGS, rig.name + '.json')
    with open(path, 'w') as f:
        json.dump(rig.to_dict(), f, indent=2)
    return path


# ---- fsds_lite conversion (3D bench, raw fidelity) --------------------------

def to_lite_lidar(spec: SensorSpec):
    from fsds_lite.sensors import LidarConfig
    hfov = float(spec.p('hfov'))
    cols = max(1, int(round(hfov / float(spec.p('h_res_deg')))))
    ch = int(spec.p('channels'))
    m = spec.mount
    return LidarConfig(name=spec.name, x=m.x, y=m.y, z=m.z, roll=m.roll, pitch=m.pitch, yaw=m.yaw,
                       n_lasers=ch, points_per_scan=ch * cols,
                       vfov_upper=float(spec.p('vfov_upper')), vfov_lower=float(spec.p('vfov_lower')),
                       hfov_start=-hfov / 2.0, hfov_end=hfov / 2.0,
                       rotations_per_second=float(spec.p('rate_hz')),
                       max_range=float(spec.p('max_range')), range_noise_std=float(spec.p('range_noise')))


def to_lite_cameras(spec: SensorSpec):
    """A camera renders at `render_width` (aspect kept) to keep the 3D bench
    fast; the algorithm scales pixel thresholds by the render/sensor ratio.
    Stereo and depth sensors get a co-located depth camera (ImageType 2)."""
    from fsds_lite.sensors import CameraConfig
    w = int(spec.p('render_width') or spec.p('width'))
    h = max(16, int(round(w * spec.p('height') / float(spec.p('width')))))
    m = spec.mount
    rgb = CameraConfig(name=spec.name, x=m.x, y=m.y, z=m.z, roll=m.roll, pitch=m.pitch, yaw=m.yaw,
                       width=w, height=h, fov_deg=float(spec.p('hfov')), rate_hz=float(spec.p('rate_hz')),
                       image_type=0)
    out = [rgb]
    if spec.type == 'stereo':
        out.append(CameraConfig(name=spec.name + '_depth', x=m.x, y=m.y, z=m.z, roll=m.roll,
                                pitch=m.pitch, yaw=m.yaw, width=w, height=h,
                                fov_deg=float(spec.p('hfov')), rate_hz=float(spec.p('rate_hz')),
                                image_type=2, depth_noise=False))
    return out
