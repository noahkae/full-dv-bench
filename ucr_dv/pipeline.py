"""
The Stack: one profile's algorithms, wired in the order the car runs them.

    sensor frames ──► per-sensor detector ──► (motion-compensated) cones per sensor
                                                         │
    ego sensors ──► odometry ──► pose ───────────────────┤
                                                         ▼
                                     fusion ──► cone map ──► mission ──► planner
                                                                          │
                          ControlCommand ◄── longitudinal ◄── speed ◄─────┤
                                         ◄── lateral ◄────────────────────┘

`Stack.step(obs, dt)` is bench-agnostic: timeless, 3D and (later) Assetto
Corsa all hand it an Observation and apply the ControlCommand it returns.
It never reads `obs.truth` unless the profile chose ground-truth odometry.
"""
from __future__ import annotations

import time
import traceback
from typing import Dict, Optional

import numpy as np

from .core import registry
from .core.profile import Profile
from .core.types import (CameraImage, ConeDetections, ControlCommand, Observation, PointCloud,
                         StackOutput, empty_cones, to_car, to_world)
from .vehicle import VehicleSpec, load_vehicle


class _SensorTrack(object):
    """Latest cones from one sensor, kept in the world frame for motion compensation."""

    def __init__(self):
        self.world = None
        self.color = None
        self.conf = None
        self.cov = None
        self.stamp = -1e9
        self.debug = {}


class Stack(object):
    def __init__(self, profile: Profile, event='autocross', vehicle: Optional[VehicleSpec] = None, seed=0):
        self.profile = profile
        self.event = event
        self.vehicle = vehicle or VehicleSpec(load_vehicle(profile.vehicle))
        self.seed = seed
        self._build()

    def _build(self):
        p = self.profile
        ctx = {'vehicle': self.vehicle, 'event': self.event, 'seed': self.seed,
               'sensor_types': {s.name: s.type for s in p.rig.sensors}}
        self.stages = {}
        for kind in ('fusion', 'odometry', 'mapping', 'planner', 'speed', 'lateral', 'longitudinal', 'mission'):
            s = p.stages[kind]
            self.stages[kind] = registry.get(kind, s['name']).create(s.get('params'), **ctx)
        self.detectors = {}
        for spec in p.rig.perception():
            algo = spec.algorithm or {}
            kind = {'lidar': 'lidar', 'camera': 'camera', 'stereo': 'camera'}.get(spec.type)
            if kind and algo.get('name'):
                self.detectors[spec.name] = registry.get(kind, algo['name']).create(algo.get('params'), **ctx)
        from .sensors.models import make_model
        self.models = {s.name: make_model(s) for s in p.rig.perception()}
        self.tracks = {}
        self.max_age = 0.35
        self.t_last = None
        self.template = self.stages['mission'].template() if hasattr(self.stages['mission'], 'template') else None
        self.errors = 0
        self.last_error = ''

    @property
    def uses_ground_truth(self):
        return getattr(self.stages['odometry'], 'uses_gt', False)

    def reset(self):
        for s in list(self.stages.values()) + list(self.detectors.values()):
            s.reset()
        self.tracks = {}
        self.t_last = None

    # ---- live tuning (called between ticks) -----------------------------
    def set_params(self, stage, params):
        """stage: a pipeline stage name or 'sensor:<name>' for a detector."""
        if stage.startswith('sensor:'):
            comp = self.detectors.get(stage[7:])
        else:
            comp = self.stages.get(stage)
        if comp is not None:
            comp.set_params(**params)

    def algo_params(self, sensor):
        d = self.detectors.get(sensor)
        if d is not None:
            return d.params
        spec = self.profile.rig.get(sensor)
        return (spec.algorithm or {}).get('params', {}) if spec.algorithm else {}

    # ---- one tick ----------------------------------------------------------
    def step(self, obs: Observation, dt=None) -> StackOutput:
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        if dt is None:
            dt = 0.0 if self.t_last is None else obs.t - self.t_last
        self.t_last = obs.t

        # 1. odometry first, so detections can be placed in the world
        odo = self.stages['odometry']
        pose = odo.update(obs, dt)
        v = odo.speed
        t1 = time.perf_counter()
        timings['odometry'] = (t1 - t0) * 1e3

        # 2. per-sensor perception
        debug = {}
        for name, frame in obs.frames.items():
            ts = time.perf_counter()
            cones, cov, dbg = self._perceive(name, frame)
            if cones is None:
                continue
            tr = self.tracks.setdefault(name, _SensorTrack())
            tr.world = to_world(cones[:, :2], pose) if len(cones) else np.zeros((0, 2))
            tr.color, tr.conf, tr.cov = cones[:, 2].copy(), cones[:, 3].copy(), cov
            tr.stamp = obs.t
            tr.debug = dbg
            timings['perception:' + name] = (time.perf_counter() - ts) * 1e3
            if dbg:
                debug['perception:' + name] = dbg
        cones_by_sensor, cov_by_sensor = {}, {}
        for name, tr in self.tracks.items():
            if obs.t - tr.stamp > self.max_age or tr.world is None:
                continue
            c = np.zeros((len(tr.world), 4))
            if len(tr.world):
                c[:, :2] = to_car(tr.world, pose)
                c[:, 2], c[:, 3] = tr.color, tr.conf
            cones_by_sensor[name] = c
            if tr.cov is not None:
                cov_by_sensor[name] = tr.cov
        t2 = time.perf_counter()

        # 3. fusion
        fused = self.stages['fusion'].fuse(cones_by_sensor, cov_by_sensor) if cones_by_sensor else empty_cones()
        t3 = time.perf_counter()
        timings['fusion'] = (t3 - t2) * 1e3

        # 4. mapping (only fresh frames update the map, so 10 Hz sensors aren't double counted)
        mapper = self.stages['mapping']
        fresh = any(n in self.tracks and self.tracks[n].stamp == obs.t for n in obs.frames)
        if fresh and hasattr(mapper, 'localize') and not self.uses_ground_truth:
            fixed = mapper.localize(fused, pose)
            if fixed is not pose:
                # move everything expressed in the old pose to the corrected one
                odo.correct(fixed)
                fused = fused.copy()
                pose = fixed
        if fresh:
            map_world = mapper.update(fused, pose)
        else:
            map_world = mapper.confirmed() if hasattr(mapper, 'confirmed') else empty_cones()
        local = mapper.local(fused, pose)
        t4 = time.perf_counter()
        timings['mapping'] = (t4 - t3) * 1e3

        # 5. mission
        mission = self.stages['mission'].update(fused, odo.distance, v, obs.t)
        # 6. planning
        ctx = {'pose': pose, 'speed': v, 'template': self.template, 'mission': mission,
               'speed_cap': mission.speed_cap, 'vehicle': self.vehicle, 'dt': dt}
        path = self.stages['planner'].plan(local, ctx)
        t5 = time.perf_counter()
        timings['planner'] = (t5 - t4) * 1e3
        profile, v_target = self.stages['speed'].profile(path, v, ctx)
        t6 = time.perf_counter()
        timings['speed'] = (t6 - t5) * 1e3

        # 7. control
        lat = self.stages['lateral']
        steer = lat.steer(path, v, ctx)
        a_ff = 0.0
        throttle, brake, a_cmd = self.stages['longitudinal'].command(v_target, v, dt, a_ff)
        t7 = time.perf_counter()
        timings['control'] = (t7 - t6) * 1e3
        timings['total'] = (t7 - t0) * 1e3

        cmd = ControlCommand(steering=float(steer), throttle=float(throttle), brake=float(brake),
                             target_speed=float(v_target), finished=bool(mission.finished))
        debug['a_cmd'] = a_cmd
        debug['mission'] = mission
        if hasattr(self.stages['planner'], 'last_debug'):
            debug['planner'] = self.stages['planner'].last_debug
        return StackOutput(command=cmd, cones_by_sensor=cones_by_sensor, cones=fused, pose=pose,
                           map_world=map_world, path=path, speed_profile=profile,
                           target=lat.target, timings_ms=timings, debug=debug, lap=mission.laps)

    def _perceive(self, name, frame):
        """frame -> (cones (N,4) car frame, cov or None, debug)."""
        if isinstance(frame, ConeDetections):
            c = frame.cones
            return c, frame.cov, {}
        det = self.detectors.get(name)
        if det is None:
            return None, None, {}
        try:
            if isinstance(frame, PointCloud):
                cones, dbg = det.detect(frame)
            elif isinstance(frame, CameraImage):
                cones, dbg = det.detect(frame)
            else:
                return None, None, {}
        except Exception:
            self.errors += 1
            self.last_error = traceback.format_exc()
            return None, None, {}
        return cones, self._model_cov(name, cones), dbg

    def _model_cov(self, name, cones):
        """Covariance for raw detections from the sensor's noise model (range/bearing)."""
        model = self.models.get(name)
        if model is None or not len(cones) or not hasattr(model, '_cov'):
            return None
        spec = model.spec
        m = spec.mount
        dx, dy = cones[:, 0] - m.x, cones[:, 1] - m.y
        r = np.hypot(dx, dy)
        b = np.arctan2(dy, dx) - np.radians(m.yaw)
        if spec.type == 'lidar':
            sr = np.full(len(r), float(spec.p('range_noise')) + 0.03)
            sb = np.radians(float(spec.p('h_res_deg')))
        else:
            px = model.box_px(r, np.full(len(r), 0.325))
            sr = model.range_sigma(r, px) * 1.5
            sb = 2.0 * float(spec.p('pixel_noise')) / spec.focal_px()
        return model._cov(r, b, sr, sb, 0)
