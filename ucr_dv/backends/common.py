"""
Pieces every backend shares:

    SensorSynth    detection-level frames + ego measurements from ground truth
                   (timeless always; 3D for sensors in 'detection' fidelity;
                   Assetto Corsa always, from a cone map overlaid on the track)
    CourseMonitor  cross-track error, progress, off-course and lost-car checks
                   against the track's reference line (same rules everywhere)
"""
from __future__ import annotations

import math

import numpy as np

from ..core.types import ConeDetections, EgoMeasurements, Pose2D, to_car
from ..sensors.models import make_model
from ..tracks import track_cones, track_reference


class SensorSynth(object):
    def __init__(self, rig, track, seed=0, detection_sensors=None, algo_params=None):
        """detection_sensors: names to synthesise (default: every perception sensor)."""
        self.rig = rig
        self.track = track
        self.cones_world = track_cones(track)
        self.heights = track.heights
        self.algo_params = algo_params or {}
        perc = rig.perception()
        if detection_sensors is not None:
            perc = [s for s in perc if s.name in detection_sensors]
        self.models = [(s, make_model(s)) for s in perc]
        self.reset(seed)

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.next_t = {s.name: 0.0 for s, _ in self.models}
        self.ego_next = {}
        self.gyro_bias = 0.0
        self.last_gnss = None

    def frames(self, t, pose: Pose2D, cones_hit):
        """Detection frames for every sensor due at time t."""
        out = {}
        standing = ~cones_hit if cones_hit is not None else np.ones(len(self.cones_world), bool)
        world = self.cones_world[standing]
        heights = self.heights[standing]
        car = None
        for spec, model in self.models:
            if t + 1e-9 < self.next_t[spec.name]:
                continue
            rate = float(spec.p('rate_hz') or 10.0)
            self.next_t[spec.name] = max(self.next_t[spec.name] + 1.0 / rate, t)
            if car is None:
                car = world.copy()
                car[:, :2] = to_car(world[:, :2], pose)
                # only cones within 120 m matter
                near = np.hypot(car[:, 0], car[:, 1]) < 120.0
                car, heights_n = car[near], heights[near]
            out[spec.name] = model.detect(car, heights_n, self.rng, t, self.algo_params.get(spec.name))
        return out

    def ego(self, t, truth):
        """EgoMeasurements from the true state (speed, yaw rate, accel, position)."""
        rig, rng = self.rig, self.rng
        e = EgoMeasurements()
        ws = rig.ego('wheel_speed')
        if ws is not None:
            e.wheel_speed = max(0.0, truth['speed'] + rng.normal(0.0, float(ws.p('noise'))))
        gss = rig.ego('gss')
        if gss is not None:
            n = float(gss.p('noise'))
            e.gss_velocity = (truth['vx'] + rng.normal(0, n), truth['vy'] + rng.normal(0, n))
        imu = rig.ego('imu')
        if imu is not None:
            self.gyro_bias += rng.normal(0.0, float(imu.p('gyro_bias')) * 0.01)
            e.yaw_rate = truth['yaw_rate'] + self.gyro_bias + rng.normal(0.0, float(imu.p('gyro_noise')))
            an = float(imu.p('accel_noise'))
            e.accel = (truth.get('ax', 0.0) + rng.normal(0, an), truth.get('ay', 0.0) + rng.normal(0, an))
        gnss = rig.ego('gnss')
        if gnss is not None:
            due = self.ego_next.get('gnss', 0.0)
            if t + 1e-9 >= due:
                self.ego_next['gnss'] = due + 1.0 / float(gnss.p('rate_hz'))
                n = float(gnss.p('noise'))
                self.last_gnss = (truth['x'] + rng.normal(0, n), truth['y'] + rng.normal(0, n))
                e.gnss_xy = self.last_gnss
        e.steering = truth.get('steering')
        return e


class CourseMonitor(object):
    """Off-course (OC) and lost-car detection, progress and cross-track error."""

    def __init__(self, track, oc_margin=0.6, lost_margin=6.0):
        self.ref = track_reference(track)
        self.oc_margin = oc_margin
        self.lost_margin = lost_margin
        self.reset()

    def reset(self):
        self.idx = None
        self.off = False
        self.off_course = 0
        self.lost = False
        self.cte = 0.0
        self.max_cte = 0.0
        self.sq_cte = 0.0
        self.n = 0
        self.s = 0.0

    def update(self, x, y):
        k, cte, s = self.ref.project(x, y, self.idx)
        self.idx = k
        self.cte, self.s = cte, s
        self.max_cte = max(self.max_cte, abs(cte))
        self.sq_cte += cte * cte
        self.n += 1
        # FS rules: off course = all four wheels past the cone line, i.e. the
        # car's centre beyond half the track width plus half its own track
        outside = abs(cte) > self.ref.half_width + self.oc_margin
        if outside and not self.off:
            self.off_course += 1
        self.off = outside
        if abs(cte) > self.ref.half_width + self.lost_margin:
            self.lost = True
        return cte

    @property
    def progress(self):
        return min(1.0, max(0.0, self.s / max(self.ref.length, 1e-6)))

    @property
    def rms_cte(self):
        return math.sqrt(self.sq_cte / max(self.n, 1))
