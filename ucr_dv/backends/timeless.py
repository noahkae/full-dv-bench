"""
Timeless bench: a fast, deterministic 2D simulator on its own clock.

A port of ucr-dv-timeless with its physics upgraded and its bugs fixed:
    - one vehicle file for both benches (data/vehicle/*.json): drive force
      from the motor curve and gearing, brake force, drag + rolling resistance
    - kinematic bicycle with CoG slip (as before), plus a lateral grip limit:
      past mu*g the car washes wide (understeer) instead of turning on rails
    - steering actuator: transport delay + slew-rate limit + first-order lag
      (the old 9-sample deque delayed by 1 sample - popleft bug)
    - collision box from the car's real length/width (old: wheelbase x CoG)
    - seeded numpy Generator everywhere (old: global np.random)
    - fsds_lite's Referee for lap timing, so both benches score identically

Sensors are detection-level models (sensors/models.py) - every rig sensor
works here, at its own rate. Runs hundreds of times faster than real time.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..core.types import ControlCommand, GroundTruth, Observation, Pose2D
from ..tracks import track_cones
from ..vehicle import G, VehicleSpec
from .base import Backend, Capabilities, EpisodeStatus
from .common import CourseMonitor, SensorSynth


class TimelessBackend(Backend):
    name = 'timeless'
    label = 'Timeless 2D'
    caps = Capabilities(lockstep=True, deterministic=True, realtime_only=False, raw_lidar=False,
                        raw_camera=False, ground_truth=True, cone_truth=True, teleport=True)

    def __init__(self, physics_hz=200.0, steering_delay=0.05, steering_tau=0.05, grip_limit=True):
        self.physics_hz = physics_hz
        self.steering_delay = steering_delay
        self.steering_tau = steering_tau
        self.grip_limit = grip_limit

    def load(self, track, profile, vehicle: VehicleSpec, seed=0, stack=None):
        self.track = track
        self.profile = profile
        self.vehicle = vehicle
        self.cones = track_cones(track)
        algo = {}
        if stack is not None:
            algo = {s.name: stack.algo_params(s.name) for s in profile.rig.perception()}
        self.synth = SensorSynth(profile.rig, track, seed, algo_params=algo)
        self.monitor = CourseMonitor(track)
        from fsds_lite.simulator import Referee
        self.referee = Referee(track)
        self.seed = seed
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed + 1)
        self.synth.reset(self.seed)
        self.t = 0.0
        self.x = self.y = self.yaw = 0.0
        self.v = 0.0
        self.vy = 0.0
        self.r = 0.0
        self.ax = self.ay = 0.0
        self.delta = 0.0
        self.delta_cmd = 0.0
        n = max(1, int(round(self.steering_delay * self.physics_hz)))
        self.delay = deque([0.0] * n, maxlen=n)
        self.cmd = ControlCommand()
        self.cones_hit = np.zeros(len(self.cones), dtype=bool)
        self.referee.reset()
        self.monitor.reset()
        self.finished = False
        self.stopped_since = None
        self._crash_dnf = None
        return self.observe()

    # ---- control ------------------------------------------------------------
    def apply(self, cmd: ControlCommand):
        self.cmd = cmd
        if cmd.finished and not self.finished:
            self.finished = True
            self.referee.finished = True
            self.referee.finish_time = self.t

    def teleport(self, x, y, yaw):
        self.x, self.y, self.yaw = x, y, yaw
        self.referee._prev = None

    # ---- physics --------------------------------------------------------------
    def _step(self, dt):
        veh, p = self.vehicle, self.vehicle.params
        c = self.cmd
        thr = min(max(c.throttle, 0.0), 1.0)
        brk = min(max(c.brake, 0.0), 1.0)
        if self.finished:
            thr, brk = 0.0, 1.0
        # steering: delay -> lag -> slew limit
        self.delay.append(max(-veh.max_steer, min(veh.max_steer, c.steering)))
        target = self.delay[0]
        lag_target = self.delta + (target - self.delta) * min(1.0, dt / max(self.steering_tau, 1e-4))
        step = max(-veh.steer_rate * dt, min(veh.steer_rate * dt, lag_target - self.delta))
        self.delta += step
        # longitudinal
        v = self.v
        f = thr * veh.max_drive_force(v) - brk * veh.max_brake_force() - veh.resistance(v)
        a = f / p.mass
        v_new = max(0.0, v + a * dt)
        # lateral (kinematic bicycle with CoG slip + grip limit)
        L = p.wheelbase
        delta = self.delta
        beta = math.atan(p.lr * math.tan(delta) / L)
        kappa = math.tan(delta) * math.cos(beta) / L
        a_lat_max = p.mu * G * 0.95
        if self.grip_limit and v_new > 0.5 and v_new * v_new * abs(kappa) > a_lat_max:
            kappa = math.copysign(a_lat_max / (v_new * v_new), kappa)
            beta = math.copysign(min(abs(beta), abs(math.atan(p.lr * kappa))), beta)
            # scrubbing speed while sliding
            v_new = max(0.0, v_new - 0.5 * dt * a_lat_max * 0.2)
        r = v_new * kappa
        self.x += v_new * math.cos(self.yaw + beta) * dt
        self.y += v_new * math.sin(self.yaw + beta) * dt
        self.yaw = math.atan2(math.sin(self.yaw + r * dt), math.cos(self.yaw + r * dt))
        self.ax = (v_new - v) / dt
        self.ay = v_new * r
        self.v, self.r, self.vy = v_new, r, v_new * math.sin(beta)
        self.t += dt

    def _collisions(self):
        p = self.vehicle.params
        rel = self.cones[:, :2] - (self.x, self.y)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        lx = rel[:, 0] * c + rel[:, 1] * s
        ly = -rel[:, 0] * s + rel[:, 1] * c
        r = self.track.radii
        new = (np.abs(lx) <= p.length / 2 + r) & (np.abs(ly) <= p.width / 2 + r) & ~self.cones_hit
        if new.any():
            self.cones_hit |= new
            self.referee.doo += int(new.sum())

    def advance(self, dt):
        n = max(1, int(round(dt * self.physics_hz)))
        h = dt / n
        for i in range(n):
            self._step(h)
            if i % 2 == 0 or i == n - 1:
                self._collisions()
                self.referee.update(self.t, (self.x, self.y))
        self.monitor.update(self.x, self.y)
        if self.finished and self.v < 0.05:
            if self.stopped_since is None:
                self.stopped_since = self.t
        return self.observe()

    # ---- observation ----------------------------------------------------------
    def truth_dict(self):
        return {'x': self.x, 'y': self.y, 'yaw': self.yaw, 'speed': self.v, 'vx': self.v * math.cos(
            math.atan2(self.vy, max(self.v, 1e-6))), 'vy': self.vy, 'yaw_rate': self.r, 'ax': self.ax,
            'ay': self.ay, 'steering': self.delta}

    def observe(self):
        pose = Pose2D(self.x, self.y, self.yaw)
        frames = self.synth.frames(self.t, pose, self.cones_hit)
        truth = self.truth_dict()
        ego = self.synth.ego(self.t, truth)
        gt = GroundTruth(pose=pose, speed=self.v, vx=truth['vx'], vy=self.vy, yaw_rate=self.r,
                         steering=self.delta, cones_world=self.cones, cones_hit=self.cones_hit.copy())
        return Observation(t=self.t, frames=frames, ego=ego, truth=gt)

    def car_state(self):
        return {'x': self.x, 'y': self.y, 'yaw': self.yaw, 'speed': self.v, 'steering': self.delta,
                'throttle': self.cmd.throttle, 'brake': self.cmd.brake, 'ax': self.ax, 'ay': self.ay}

    def status(self):
        ref = self.referee.summary()
        st = EpisodeStatus(t=self.t, laps=list(ref['laps']), laps_required=ref['laps_required'],
                           doo=ref['doo'], off_course=self.monitor.off_course, crashes=ref['crashes'],
                           complete=ref['complete'], finished=self.finished,
                           stopped=self.stopped_since is not None and self.t - self.stopped_since > 0.3,
                           progress=self.monitor.progress, cte=self.monitor.cte, speed=self.v,
                           run_time=ref['run_time'])
        if self.monitor.lost:
            st.dnf = 'left the track (more than {:.0f} m off the centreline)'.format(
                self.monitor.ref.half_width + self.monitor.lost_margin)
        st.extras = {'max_cte': self.monitor.max_cte, 'rms_cte': self.monitor.rms_cte}
        return st
