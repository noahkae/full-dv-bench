"""
3D bench: the vendored fsds_lite simulator, run in LOCKSTEP with the stack.

    - 4-wheel dynamics with tyres, load transfer, gearbox (fsds_lite Vehicle)
    - raw sensors: exact ray-cast LiDAR point clouds and moderngl-rendered
      camera images, built straight from the rig (no settings.json)
    - sensors set to 'detection' fidelity use the same synthetic models as the
      timeless bench, so you can promote a stack in two rungs:
        rung 2 = 3D dynamics + detection-level perception
        rung 3 = 3D dynamics + raw perception through the real algorithms
    - fsds_lite's Referee scores it, CourseMonitor adds OC / lost-car checks

Lockstep: the runner's thread calls advance(dt), which steps physics
dt*physics_hz times; LiDAR samples arrive synchronously, camera frames come
from the render thread one camera period late (like a real camera's latency)
and `LockstepCameraRenderer.wait_idle()` makes that arrival deterministic.

The embedded 3D/top-down view (fsds_lite SimPanel) reads `self.sim`.
"""
from __future__ import annotations

import math
import threading

import numpy as np

from .. import paths  # noqa: F401
from ..core.types import CameraImage, ControlCommand, GroundTruth, Observation, PointCloud, Pose2D
from ..sensors.rig import to_lite_cameras, to_lite_lidar
from ..tracks import track_cones
from .base import Backend, Capabilities, EpisodeStatus
from .common import CourseMonitor, SensorSynth


def _rot(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _lockstep_renderer_cls():
    from fsds_lite.render.camera_sensor import CameraRenderer

    class LockstepCameraRenderer(CameraRenderer):
        """Never drops a request; `wait_idle()` blocks until every request is rendered."""

        def __init__(self, *a, **kw):
            super(LockstepCameraRenderer, self).__init__(*a, **kw)
            self._cond = threading.Condition()
            self.submitted = 0
            self.processed = 0

        def submit(self, request):
            with self._cond:
                self.submitted += 1
            try:
                self.queue.put(request, timeout=1.0)
            except Exception:
                with self._cond:
                    self.processed += 1
                    self._cond.notify_all()

        def _render(self, ctx, renderer, targets, req):
            try:
                return super(LockstepCameraRenderer, self)._render(ctx, renderer, targets, req)
            finally:
                with self._cond:
                    self.processed += 1
                    self._cond.notify_all()

        def wait_idle(self, timeout=1.0):
            with self._cond:
                self._cond.wait_for(lambda: self.processed >= self.submitted, timeout)

    return LockstepCameraRenderer


class LiteBackend(Backend):
    name = 'lite'
    label = 'fsds_lite 3D'
    caps = Capabilities(lockstep=True, deterministic=True, realtime_only=False, max_speedup=20.0,
                        raw_lidar=True, raw_camera=True, ground_truth=True, cone_truth=True,
                        teleport=True, has_3d_view=True)

    def __init__(self, physics_hz=500.0, cameras=True, msaa=0, fidelity_override=None):
        self.physics_hz = physics_hz
        self.want_cameras = cameras
        self.msaa = msaa
        self.fidelity_override = fidelity_override or {}
        self.sim = None
        self.renderer = None
        self.camera_error = None

    def fidelity(self, spec):
        if spec.name in self.fidelity_override:
            f = self.fidelity_override[spec.name]
            if f == 'detection':
                return 'detection'
        f = super(LiteBackend, self).fidelity(spec)
        if f == 'raw' and spec.type in ('camera', 'stereo') and not self.want_cameras:
            return 'detection'
        return f

    # ---- build ------------------------------------------------------------
    def load(self, track, profile, vehicle, seed=0, stack=None):
        from fsds_lite.sensors import SensorSuite
        from fsds_lite.simulator import Simulator
        self.track, self.profile, self.vehicle, self.seed = track, profile, vehicle, seed
        self.cones = track_cones(track)
        rig = profile.rig
        self.realised = {s.name: self.fidelity(s) for s in rig.perception()}
        lidars, cams = [], []
        self.mounts = {}
        self.cam_owner = {}
        for spec in rig.perception():
            if self.realised[spec.name] != 'raw':
                continue
            m = spec.mount
            self.mounts[spec.name] = (m.x, m.y, m.z, math.radians(m.roll), math.radians(m.pitch),
                                      math.radians(m.yaw))
            if spec.type == 'lidar':
                lidars.append(to_lite_lidar(spec))
            else:
                for c in to_lite_cameras(spec):
                    cams.append(c)
                    self.cam_owner[c.name] = spec
        self.suite = SensorSuite('FSCar', lidars, None, None, None, cams)
        self.sim = Simulator(track, self.suite, vehicle.params, physics_hz=self.physics_hz, seed=seed)
        self._lock = threading.Lock()
        self._buf = {}
        self.sim.add_listener(self._on_sample)
        if cams:
            try:
                cls = _lockstep_renderer_cls()
                self.renderer = cls(self.sim, cams, samples=self.msaa,
                                    sensor_mounts=[(l.x, l.y, l.z) for l in lidars]).start()
            except Exception as e:                       # no GPU / no moderngl
                self.camera_error = str(e)
                self.renderer = None
                for c in cams:
                    self.realised[self.cam_owner[c.name].name] = 'detection'
                self.suite.cameras = []
        detection = [n for n, f in self.realised.items() if f == 'detection']
        algo = {s.name: stack.algo_params(s.name) for s in rig.perception()} if stack is not None else {}
        self.synth = SensorSynth(rig, track, seed, detection_sensors=detection, algo_params=algo)
        self.monitor = CourseMonitor(track)
        self.reset(seed)

    def _on_sample(self, sample):
        from fsds_lite.simulator import CameraSample, LidarSample
        if isinstance(sample, (LidarSample, CameraSample)):
            with self._lock:
                self._buf[sample.name] = sample

    # ---- lifecycle -------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        sim = self.sim
        sim.reset()
        # reseed so the same seed gives the same run (reset() keeps the RNG state)
        sim.rng = np.random.default_rng(self.seed)
        for lid in sim.lidars:
            lid.rng = sim.rng
        sim._prng.seed(self.seed)
        sim.enable_api_control(True)
        self.synth.reset(self.seed)
        self.monitor.reset()
        self.cmd = ControlCommand()
        self.finished = False
        self.stopped_since = None
        with self._lock:
            self._buf = {}
        return self.observe()

    def close(self):
        if self.renderer is not None:
            try:
                self.renderer.stop()
            except Exception:
                pass
            self.renderer = None
        if self.sim is not None:
            self.sim.stop()

    # ---- control ------------------------------------------------------------
    def apply(self, cmd):
        from fsds_lite.simulator import Controls
        self.cmd = cmd
        if cmd.finished and not self.finished:
            self.finished = True
            self.sim.finish()
            return
        if self.finished:
            return
        steer = -cmd.steering / max(self.vehicle.max_steer, 1e-6)      # +left rad -> fsds +right [-1, 1]
        self.sim.set_api_controls(Controls(throttle=float(min(max(cmd.throttle, 0.0), 1.0)),
                                           steering=float(min(max(steer, -1.0), 1.0)),
                                           brake=float(min(max(cmd.brake, 0.0), 1.0))))

    def teleport(self, x, y, yaw):
        self.sim.teleport(x, y, yaw)

    def advance(self, dt):
        n = max(1, int(round(dt * self.physics_hz)))
        self.sim.step(n)
        if self.renderer is not None:
            self.renderer.wait_idle(1.0)
        s = self.sim.vehicle.state
        self.monitor.update(s.x, s.y)
        if self.finished and s.speed < 0.05 and self.stopped_since is None:
            self.stopped_since = self.sim.sim_time
        return self.observe()

    # ---- observation ----------------------------------------------------------
    def truth_dict(self):
        s = self.sim.vehicle.state
        return {'x': s.x, 'y': s.y, 'yaw': s.yaw, 'speed': s.speed, 'vx': s.vx, 'vy': s.vy,
                'yaw_rate': s.r, 'ax': s.ax, 'ay': s.ay, 'steering': s.delta}

    def observe(self):
        sim = self.sim
        t = sim.sim_time
        with sim.lock:
            truth = self.truth_dict()
            hit = sim.cones_hit.copy()
        pose = Pose2D(truth['x'], truth['y'], truth['yaw'])
        frames = self.synth.frames(t, pose, hit)
        with self._lock:
            buf, self._buf = self._buf, {}
        depth = {}
        for name, smp in buf.items():
            if hasattr(smp, 'points'):
                m = self.mounts.get(name)
                if m is None:
                    continue
                R = _rot(m[3], m[4], m[5])
                pts = smp.points.astype(np.float64) @ R.T + np.array(m[:3])
                frames[name] = PointCloud(name, t, pts, m[:3])
            elif hasattr(smp, 'image'):
                spec = self.cam_owner.get(name)
                if spec is None:
                    continue
                if name.endswith('_depth'):
                    depth[spec.name] = smp.image
                    continue
                frames[spec.name] = CameraImage(spec.name, t, smp.image, smp.intrinsics, self.mounts[spec.name])
        for owner, d in depth.items():
            if owner in frames and isinstance(frames[owner], CameraImage):
                frames[owner].depth = d
        ego = self.synth.ego(t, truth)
        gt = GroundTruth(pose=pose, speed=truth['speed'], vx=truth['vx'], vy=truth['vy'],
                         yaw_rate=truth['yaw_rate'], steering=truth['steering'],
                         cones_world=self.cones, cones_hit=hit)
        return Observation(t=t, frames=frames, ego=ego, truth=gt)

    def car_state(self):
        s = self.sim.vehicle.state
        c = self.cmd
        return {'x': s.x, 'y': s.y, 'yaw': s.yaw, 'speed': s.speed, 'steering': s.delta,
                'throttle': c.throttle, 'brake': c.brake, 'ax': s.ax, 'ay': s.ay,
                'gear': s.gear, 'rpm': s.rpm, 'roll': s.roll, 'pitch': s.pitch}

    def status(self):
        sim = self.sim
        ref = sim.referee.summary()
        st = EpisodeStatus(t=sim.sim_time, laps=list(ref['laps']), laps_required=ref['laps_required'],
                           doo=ref['doo'], off_course=self.monitor.off_course, crashes=ref['crashes'],
                           complete=ref['complete'], finished=self.finished,
                           stopped=self.stopped_since is not None and sim.sim_time - self.stopped_since > 0.3,
                           progress=self.monitor.progress, cte=self.monitor.cte,
                           speed=sim.vehicle.state.speed, run_time=ref['run_time'])
        if self.monitor.lost:
            st.dnf = 'left the track'
        st.extras = {'max_cte': self.monitor.max_cte, 'rms_cte': self.monitor.rms_cte,
                     'fidelity': dict(self.realised), 'camera_error': self.camera_error}
        return st
