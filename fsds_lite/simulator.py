"""
The simulator core: one vehicle, one track, a sensor suite, and a clock.

Threading model: `run()` steps physics on its own thread at a fixed rate,
locked to wall-clock time (scaled by `realtime_factor`). Everything else -
the RPC server, the ROS bridge, the viewer - talks to it through the
thread-safe methods below. Sensor samples are pushed to listeners (callables
taking one sample object) from the physics thread - or, for camera images,
from the render thread - so listeners must be quick (publish a message, store
a value) and must not call back into blocking code.

Cameras: the physics thread never touches the GPU. When a camera is due it
hands a CameraRequest (pose + stamp) to `camera_sink`, normally
render.camera_sensor.CameraRenderer, which renders on its own thread and
calls `emit()` with the CameraSample. With no sink, cameras are simply off.

Referee: like FSDS's, it counts knocked cones (DOO) and records lap times
as the car crosses the track's timing lines (`Referee`). Solid trackside
objects (barriers, tyre stacks, poles, timekeeping equipment) stop the car
dead and count as crashes; stacked spare cones just get knocked over.

API control mirrors AirSim: commands from setCarControls are ignored unless
API control is enabled, and reset() drops API control. When API control is
off, the car follows `manual` controls (the viewer's keyboard).
"""
import logging
import math
import random
import threading
import time
from dataclasses import dataclass

import numpy as np

from .geometry import enu_to_geodetic, rpy_to_matrix, rpy_to_quat
from .sensors import Lidar
from .vehicle import Vehicle

log = logging.getLogger('fsds_lite.sim')

RUN_TICK = 0.004        # s between physics batches in real-time mode


@dataclass
class Controls:
    throttle: float = 0.0
    steering: float = 0.0
    brake: float = 0.0
    handbrake: bool = False


# --- sensor samples (stamp = wall-clock seconds, ROS-style) --------------

@dataclass
class LidarSample:
    name: str
    stamp: float
    points: np.ndarray              # (N, 3) float32, sensor frame
    pose: tuple                     # sensor (x, y, z, yaw) in world
    tilt: tuple = (0.0, 0.0)        # body (roll, pitch) at capture


@dataclass
class BodyPose:
    x: float
    y: float
    yaw: float
    roll: float = 0.0
    pitch: float = 0.0

    def matrix(self):
        return rpy_to_matrix(self.roll, self.pitch, self.yaw)


@dataclass
class CameraRequest:
    camera: object                  # sensors.CameraConfig
    stamp: float
    body: BodyPose
    cones_hit: np.ndarray
    spare_hit: np.ndarray = None    # knocked spare-cone stacks


@dataclass
class CameraSample:
    name: str
    stamp: float
    image: np.ndarray               # (H, W, 3) uint8, BGR like the FSDS bridge
    intrinsics: tuple               # (fx, fy, cx, cy)
    position: tuple                 # camera position in world
    orientation: tuple              # camera (x, y, z, w) in world


@dataclass
class ImuSample:
    name: str
    stamp: float
    orientation: tuple              # (x, y, z, w)
    angular_velocity: tuple
    linear_acceleration: tuple


@dataclass
class GpsSample:
    name: str
    stamp: float
    latitude: float
    longitude: float
    altitude: float
    eph: float
    epv: float
    velocity: tuple                 # world ENU


@dataclass
class GssSample:
    name: str
    stamp: float
    linear_velocity: tuple          # body frame
    angular_velocity: tuple


@dataclass
class WheelStatesSample:
    stamp: float
    rpm: tuple                      # FL, FR, RL, RR
    rotation_angle: tuple           # rad
    steering_angle: tuple           # rad (ENU, +ve = left)
    name: str = 'wheel_states'


@dataclass
class OdomSample:
    stamp: float
    position: tuple
    orientation: tuple
    linear_velocity: tuple          # body frame
    angular_velocity: tuple
    linear_acceleration: tuple      # body frame


class Referee(object):
    """Lap timing and penalties, after FSDS's AReferee.

    Each forward crossing of the track's start/finish line(s) is a timing
    event. On a closed course (autocross, trackdrive, skidpad) the first
    crossing starts the clock and each later one completes a lap; on
    acceleration the start line starts it and the finish line stops it.
    The mission is complete after `track.laps` laps (skidpad: 4).
    """

    def __init__(self, track):
        self.track = track
        self.reset()

    def reset(self):
        self.laps = []
        self.lap_start = None
        self.doo = 0
        self.crashes = 0
        self.mission_complete = False
        self.finished = False           # the AS sent /signal/finished
        self.finish_time = None
        self._prev = None

    def update(self, t, pos):
        """Feed the car position; returns 'lap' / 'start' / 'complete' events."""
        tr, prev, events = self.track, self._prev, []
        self._prev = pos
        if prev is None or tr.start_line is None or self.mission_complete:
            return events
        closed = tr.finish_line is tr.start_line
        if tr.start_line.crossed(prev, pos):
            if closed and self.lap_start is not None:
                self.laps.append(t - self.lap_start)
                events.append('lap')
            if closed or self.lap_start is None:
                self.lap_start = t
                events.append('start')
        if not closed and self.lap_start is not None and tr.finish_line.crossed(prev, pos):
            self.laps.append(t - self.lap_start)
            events.append('lap')
        if len(self.laps) >= tr.laps:
            self.mission_complete = True
            events.append('complete')
        return events

    def run_time(self):
        """Official time for the mission (no penalties), or None if not done."""
        if not self.mission_complete:
            return None
        if self.track.mission == 'skidpad':
            return 0.5 * (self.laps[1] + self.laps[3])     # D 4.2.2: mean of the timed laps
        return sum(self.laps[:self.track.laps])

    def summary(self):
        return {'mission': self.track.mission, 'laps': list(self.laps), 'laps_required': self.track.laps,
                'doo': self.doo, 'crashes': self.crashes, 'complete': self.mission_complete,
                'finished': self.finished, 'run_time': self.run_time(),
                'lap_running': self.lap_start is not None and not self.mission_complete}


def _obb_overlap(a, b):
    """Separating-axis test for two yawed rectangles (cx, cy, yaw, half_x, half_y)."""
    axes = []
    for r in (a, b):
        c, s = math.cos(r[2]), math.sin(r[2])
        axes += [(c, s), (-s, c)]
    dx, dy = b[0] - a[0], b[1] - a[1]
    for ax, ay in axes:
        def extent(r):
            c, s = math.cos(r[2]), math.sin(r[2])
            return r[3] * abs(c * ax + s * ay) + r[4] * abs(-s * ax + c * ay)
        if abs(dx * ax + dy * ay) > extent(a) + extent(b):
            return False
    return True


@dataclass
class _Periodic:
    period: float
    fire: object
    next_t: float = 0.0


class Simulator(object):
    def __init__(self, track, sensors, vehicle_params=None, physics_hz=1000.0,
                 odom_hz=250.0, seed=None, spawn=(0.0, 0.0, 0.0)):
        self.track = track
        self.sensors = sensors
        self.vehicle = Vehicle(vehicle_params)
        self.dt = 1.0 / physics_hz
        self.spawn = spawn
        self.rng = np.random.default_rng(seed)
        self.lock = threading.RLock()
        self.listeners = []
        self.camera_sink = None             # set by the camera renderer, if any

        self.lidars = [Lidar(cfg, self.rng) for cfg in sensors.lidars]
        self.latest = {}                     # last sample per sensor name, for RPC getters
        self.api_control = False
        self.api_controls = Controls()
        self.manual = Controls()
        self.paused = False
        self.realtime_factor = 1.0
        self._continue_for = None
        self._stop = threading.Event()
        self._clock = None                   # (monotonic time, ideal sim time then) from run()

        self.sim_time = 0.0
        self.wall_epoch = time.time()
        self.cones_hit = np.zeros(len(track), dtype=bool)
        self.spare_hit = np.zeros(len(track.cone_obs), dtype=bool)
        self._cone_xy = track.positions.copy()
        self.referee = Referee(track)
        self._in_contact = False
        self._last_contact = -1e9
        self._safe_pose = None
        self._gyro_bias = [0.0, 0.0, 0.0]
        self._accel_bias = [0.0, 0.0, 0.0]
        # plain-Python RNG for per-sample scalar noise (numpy is slow at size 3)
        self._prng = random.Random(seed)
        self._obs_xy = np.array([(o.x, o.y) for o in track.obstacles]).reshape(-1, 2)
        self._obs_r = np.array([o.bound_radius for o in track.obstacles])

        self._tasks = []
        for lidar in self.lidars:
            self._tasks.append(_Periodic(1.0 / lidar.cfg.rotations_per_second,
                                         self._make_lidar_task(lidar)))
        if sensors.imu:
            self._tasks.append(_Periodic(1.0 / sensors.imu.rate_hz, self._imu))
        if sensors.gps:
            self._tasks.append(_Periodic(1.0 / sensors.gps.rate_hz, self._gps))
        if sensors.gss:
            self._tasks.append(_Periodic(1.0 / sensors.gss.rate_hz, self._gss))
        self._tasks.append(_Periodic(1.0 / odom_hz, self._odom))
        self._tasks.append(_Periodic(0.01, self._wheel_states))
        for cam in sensors.cameras or []:
            self._tasks.append(_Periodic(1.0 / cam.rate_hz, self._make_camera_task(cam)))
        self._tasks.append(_Periodic(0.01, self._check_collisions))
        self.reset()

    # ---- control API (thread-safe) ---------------------------------------
    def add_listener(self, fn):
        self.listeners.append(fn)

    def stamp(self):
        return self.wall_epoch + self.sim_time

    def reset(self):
        with self.lock:
            self.vehicle.reset(*self.spawn)
            self.api_control = False
            self.api_controls = Controls()
            self.manual = Controls()
            self.cones_hit[:] = False
            self.spare_hit[:] = False
            self.referee.reset()
            self._in_contact = False
            self._safe_pose = None
            self._gyro_bias = [0.0, 0.0, 0.0]
            self._accel_bias = [0.0, 0.0, 0.0]
            for task in self._tasks:
                task.next_t = self.sim_time

    def enable_api_control(self, enabled):
        with self.lock:
            self.api_control = bool(enabled)
            if not enabled:
                self.api_controls = Controls()

    def set_api_controls(self, controls):
        """Returns False (and ignores the command) if API control is disabled."""
        with self.lock:
            if not self.api_control:
                return False
            self.api_controls = controls
            return True

    def set_manual_controls(self, controls):
        with self.lock:
            self.manual = controls

    def active_controls(self):
        with self.lock:
            return self.api_controls if self.api_control else self.manual

    def set_paused(self, paused):
        with self.lock:
            self.paused = bool(paused)
            self._continue_for = None

    def continue_for(self, seconds):
        with self.lock:
            self._continue_for = self.sim_time + float(seconds)
            self.paused = False

    def teleport(self, x, y, yaw):
        with self.lock:
            self.vehicle.teleport(x, y, yaw)
            self.referee._prev = None           # a teleport is not a line crossing
            self._safe_pose = None

    def finish(self):
        """The AS says it is done (/signal/finished): hand control back and brake."""
        with self.lock:
            if self.referee.finished:
                return
            self.referee.finished = True
            self.referee.finish_time = self.sim_time
            self.api_control = False
            self.api_controls = Controls()
            self.manual = Controls(brake=1.0)
            summary = self.referee.summary()
        log.info('FINISHED signal: %s', summary)

    def display_lag(self):
        """Sim seconds between the last physics step and "now" on the wall clock.

        Physics runs in batches, so at any instant the state is slightly
        behind. Views extrapolate by this much so motion looks smooth instead
        of jumping from batch to batch.
        """
        clock = self._clock
        if clock is None or self.paused:
            return 0.0
        mono, ideal = clock
        # anchored to the ideal sim time rather than to "budget left", so this
        # stays right even when called halfway through a batch of steps
        lag = ideal + (time.monotonic() - mono) * self.realtime_factor - self.sim_time
        return min(max(lag, 0.0), 0.1)

    def snapshot(self):
        """Cheap copy of everything a viewer needs.

        x/y/yaw are extrapolated to the present (see display_lag); the raw
        physics pose is in raw_x/raw_y/raw_yaw.
        """
        with self.lock:
            s = self.vehicle.state
            lag = self.display_lag()
            vxw, vyw = s.world_velocity()
            return {
                'time': self.sim_time + lag,
                'mono': time.monotonic(),       # wall time this snapshot describes
                'raw_x': s.x, 'raw_y': s.y, 'raw_yaw': s.yaw,
                'x': s.x + vxw * lag, 'y': s.y + vyw * lag, 'yaw': s.yaw + s.r * lag,
                'vx': s.vx, 'vy': s.vy, 'r': s.r, 'speed': s.speed,
                'delta': s.delta,
                'roll': s.roll, 'pitch': s.pitch, 'wheel_spin': s.wheel_spin,
                'omega': list(s.omega), 'wheel_angle': list(s.wheel_angle), 'slip': list(s.slip),
                'ax': s.ax, 'ay': s.ay, 'fz': s.fz, 'gear': s.gear, 'rpm': s.rpm,
                'controls': self.active_controls(),
                'api_control': self.api_control,
                'paused': self.paused,
                'cones_hit': self.cones_hit.copy(),
                'spare_hit': self.spare_hit.copy(),
                'referee': self.referee.summary(),
                'cone_xy': self._cone_xy.copy(),
                'lidar': {n: v for n, v in self.latest.items() if isinstance(v, LidarSample)},
            }

    # ---- stepping ---------------------------------------------------------
    def step(self, n=1):
        """Advance `n` physics steps. Safe to call directly (tests, lockstep)."""
        for _ in range(n):
            with self.lock:
                if self.paused:
                    return
                c = self.api_controls if self.api_control else self.manual
                brake = 1.0 if c.handbrake else c.brake
                self.vehicle.step(self.dt, c.throttle, brake, c.steering)
                self.sim_time += self.dt
                if self._continue_for is not None and self.sim_time >= self._continue_for:
                    self._continue_for = None
                    self.paused = True
                due = []
                for task in self._tasks:
                    if self.sim_time + 1e-9 >= task.next_t:
                        task.next_t += task.period
                        if task.next_t < self.sim_time:     # fell far behind; don't burst
                            task.next_t = self.sim_time + task.period
                        due.append(task)
            # samples are produced outside the lock so the RPC thread is not
            # blocked behind a LiDAR scan
            for task in due:
                sample = task.fire()
                if sample is not None:
                    self.emit(sample)

    def emit(self, sample):
        """Record a sample and hand it to every listener (any thread)."""
        self.latest[getattr(sample, 'name', 'odom')] = sample
        for fn in self.listeners:
            fn(sample)

    def run(self):
        """Real-time loop; returns when stop() is called."""
        self._stop.clear()
        last = time.monotonic()
        budget = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            budget += (now - last) * self.realtime_factor
            last = now
            budget = min(budget, 0.25)              # never try to catch up more than 250 ms
            self._clock = (now, self.sim_time + budget)
            steps = int(budget / self.dt)
            if steps:
                self.step(steps)
                budget -= steps * self.dt
            # wake every ~4 ms and run the steps owed in one batch: waking 1000x/s
            # to run one step each costs more than the physics itself. Commands
            # still land within 4 ms, and views extrapolate (display_lag).
            time.sleep(RUN_TICK)
        self._clock = None

    def stop(self):
        self._stop.set()

    # ---- sensor tasks -----------------------------------------------------
    def body_pose(self):
        with self.lock:
            s = self.vehicle.state
            return BodyPose(s.x, s.y, s.yaw, s.roll, s.pitch)

    def _make_camera_task(self, cam):
        def fire():
            sink = self.camera_sink
            if sink is not None:
                with self.lock:
                    req = CameraRequest(cam, self.stamp(), self.body_pose(), self.cones_hit.copy(),
                                        self.spare_hit.copy())
                sink(req)
            return None
        return fire

    def _make_lidar_task(self, lidar):
        def fire():
            tr = self.track
            with self.lock:
                b = self.body_pose()
                standing = ~self.cones_hit
                spares = tr.cone_obs[~self.spare_hit]
                stamp = self.stamp()
            # track cones and standing spare-cone stacks are all upright cones
            xy = np.vstack([self._cone_xy[standing], spares[:, :2]])
            radii = np.concatenate([tr.radii[standing], spares[:, 2]])
            heights = np.concatenate([tr.heights[standing], spares[:, 3]])
            pts = lidar.scan(b.x, b.y, b.yaw, xy, radii, heights, b.roll, b.pitch,
                             boxes=tr.boxes, cylinders=tr.cylinders)
            o = np.array(b.matrix()) @ lidar.offset
            pose = (b.x + o[0], b.y + o[1], o[2], b.yaw + math.radians(lidar.cfg.yaw))
            return LidarSample(lidar.cfg.name, stamp, pts, pose, (b.roll, b.pitch))
        return fire

    def _imu(self):
        cfg = self.sensors.imu
        with self.lock:
            s = self.vehicle.state
            # specific force (planar accel + gravity) expressed in the tilted body frame
            tilt = np.array(rpy_to_matrix(s.roll, s.pitch, 0.0))
            acc = list(tilt.T @ np.array([s.ax, s.ay, 9.81]))
            gyro = [s.roll_rate, s.pitch_rate, s.r]
            stamp, quat = self.stamp(), rpy_to_quat(s.roll, s.pitch, s.yaw)
        if cfg.noise:
            # AirSim ImuSimple: white noise + slowly random-walking bias
            g = self._prng.gauss
            sqrt_dt = math.sqrt(max(1.0 / cfg.rate_hz, 1e-3))
            g_white, g_walk = cfg.gyro_arw / sqrt_dt, cfg.gyro_bias_stability / math.sqrt(cfg.gyro_tau) * sqrt_dt
            a_white, a_walk = cfg.accel_vrw / sqrt_dt, cfg.accel_bias_stability / math.sqrt(cfg.accel_tau) * sqrt_dt
            gb, ab = self._gyro_bias, self._accel_bias
            gyro = [gyro[i] + g(0.0, g_white) + gb[i] for i in range(3)]
            acc = [acc[i] + g(0.0, a_white) + ab[i] for i in range(3)]
            for i in range(3):
                gb[i] += g(0.0, g_walk)
                ab[i] += g(0.0, a_walk)
        return ImuSample(cfg.name, stamp, quat, tuple(float(g) for g in gyro),
                         tuple(float(a) for a in acc))

    def _gps(self):
        cfg = self.sensors.gps
        with self.lock:
            s = self.vehicle.state
            x, y, speed, stamp = s.x, s.y, s.speed, self.stamp()
            vel = s.world_velocity()
        z = 0.0
        if speed > cfg.noise_min_speed:
            x += self.rng.normal(0.0, cfg.eph)
            y += self.rng.normal(0.0, cfg.eph)
            z += self.rng.normal(0.0, cfg.epv)
        lat, lon, alt = enu_to_geodetic(x, y, z)
        return GpsSample(cfg.name, stamp, lat, lon, alt, cfg.eph, cfg.epv, (vel[0], vel[1], 0.0))

    def _gss(self):
        with self.lock:
            s = self.vehicle.state
            return GssSample(self.sensors.gss.name, self.stamp(),
                             (s.vx, s.vy, 0.0), (0.0, 0.0, s.r))

    def _odom(self):
        with self.lock:
            s = self.vehicle.state
            return OdomSample(self.stamp(), (s.x, s.y, 0.0), rpy_to_quat(s.roll, s.pitch, s.yaw),
                              (s.vx, s.vy, 0.0), (s.roll_rate, s.pitch_rate, s.r),
                              (s.ax, s.ay, 0.0))

    def _wheel_states(self):
        with self.lock:
            s = self.vehicle.state
            steer = (s.delta, s.delta, 0.0, 0.0)
            return WheelStatesSample(self.stamp(), tuple(w * 60.0 / (2 * math.pi) for w in s.omega),
                                     tuple(s.wheel_angle), steer)

    def _check_collisions(self):
        """Cones, spare cones, solid objects, and lap timing (100 Hz)."""
        events = []
        with self.lock:
            s, p, tr = self.vehicle.state, self.vehicle.p, self.track
            c, sn = math.cos(s.yaw), math.sin(s.yaw)

            def in_box(xy, r):
                rel = xy - (s.x, s.y)
                lx = rel[:, 0] * c + rel[:, 1] * sn          # in the car frame
                ly = -rel[:, 0] * sn + rel[:, 1] * c
                return (np.abs(lx) <= p.length / 2 + r) & (np.abs(ly) <= p.width / 2 + r)

            new = in_box(self._cone_xy, tr.radii) & ~self.cones_hit
            if new.any():
                self.cones_hit |= new
                self.referee.doo += int(new.sum())           # DOO: down or out
            if len(tr.cone_obs):
                self.spare_hit |= in_box(tr.cone_obs[:, :2], tr.cone_obs[:, 2])

            # solid objects: the car stops dead where it touched
            car = (s.x, s.y, s.yaw, p.length / 2.0, p.width / 2.0)
            reach = math.hypot(p.length, p.width) / 2.0
            touching = False
            near = []
            if len(self._obs_xy):
                d = np.hypot(self._obs_xy[:, 0] - s.x, self._obs_xy[:, 1] - s.y)
                near = np.nonzero(d <= reach + self._obs_r)[0]
            for k in near:
                o = tr.obstacles[k]
                if o.kind == 'cone':
                    continue
                if o.kind == 'box':
                    touching = _obb_overlap(car, (o.x, o.y, o.yaw, o.half_x, o.half_y))
                else:                                        # cylinder vs car rectangle
                    rx, ry = o.x - s.x, o.y - s.y
                    lx, ly = rx * c + ry * sn, -rx * sn + ry * c
                    qx = min(max(lx, -car[3]), car[3])
                    qy = min(max(ly, -car[4]), car[4])
                    touching = math.hypot(lx - qx, ly - qy) <= o.radius
                if touching:
                    # one incident per contact: pushing on against the wall
                    # (put back, creep forward, touch again) isn't a new crash
                    if not self._in_contact and self.sim_time - self._last_contact > 1.0:
                        self.referee.crashes += 1
                        events.append('crash into {}'.format(o.label))
                    self._last_contact = self.sim_time
                    if self._safe_pose is not None:
                        self.vehicle.teleport(*self._safe_pose)
                    self.vehicle.stop()
                    break
            self._in_contact = touching
            if not touching:
                self._safe_pose = (s.x, s.y, s.yaw)

            events += self.referee.update(self.sim_time, (s.x, s.y))
            summary = self.referee.summary() if events else None
        for e in events:
            if e == 'lap':
                log.info('lap %d: %.3f s', len(summary['laps']), summary['laps'][-1])
            elif e == 'complete':
                log.info('mission %s complete: run time %.3f s, %d cones hit',
                         summary['mission'], summary['run_time'], summary['doo'])
            elif e.startswith('crash'):
                log.warning('%s - car stopped', e)
        return None
