"""
EpisodeRunner: drives one Stack on one Backend for one run.

Qt-free and thread-agnostic. The GUI runs it on a worker thread, posts
commands (pause / step / stop / live params / speed) through `post()`, and
reads `latest` (an atomically swapped snapshot) on a timer - the worker
never touches Qt.

Termination (whichever comes first):
    finished   the stack said finished and the car has stopped
    timeout    per-event time limit (generous; DNF)
    lost       the car is far off the track (DNF)
    stalled    no progress for `stall_s` while not finished (DNF)
    error      the stack raised (DNF, traceback kept)
    stopped    the user pressed stop
"""
from __future__ import annotations

import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..core.types import ControlCommand, StackOutput
from .scoring import score

TIMEOUTS = {'acceleration': 40.0, 'skidpad': 120.0, 'autocross': 150.0, 'trackdrive': 900.0}


@dataclass
class RunConfig:
    track: str = 'autocross'
    bench: str = 'timeless'
    seed: int = 0
    timeout: Optional[float] = None
    realtime_factor: float = 0.0       # 0 = as fast as possible; 1 = real time
    stall_s: float = 15.0
    record_every: int = 1              # log every n ticks
    keep_frames: bool = False          # keep per-tick detections in the log (perception analysis)


@dataclass
class Frame:
    """What a view needs to draw one moment of a run (car frame data + world pose)."""
    t: float
    car: Dict
    out: StackOutput
    status: object
    tick: int
    frames: Dict = field(default_factory=dict)      # this tick's raw sensor frames (references)


class RunLog(object):
    COLS = ('t', 'x', 'y', 'yaw', 'speed', 'target_speed', 'steer_cmd', 'steer', 'throttle', 'brake',
            'est_x', 'est_y', 'est_yaw', 'cte', 'n_cones', 'compute_ms', 'progress', 'doo', 'ax', 'ay')

    def __init__(self):
        self.rows = []
        self.detections = []    # optional per-tick (t, cones car frame, truth pose)
        self.events = []

    def add(self, row):
        self.rows.append(row)

    def arrays(self) -> Dict[str, np.ndarray]:
        if not self.rows:
            return {c: np.zeros(0) for c in self.COLS}
        a = np.array(self.rows, dtype=float)
        return {c: a[:, i] for i, c in enumerate(self.COLS)}


class EpisodeRunner(object):
    def __init__(self, backend, stack, config: RunConfig, track=None, on_event=None):
        self.backend = backend
        self.stack = stack
        self.cfg = config
        self.track = track
        self.log = RunLog()
        self.latest: Optional[Frame] = None
        self.result = None
        self.on_event = on_event
        self._q = queue.Queue()
        self._paused = False
        self._step_once = False
        self._stop = False
        self.running = False
        self.error = None
        self.dt = 1.0 / max(1.0, stack.profile.pipeline_hz)
        self.thread = None

    # ---- control from other threads ----------------------------------------
    def post(self, kind, **kw):
        self._q.put((kind, kw))

    def stop(self):
        self.post('stop')

    def _drain(self):
        while True:
            try:
                kind, kw = self._q.get_nowait()
            except queue.Empty:
                return
            if kind == 'stop':
                self._stop = True
            elif kind == 'pause':
                self._paused = kw.get('value', not self._paused)
            elif kind == 'step':
                self._step_once = True
            elif kind == 'params':
                self.stack.set_params(kw['stage'], kw['params'])
                self._event('params', '{} {}'.format(kw['stage'], kw['params']))
            elif kind == 'speed':
                self.cfg.realtime_factor = kw['value']

    def _event(self, kind, text):
        t = self.latest.t if self.latest else 0.0
        self.log.events.append((t, kind, text))
        if self.on_event:
            try:
                self.on_event(kind, text)
            except Exception:
                pass

    # ---- run ---------------------------------------------------------------
    def start_thread(self):
        self.thread = threading.Thread(target=self.run, name='episode', daemon=True)
        self.thread.start()
        return self

    def run(self):
        cfg = self.cfg
        b, stack = self.backend, self.stack
        self.running = True
        event = getattr(self.track, 'mission', 'autocross')
        timeout = cfg.timeout or TIMEOUTS.get(event, 200.0)
        reason = None
        tick = 0
        wall0 = time.monotonic()
        best_progress, best_t = 0.0, 0.0
        laps_seen = 0
        try:
            obs = b.reset(cfg.seed)
            stack.reset()
            while True:
                self._drain()
                if self._stop:
                    reason = 'stopped'
                    break
                if self._paused and not self._step_once:
                    time.sleep(0.02)
                    continue
                self._step_once = False
                out = stack.step(obs, self.dt)
                b.apply(out.command)
                obs = b.advance(self.dt)
                st = b.status()
                car = b.car_state()
                tick += 1
                if tick % cfg.record_every == 0:
                    self.log.add((obs.t, car['x'], car['y'], car['yaw'], car['speed'], out.command.target_speed,
                                  out.command.steering, car['steering'], out.command.throttle, out.command.brake,
                                  out.pose.x, out.pose.y, out.pose.yaw, st.cte, len(out.cones),
                                  out.timings_ms.get('total', 0.0), st.progress, st.doo,
                                  car.get('ax', 0.0), car.get('ay', 0.0)))
                    if cfg.keep_frames and obs.truth is not None:
                        self.log.detections.append((obs.t, out.cones.copy(), obs.truth.pose))
                self.latest = Frame(obs.t, car, out, st, tick, dict(obs.frames))
                if len(st.laps) > laps_seen:
                    laps_seen = len(st.laps)
                    self._event('lap', 'lap {} {:.2f} s'.format(laps_seen, st.laps[-1]))
                # termination
                if st.finished and st.stopped:
                    reason = 'finished'
                    break
                if st.dnf:
                    reason = 'lost'
                    break
                if obs.t > timeout:
                    reason = 'timeout'
                    break
                prog = st.progress + len(st.laps)
                if prog > best_progress + 0.002 or st.speed > 0.5:
                    best_progress, best_t = max(best_progress, prog), obs.t
                elif obs.t - best_t > cfg.stall_s and not st.finished:
                    reason = 'stalled'
                    break
                if cfg.realtime_factor > 0:
                    target = wall0 + obs.t / cfg.realtime_factor
                    lag = target - time.monotonic()
                    if lag > 0:
                        time.sleep(lag)
                    elif lag < -0.5:
                        wall0 -= lag            # fell behind (slow PC): don't try to catch up
        except Exception:
            self.error = traceback.format_exc()
            reason = 'error'
        self.running = False
        st = b.status()
        self.result = score(st, reason, event, uses_gt=stack.uses_ground_truth, log=self.log,
                            stack_errors=stack.errors, error=self.error)
        self.result['wall_s'] = round(time.monotonic() - wall0, 3)
        self.result['sim_s'] = round(st.t, 3)
        self.result['speedup'] = round(st.t / max(self.result['wall_s'], 1e-6), 1)
        self._event('end', '{}: {}'.format(reason, self.result.get('summary', '')))
        return self.result
