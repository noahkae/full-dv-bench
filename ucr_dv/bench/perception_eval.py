"""
Perception bench: how good is the rig + perception, independent of driving?

A ground-truth driver follows the track's reference line at a fixed speed
(so planning/control mistakes can't spoil the measurement) while the
profile's perception runs on every frame. Every sensor and the fused output
are scored against the true cones (perception/metrics.py): precision,
recall, recall vs range, position RMSE/bias, colour accuracy, compute time.

Works on both benches: timeless (detection-level models) and 3D (raw point
clouds and rendered images through the real detectors).
"""
from __future__ import annotations

import math

import numpy as np

from ..core.types import ControlCommand, empty_cones, to_car
from ..perception.metrics import PerceptionStats
from ..planning.pathutil import point_at
from ..tracks import track_reference
from .session import prepare


def _gt_driver(ref, truth, speed, lookahead=4.0, wheelbase=1.53, max_steer=0.47, hint=None):
    """Pure pursuit on the reference line using the TRUE pose."""
    pose = truth.pose
    k, cte, s = ref.project(pose.x, pose.y, hint)
    n = len(ref.xy)
    idx = [(k + i) % n if ref.closed else min(n - 1, k + i) for i in range(0, 40)]
    local = to_car(ref.xy[idx], pose)
    tx, ty = point_at(local, lookahead)
    alpha = math.atan2(ty, tx)
    steer = math.atan2(2.0 * wheelbase * math.sin(alpha), max(math.hypot(tx, ty), 0.5))
    steer = max(-max_steer, min(max_steer, steer))
    a = 1.5 * (speed - truth.speed)
    return steer, a, k


def evaluate(profile, track_name, bench='timeless', speed=6.0, distance=None, seed=0, eval_range=30.0,
             progress=None, stop_flag=None, keep_frames=40):
    """Drive one lap (or `distance` m) on the reference; returns a report dict."""
    runner, track, stack, b = prepare(profile, track_name, bench, seed)
    ref = track_reference(track)
    distance = distance or min(ref.length, 400.0)
    dt = 1.0 / max(1.0, profile.pipeline_hz)
    obs = b.reset(seed)
    stack.reset()
    stats = {}
    timings = {}
    frames = []
    travelled, hint = 0.0, None
    veh = stack.vehicle
    steps = 0
    try:
        while travelled < distance:
            if stop_flag is not None and stop_flag():
                break
            out = stack.step(obs, dt)
            truth = obs.truth
            gt = truth.cones_world[~truth.cones_hit].copy()
            gt[:, :2] = to_car(gt[:, :2], truth.pose)
            for name, cones in list(out.cones_by_sensor.items()) + [('fused', out.cones)]:
                stats.setdefault(name, PerceptionStats()).add(cones, gt, eval_range=eval_range)
            for k, v in out.timings_ms.items():
                if k.startswith('perception:') or k == 'fusion':
                    timings.setdefault(k.replace('perception:', ''), []).append(v)
            steps += 1
            if keep_frames and steps % max(1, int(1.0 / dt)) == 0 and len(frames) < keep_frames:
                frames.append({'t': obs.t, 'gt': gt, 'fused': out.cones.copy(),
                               'by_sensor': {k: v.copy() for k, v in out.cones_by_sensor.items()},
                               'debug': {k: v for k, v in out.debug.items() if k.startswith('perception:')}})
            steer, a, hint = _gt_driver(ref, truth, speed, wheelbase=veh.wheelbase, max_steer=veh.max_steer,
                                        hint=hint)
            thr, brk = veh.accel_to_pedals(a, truth.speed)
            b.apply(ControlCommand(steering=steer, throttle=thr, brake=brk, target_speed=speed))
            obs = b.advance(dt)
            travelled += obs.truth.speed * dt
            if progress and steps % 20 == 0:
                progress(min(1.0, travelled / distance))
    finally:
        if bench != 'timeless':
            b.close()
    report = {'track': track_name, 'bench': bench, 'distance': round(travelled, 1), 'frames': steps,
              'sensors': {k: v.summary() for k, v in stats.items()},
              'timings_ms': {k: {'mean': round(float(np.mean(v)), 3), 'p95': round(float(np.percentile(v, 95)), 3)}
                             for k, v in timings.items()},
              'fidelity': getattr(b, 'realised', {s.name: 'detection' for s in profile.rig.perception()})}
    return report, frames, stats
