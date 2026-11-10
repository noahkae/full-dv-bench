"""
Speed profiles: path (car frame) -> target speed per point, and the
target the longitudinal controller chases now.

    grip_limited   three-pass profile (ucr-dv-timeless SpeedProfile, Stanford
                   style): v <= sqrt(a_lat / |k|), then a forward pass limited by
                   acceleration and a backward pass limited by braking, with the
                   path's end speed capped so the car can always stop inside
                   what it can see (unknown-track rule of thumb).
    constant       one speed everywhere (tuning controllers in isolation)
"""
from __future__ import annotations

import numpy as np

from ..core.registry import Component, P, register
from .pathutil import arc_length, curvature, smooth


@register('speed', 'grip_limited', label='Grip-limited forward/backward profile',
          params=[P('v_max', 12.0, 1.0, 35.0, 0.5, 'm/s', 'hard speed cap'),
                  P('v_min', 2.0, 0.0, 10.0, 0.5, 'm/s', 'never plan slower than this (while driving)'),
                  P('a_lat', 8.0, 1.0, 20.0, 0.5, 'm/s2', 'lateral acceleration limit (~ mu g)'),
                  P('a_accel', 4.0, 0.5, 15.0, 0.5, 'm/s2', 'acceleration limit'),
                  P('a_brake', 6.0, 0.5, 20.0, 0.5, 'm/s2', 'braking limit'),
                  P('end_speed', 4.0, 0.0, 20.0, 0.5, 'm/s', 'speed allowed at the end of the visible path'),
                  P('preview', 1.5, 0.0, 10.0, 0.25, 'm', 'distance ahead whose speed is the target'),
                  P('curvature_smoothing', 0.5, 0.0, 1.0, 0.05, '', 'smooth the path before measuring curvature'),
                  P('target_rise', 6.0, 0.5, 30.0, 0.5, 'm/s2', 'how fast the target may rise (rejects path flicker)')],
          refs=['ucr-dv-timeless SpeedProfile', 'KIT 2022 (g-g limits)'])
class GripLimited(Component):
    """sqrt(a_lat / k) cap, then acceleration and braking passes."""

    def __init__(self, params, **context):
        super(GripLimited, self).__init__(params, **context)
        self.reset()

    def reset(self):
        self._last = None
        self._t = 0.0

    def profile(self, path, v_now, ctx):
        p = self.params
        if len(path) < 3:
            return np.array([p['v_min']] * max(1, len(path))), p['v_min']
        sm = smooth(path, p['curvature_smoothing'], passes=10) if p['curvature_smoothing'] > 0 else path
        k = np.abs(curvature(sm))
        s = arc_length(path)
        ds = np.diff(s)
        v = np.minimum(np.sqrt(p['a_lat'] / np.maximum(k, 1e-4)), p['v_max'])
        v[-1] = min(v[-1], max(p['end_speed'], 0.0))
        cap = ctx.get('speed_cap')
        if cap is not None:
            v = np.minimum(v, cap)
        fwd = v.copy()
        fwd[0] = min(v[0], max(v_now, p['v_min']))
        for i in range(1, len(v)):
            fwd[i] = min(v[i], np.sqrt(fwd[i - 1] ** 2 + 2 * p['a_accel'] * ds[i - 1]))
        bwd = fwd.copy()
        for i in range(len(v) - 2, -1, -1):
            bwd[i] = min(fwd[i], np.sqrt(bwd[i + 1] ** 2 + 2 * p['a_brake'] * ds[i]))
        target = float(np.interp(p['preview'] + v_now * 0.1, s, bwd))
        floor = p['v_min'] if cap is None else min(p['v_min'], cap)
        target = max(target, floor)
        # a short path for one tick shouldn't slam the brakes and then the throttle:
        # the target may fall immediately but only rise at `target_rise`
        dt = ctx.get('dt', 0.05)
        if self._last is not None and target > self._last:
            target = min(target, self._last + p['target_rise'] * dt)
        self._last = target
        return bwd, target


@register('speed', 'constant', label='Constant speed',
          params=[P('speed', 5.0, 0.0, 30.0, 0.5, 'm/s', 'target speed')])
class ConstantSpeed(Component):
    """Same target everywhere - for tuning steering on its own."""

    def profile(self, path, v_now, ctx):
        v = self.params['speed']
        cap = ctx.get('speed_cap')
        if cap is not None:
            v = min(v, cap)
        return np.full(max(1, len(path)), v), v
