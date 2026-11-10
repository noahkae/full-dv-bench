"""
Longitudinal control: target speed -> (throttle, brake).

    pid_speed   PID on speed error giving a desired acceleration, plus a
                feed-forward of the planned acceleration and the car's
                drag/rolling resistance; then the vehicle's pedal map
                (max drive force at this speed, max brake force) turns the
                acceleration into pedals. Anti-windup by conditional
                integration (as in ucr-dv-timeless PID).
    bang_bang   full throttle below target, brake above (for comparison)
"""
from __future__ import annotations

from ..core.registry import Component, P, register


@register('longitudinal', 'pid_speed', label='PID speed + pedal map',
          params=[P('kp', 1.2, 0.0, 10.0, 0.05, '1/s', 'acceleration per m/s of error'),
                  P('ki', 0.3, 0.0, 5.0, 0.05, '1/s2', 'integral gain'),
                  P('kd', 0.0, 0.0, 2.0, 0.01, '', 'derivative gain'),
                  P('a_max', 6.0, 0.5, 20.0, 0.5, 'm/s2', 'acceleration request limit'),
                  P('a_min', -8.0, -20.0, -0.5, 0.5, 'm/s2', 'deceleration request limit'),
                  P('deadband', 0.1, 0.0, 1.0, 0.05, 'm/s', 'ignore smaller errors')],
          refs=['KIT 2022 (PI speed)', 'ucr-dv-timeless PID'])
class PidSpeed(Component):
    """Speed PID -> desired acceleration -> pedals through the vehicle model."""

    def __init__(self, params, **context):
        super(PidSpeed, self).__init__(params, **context)
        self.vehicle = context.get('vehicle')
        self.reset()

    def reset(self):
        self.i = 0.0
        self.prev_e = None

    def command(self, v_target, v, dt, a_ff=0.0):
        p = self.params
        e = v_target - v
        if abs(e) < p['deadband']:
            e = 0.0
        d = 0.0 if self.prev_e is None or dt <= 0 else (e - self.prev_e) / dt
        self.prev_e = e
        a = p['kp'] * e + p['ki'] * self.i + p['kd'] * d + a_ff
        if p['a_min'] < a < p['a_max']:
            self.i += e * dt                            # integrate only when not saturated
        a = max(p['a_min'], min(p['a_max'], a))
        if v_target <= 0.05 and v < 0.5:
            return 0.0, 1.0, a                          # hold the car at standstill
        if self.vehicle is None:
            return (min(1.0, a / 5.0), 0.0, a) if a >= 0 else (0.0, min(1.0, -a / 8.0), a)
        thr, brk = self.vehicle.accel_to_pedals(a, v)
        return thr, brk, a


@register('longitudinal', 'bang_bang', label='Bang-bang',
          params=[P('hysteresis', 0.3, 0.0, 3.0, 0.05, 'm/s', 'switching band'),
                  P('throttle', 0.6, 0.05, 1.0, 0.05, '', 'throttle when too slow'),
                  P('brake', 0.4, 0.05, 1.0, 0.05, '', 'brake when too fast')])
class BangBang(Component):
    """Full-on/full-off - shows why a PID is worth having."""

    def command(self, v_target, v, dt, a_ff=0.0):
        p = self.params
        if v < v_target - p['hysteresis']:
            return p['throttle'], 0.0, 1.0
        if v > v_target + p['hysteresis'] or v_target <= 0.05:
            return 0.0, p['brake'], -1.0
        return 0.0, 0.0, 0.0
