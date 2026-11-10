"""
One vehicle definition for every bench.

The file format is fsds_lite's VehicleParams JSON (data/vehicle/*.json). The
3D bench uses it as-is; the timeless 2D model and the controllers use the
summary quantities below (max drive force at a speed, brake force, drag,
wheelbase, steering limits), so both benches drive the same car.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import paths
from fsds_lite.vehicle import G, RPM_PER_RAD_S, VehicleParams


def load_vehicle(name='ucr_ev') -> VehicleParams:
    return VehicleParams.from_json(paths.vehicle_file(name))


@dataclass
class VehicleSpec:
    """The handful of numbers a controller needs, derived from VehicleParams."""
    params: VehicleParams

    @property
    def wheelbase(self):
        return self.params.wheelbase

    @property
    def lf(self):
        return self.params.lf

    @property
    def lr(self):
        return self.params.lr

    @property
    def max_steer(self):
        return math.radians(self.params.max_steer_deg)

    @property
    def steer_rate(self):
        return math.radians(self.params.steer_rate_deg_s)

    @property
    def mass(self):
        return self.params.mass

    @property
    def length(self):
        return self.params.length

    @property
    def width(self):
        return self.params.width

    @property
    def mu(self):
        return self.params.mu

    def max_drive_force(self, v):
        """Best-gear wheel force at road speed v (no slip), N."""
        p = self.params
        best = 0.0
        for g in range(1, len(p.gear_ratios) + 1):
            ratio = p.overall_ratio(g)
            rpm = max(v, 0.0) / p.wheel_radius * RPM_PER_RAD_S * ratio
            if rpm > p.max_rpm:
                continue
            best = max(best, p.torque_at(max(rpm, p.idle_rpm)) * ratio * p.drivetrain_efficiency / p.wheel_radius)
        # driven (rear) axle traction limit
        rear_load = p.mass * G * p.lf / p.wheelbase
        return min(best, p.mu * rear_load * 0.95)

    def max_brake_force(self):
        p = self.params
        return min(p.max_brake_force, p.mu * p.mass * G)

    def resistance(self, v):
        p = self.params
        f = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * v * v
        if v > 0.01:
            f += p.rolling_resistance * p.mass * G
        return f

    def accel_to_pedals(self, a_cmd, v):
        """Desired longitudinal acceleration -> (throttle, brake) via the pedal map."""
        need = self.mass * a_cmd + self.resistance(v)
        if need >= 0.0:
            fmax = max(self.max_drive_force(v), 1.0)
            return min(1.0, need / fmax), 0.0
        return 0.0, min(1.0, -need / max(self.max_brake_force(), 1.0))

    def top_speed(self):
        return self.params.top_speed()
