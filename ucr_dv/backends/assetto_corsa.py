"""
Assetto Corsa backend - PLANNED, not implemented. See docs/ASSETTO_CORSA.md.

The class exists so the interface is pinned down now: capabilities that the
GUI and runner already respect (real-time only, no raw sensors, cone truth
only through an overlaid cone map), and the coordinate conversion the
implementation will need.
"""
from __future__ import annotations

import math

from .base import Backend, Capabilities

NOT_YET = ('The Assetto Corsa bench is planned but not implemented yet - see docs/ASSETTO_CORSA.md '
           'for the readiness checklist.')


def ac_to_enu(x, y, z, heading=None):
    """AC world (x right, y up, z forward; left-handed) -> ENU (x, y, z up).

    AC's ground plane is (x, z); flipping one axis turns it right-handed.
    UNVERIFIED: check both the axis signs and the heading convention against
    two known poses in a real AC session before relying on this.
    Heading (AC, radians about +y) maps to ENU yaw about +z.
    """
    e = (z, -x, y)
    if heading is None:
        return e
    return e, math.atan2(math.sin(heading), math.cos(heading))


class AssettoCorsaBackend(Backend):
    name = 'assetto_corsa'
    label = 'Assetto Corsa (planned)'
    caps = Capabilities(lockstep=False, deterministic=False, realtime_only=True, max_speedup=1.0,
                        raw_lidar=False, raw_camera=False, ground_truth=True, cone_truth=True,
                        teleport=False, has_3d_view=False)

    def __init__(self, **kw):
        self.kw = kw

    def load(self, track, profile, vehicle, seed=0, stack=None):
        raise NotImplementedError(NOT_YET)

    def reset(self, seed=None):
        raise NotImplementedError(NOT_YET)

    def apply(self, cmd):
        raise NotImplementedError(NOT_YET)

    def advance(self, dt):
        raise NotImplementedError(NOT_YET)

    def status(self):
        raise NotImplementedError(NOT_YET)

    def car_state(self):
        raise NotImplementedError(NOT_YET)
