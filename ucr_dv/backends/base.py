"""
Backend interface: what a simulator must provide to run a Stack.

Implemented: TimelessBackend (2D, lockstep, faster than real time) and
LiteBackend (fsds_lite 3D, lockstep). Planned: AssettoCorsaBackend
(real-time only) - see assetto_corsa.py and docs/ASSETTO_CORSA.md.

Lifecycle, driven by bench.runner.EpisodeRunner:
    b = Backend(); b.load(track, profile, vehicle, seed)
    obs = b.reset()
    loop: b.apply(cmd); obs = b.advance(dt); status = b.status()
    b.close()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.types import ControlCommand, Observation


@dataclass
class Capabilities:
    lockstep: bool = True          # advance() steps sim time exactly (else it waits on the wall clock)
    deterministic: bool = True     # same seed -> same run
    realtime_only: bool = False    # cannot run faster than real time
    max_speedup: float = 1e9
    raw_lidar: bool = False        # point clouds available
    raw_camera: bool = False       # rendered images available
    ground_truth: bool = True      # pose truth (metrics, GT odometry)
    cone_truth: bool = True        # true cone positions (perception metrics)
    teleport: bool = False
    has_3d_view: bool = False


@dataclass
class EpisodeStatus:
    t: float = 0.0
    laps: List[float] = field(default_factory=list)
    laps_required: int = 1
    doo: int = 0                   # cones down or out
    off_course: int = 0
    crashes: int = 0
    complete: bool = False         # referee: all laps done
    finished: bool = False         # the stack signalled finished
    stopped: bool = False          # car at rest after finishing
    dnf: Optional[str] = None      # reason, if the run is lost
    progress: float = 0.0          # 0..1 along the reference line (per lap)
    cte: float = 0.0               # cross-track error from the reference, m
    speed: float = 0.0
    run_time: Optional[float] = None
    extras: Dict = field(default_factory=dict)


class Backend(object):
    name = 'base'
    label = 'Backend'
    caps = Capabilities()

    def fidelity(self, spec) -> str:
        """'raw' or 'detection' for a rig sensor on this backend."""
        if spec.type == 'timeless_classic':
            return 'detection'
        want = spec.fidelity
        can_raw = (spec.type == 'lidar' and self.caps.raw_lidar) or \
                  (spec.type in ('camera', 'stereo') and self.caps.raw_camera)
        if want == 'raw':
            return 'raw' if can_raw else 'detection'
        if want == 'detection':
            return 'detection'
        return 'raw' if can_raw else 'detection'

    def load(self, track, profile, vehicle, seed=0):
        raise NotImplementedError

    def reset(self, seed=None) -> Observation:
        raise NotImplementedError

    def apply(self, cmd: ControlCommand):
        raise NotImplementedError

    def advance(self, dt: float) -> Observation:
        raise NotImplementedError

    def status(self) -> EpisodeStatus:
        raise NotImplementedError

    def car_state(self) -> Dict:
        """x, y, yaw, speed, steering (rad, +left), throttle, brake - for logs and views."""
        raise NotImplementedError

    def close(self):
        pass
