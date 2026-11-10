"""
Mission logic: which event is running, lap counting, when to stop.

The car does not get the referee's lap count - like the real AS it has to
work it out. `auto` counts timing-line crossings from the big orange cones
(each pair marks a start/finish line), predicting the crossing point with
odometry so it works after the cones leave the FOV (ported idea from
ucr-dv-timeless LapCounter). After the event's laps it brakes to a stop and
reports `finished` (the AS 'finished' signal).

Events (FS rules): acceleration 1 run (start + finish line), skidpad 4 laps
on a known layout, autocross 1 lap, trackdrive 10 laps.

The mission also provides the known-layout templates (world frame) for the
static events, for the `track_template` planner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.registry import Component, P, register
from ..core.types import BIG_ORANGE

EVENT_CROSSINGS = {
    # crossings of a big-orange line before the event is over
    'acceleration': 2,      # start line, finish line
    'autocross': 2,         # start, then the same line after one lap
    'trackdrive': 11,       # start + 10 laps
    'skidpad': 5,           # entry + 4 laps through the centre line
}


@dataclass
class MissionState:
    event: str = 'autocross'
    crossings: int = 0
    laps: int = 0
    stopping: bool = False
    finished: bool = False
    speed_cap: float = None
    next_line_at: float = None      # odometry distance of the predicted next crossing


def skidpad_template(spacing=0.5):
    """Rules layout (D 4.1): entry, right circle x2 (clockwise), left x2, exit."""
    xc = 15.0 + 0.9
    half_gap = 18.25 / 2.0
    r = (15.25 / 2.0 + 21.25 / 2.0) / 2.0
    pts = [np.column_stack([np.arange(0.0, xc, spacing), np.zeros(int(math.ceil(xc / spacing)))])]
    n = int(2 * math.pi * r / spacing)
    for centre, sign in (((xc, -half_gap), -1.0), ((xc, half_gap), 1.0)):
        a0 = math.atan2(0.0 - centre[1], 0.0)
        a = a0 + sign * np.linspace(0.0, 4 * math.pi, 2 * n, endpoint=False)
        pts.append(np.column_stack([centre[0] + r * np.cos(a), centre[1] + r * np.sin(a)]))
    pts.append(np.column_stack([np.arange(xc, xc + 25.0, spacing), np.zeros(int(math.ceil(25.0 / spacing)))]))
    return np.vstack(pts)


def acceleration_template(spacing=0.5):
    return np.column_stack([np.arange(0.0, 160.0, spacing), np.zeros(int(160.0 / spacing))])


@register('mission', 'auto', label='Auto (FS event rules)',
          params=[P('min_lap_distance', 30.0, 5.0, 200.0, 5.0, 'm', 'ignore line crossings closer than this'),
                  P('line_window', 12.0, 3.0, 30.0, 1.0, 'm', 'look for big orange cones this far ahead'),
                  P('run_out', 10.0, 0.0, 30.0, 1.0, 'm', 'keep driving this far past the last line before braking'),
                  P('accel_run_distance', 75.0, 50.0, 100.0, 1.0, 'm', 'acceleration: start to finish'),
                  P('use_templates', True, help='provide known layouts for skidpad / acceleration')],
          refs=['FS Rules D4-D8', 'ucr-dv-timeless LapCounter'])
class AutoMission(Component):
    """Counts big-orange line crossings; stops after the event's laps."""

    def __init__(self, params, **context):
        super(AutoMission, self).__init__(params, **context)
        self.event = context.get('event', 'autocross')
        self.reset()

    def reset(self):
        self.state = MissionState(event=self.event)
        self._last_cross_dist = -1e9
        self._stop_dist = None
        self._stopped_since = None
        self._line_frames = 0

    def template(self):
        if not self.params['use_templates']:
            return None
        if self.event == 'skidpad':
            return skidpad_template()
        if self.event == 'acceleration':
            return acceleration_template()
        return None

    def update(self, cones, distance, speed, t):
        """cones: car frame (fused or local map); distance: odometry metres."""
        st = self.state
        p = self.params
        needed = EVENT_CROSSINGS.get(self.event, 2)
        big = cones[(cones[:, 2] == BIG_ORANGE) & (cones[:, 0] > 0.3) & (cones[:, 0] < p['line_window'])
                    & (np.abs(cones[:, 1]) < 4.0)] if len(cones) else cones
        # a timing line = big orange on BOTH sides, seen in consecutive frames
        both = len(big) >= 2 and (big[:, 1] > 0.3).any() and (big[:, 1] < -0.3).any()
        self._line_frames = self._line_frames + 1 if both else 0
        if self._line_frames >= 2 and distance - self._last_cross_dist > p['min_lap_distance']:
            st.next_line_at = distance + float(np.mean(big[:, 0]))
        if st.next_line_at is not None and distance >= st.next_line_at:
            st.crossings += 1
            st.laps = max(0, st.crossings - 1)
            self._last_cross_dist = distance
            st.next_line_at = None
        if self.event == 'acceleration' and st.crossings >= 1 and self._stop_dist is None:
            # the finish line is 75 m on; braking starts there even if it is missed
            self._stop_dist = self._last_cross_dist + p['accel_run_distance'] + p['run_out']
        if st.crossings >= needed and self._stop_dist is None:
            self._stop_dist = distance + p['run_out']
        if self._stop_dist is not None and distance >= self._stop_dist:
            st.stopping = True
        if st.stopping:
            st.speed_cap = 0.0
            if speed < 0.2:
                if self._stopped_since is None:
                    self._stopped_since = t
                elif t - self._stopped_since > 0.5:
                    st.finished = True
        return st
