"""
Tracks for every bench, as fsds_lite `Track` objects.

Sources:
    builtin  'acceleration', 'skidpad', 'autocross', 'trackdrive' (FS rules, fsds_lite)
    CSV      FSDS custom-map format (data/tracks/*.csv)
    JSON     ucr-dv-timeless maps (data/tracks/timeless_*.json); moved so the
             car starts at the origin facing +x, like every other track

Every track also gets a *reference line* (`track_reference`) - the ordered
centreline the metrics use for cross-track error, progress and off-course
detection, and that the perception bench's ground-truth driver follows.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import paths  # noqa: F401  (puts fsds_lite on sys.path)
from fsds_lite.track import BUILTIN, TimingLine, Track, add_scenery
from fsds_lite.track import load_track as _lite_load_track

from .core.types import COLOR_IDS, UNKNOWN

TIMELESS_COLORS = {'yellow_cones': 'yellow', 'blue_cones': 'blue', 'orange_cones': 'orange',
                   'big_cones': 'big_orange'}


def list_tracks() -> List[str]:
    names = list(BUILTIN)
    for f in sorted(os.listdir(paths.TRACKS)):
        stem, ext = os.path.splitext(f)
        if ext in ('.csv', '.json'):
            names.append(stem)
    return names


def load_track(spec: str, scenery: bool = True) -> Track:
    """Builtin name, CSV/JSON path, or the stem of a file in data/tracks/."""
    if spec in BUILTIN:
        return _lite_load_track(spec, scenery=scenery)
    path = spec
    if not os.path.isfile(path):
        for ext in ('', '.json', '.csv'):
            cand = os.path.join(paths.TRACKS, spec + ext)
            if os.path.isfile(cand):
                path = cand
                break
        else:
            raise ValueError('unknown track {!r}; have {}'.format(spec, ', '.join(list_tracks())))
    if path.endswith('.json'):
        track = load_timeless_json(path)
        return add_scenery(track) if scenery else track
    return _lite_load_track(path, scenery=scenery)


def load_timeless_json(path: str) -> Track:
    with open(path) as f:
        m = json.load(f)
    x0, y0 = m['car_position']
    yaw0 = math.radians(float(m.get('car_heading', 0.0)))
    c, s = math.cos(-yaw0), math.sin(-yaw0)

    def tf(pts):
        p = np.asarray(pts, dtype=float).reshape(-1, 2) - (x0, y0)
        return np.column_stack([c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]])

    pos, col = [], []
    for key, tag in TIMELESS_COLORS.items():
        pts = m.get(key) or []
        if pts:
            pos.append(tf(pts))
            col += [tag] * len(pts)
    positions = np.vstack(pos) if pos else np.zeros((0, 2))

    def line(seg):
        a, b = tf(seg)
        # forward = the direction the car travels across it: towards +x at the start
        d = b - a
        n = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-9)
        if n[0] < 0:            # the car starts facing +x, so it crosses towards +x
            n = -n
        return TimingLine(tuple(a), tuple(b), (float(n[0]), float(n[1])))

    start = line(m['start_line'])
    finish = line(m['finish_line'])
    same = np.allclose(m['start_line'], m['finish_line'])
    centre = tf(m['center_line']) if m.get('center_line') else None
    name = os.path.splitext(os.path.basename(path))[0]
    track = Track(name, positions, col, start, start if same else finish,
                  mission='autocross' if same else 'acceleration', laps=1, centreline=centre)
    return track


def track_cones(track: Track) -> np.ndarray:
    """(N, 4) world cones [x, y, color, 1] in ucr_dv colour ids."""
    out = np.ones((len(track), 4))
    out[:, :2] = track.positions
    out[:, 2] = [COLOR_IDS.get(c, UNKNOWN) for c in track.colors]
    return out


# ---- reference line ---------------------------------------------------------

@dataclass
class Reference:
    xy: np.ndarray          # (N, 2) ordered from the start
    s: np.ndarray           # (N,) arc length
    closed: bool
    half_width: float       # typical half track width, m

    @property
    def length(self):
        return float(self.s[-1])

    def project(self, x, y, hint: Optional[int] = None, window: int = 200):
        """(index, signed cross-track error (+left), arc length) of the closest point.

        `hint` restricts the search to a window around a previous index, which
        keeps figure-of-eights (skidpad) and hairpins from jumping branches.
        """
        n = len(self.xy)
        if hint is None:
            idx = np.arange(n)
        else:
            idx = np.arange(hint - window // 4, hint + window) % n if self.closed else \
                np.arange(max(0, hint - window // 4), min(n, hint + window))
        d = np.hypot(self.xy[idx, 0] - x, self.xy[idx, 1] - y)
        k = int(idx[int(np.argmin(d))])
        j = (k + 1) % n if (self.closed or k + 1 < n) else k - 1
        t = self.xy[j] - self.xy[k] if j > k or self.closed else self.xy[k] - self.xy[j]
        tn = t / max(np.linalg.norm(t), 1e-9)
        rel = np.array([x, y]) - self.xy[k]
        cte = float(tn[0] * rel[1] - tn[1] * rel[0])
        return k, cte, float(self.s[k] + np.dot(rel, tn))


def _order_points(pts, start=(0.0, 0.0), heading=(1.0, 0.0), max_gap=8.0):
    """Greedy nearest-neighbour walk from `start`, preferring points ahead."""
    pts = np.asarray(pts, dtype=float)
    remaining = list(range(len(pts)))
    cur = np.asarray(start, dtype=float)
    direction = np.asarray(heading, dtype=float)
    out = []
    while remaining:
        cand = pts[remaining]
        rel = cand - cur
        d = np.hypot(rel[:, 0], rel[:, 1])
        ahead = rel @ direction
        score = d + np.where(ahead < 0, 3.0 * d, 0.0)
        k = int(np.argmin(score))
        if d[k] > max_gap and out:
            break
        nxt = cand[k]
        if d[k] > 1e-6:
            direction = (nxt - cur) / d[k]
        cur = nxt
        out.append(nxt)
        remaining.pop(k)
    return np.array(out)


def _resample(path, spacing=0.5, closed=False):
    pts = np.vstack([path, path[:1]]) if closed else path
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(s[-1] / spacing))
    t = np.linspace(0.0, s[-1], n, endpoint=not closed)
    return np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])


def _smooth(path, k=5, closed=False, passes=3):
    p = path.copy()
    for _ in range(passes):
        if closed:
            p = sum(np.roll(p, i, axis=0) for i in range(-k, k + 1)) / (2 * k + 1)
        else:
            pad = np.vstack([np.repeat(p[:1], k, 0), p, np.repeat(p[-1:], k, 0)])
            ker = np.ones(2 * k + 1) / (2 * k + 1)
            p = np.column_stack([np.convolve(pad[:, i], ker, mode='valid') for i in range(2)])
    return p


def track_reference(track: Track) -> Reference:
    # cached on the track itself (an id()-keyed cache breaks when ids are reused)
    cached = getattr(track, '_ucr_reference', None)
    if cached is not None:
        return cached
    closed = track.mission in ('autocross', 'trackdrive') or (
        track.start_line is not None and track.finish_line is track.start_line
        and track.mission != 'skidpad')
    if track.centreline is not None and len(track.centreline) > 3:
        c = np.asarray(track.centreline, dtype=float)
        if track.mission == 'skidpad':
            ref_xy, closed = _skidpad_reference(track), False
        else:
            ref_xy = _order_points(c, max_gap=6.0) if not closed else _roll_to_start(c)
            ref_xy = _resample(ref_xy, 0.5, closed)
    else:
        ref_xy = _reference_from_cones(track, closed)
    if closed and len(ref_xy) > 2:
        gap = np.hypot(*(ref_xy[-1] - ref_xy[0]))
        closed = gap < 6.0
    seg = np.hypot(*np.diff(np.vstack([ref_xy, ref_xy[:1]]) if closed else ref_xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])[:len(ref_xy)]
    blue = track.positions[[c == 'blue' for c in track.colors]]
    yellow = track.positions[[c == 'yellow' for c in track.colors]]
    if len(blue) and len(yellow):
        d = np.min(np.hypot(blue[:, None, 0] - yellow[None, :, 0], blue[:, None, 1] - yellow[None, :, 1]), axis=1)
        half = float(np.median(d)) / 2.0
    else:
        half = 1.5
    ref = Reference(ref_xy, s, closed, half)
    track._ucr_reference = ref
    return ref


def _roll_to_start(c):
    """Rotate a closed centreline so index 0 is nearest the spawn and it runs forward."""
    k = int(np.argmin(np.hypot(c[:, 0], c[:, 1])))
    c = np.roll(c, -k, axis=0)
    if len(c) > 2 and (c[min(5, len(c) - 1)] - c[0])[0] < 0:
        c = np.roll(c[::-1], 1, axis=0)
    return c


def _skidpad_reference(track):
    """Entry -> right circle x2 -> left circle x2 -> exit, the driven order."""
    right, left = track.skidpad_centres
    right, left = np.array(right), np.array(left)
    r = np.linalg.norm(right - (right + left) / 2.0)
    r_mid = float(np.mean([7.625, 10.625]))
    xc = right[0]
    entry = np.column_stack([np.linspace(-1.0, xc, 40), np.zeros(40)])
    # clockwise around the right centre starting at the crossing point (xc, 0)
    a0 = math.atan2(0.0 - right[1], 0.0)
    a = a0 - np.linspace(0, 4 * math.pi, 400, endpoint=False)
    rc = right + r_mid * np.column_stack([np.cos(a), np.sin(a)])
    a1 = math.atan2(0.0 - left[1], 0.0)
    a = a1 + np.linspace(0, 4 * math.pi, 400, endpoint=False)
    lc = left + r_mid * np.column_stack([np.cos(a), np.sin(a)])
    exit_ = np.column_stack([np.linspace(xc, xc + 26.0, 60), np.zeros(60)])
    del r
    return np.vstack([entry, rc, lc, exit_])


def _reference_from_cones(track, closed):
    pos = track.positions
    cols = np.array(track.colors)
    blue, yellow = pos[cols == 'blue'], pos[cols == 'yellow']
    if not len(blue) or not len(yellow):
        others = pos if len(pos) else np.array([[0.0, 0.0], [10.0, 0.0]])
        return _resample(_order_points(others), 0.5)
    mids = []
    for a, b in ((blue, yellow), (yellow, blue)):
        d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
        j = np.argmin(d, axis=1)
        ok = d[np.arange(len(a)), j] < 7.0
        mids.append((a[ok] + b[j[ok]]) / 2.0)
    mids = np.vstack(mids)
    ordered = _order_points(mids, max_gap=9.0)
    if len(ordered) < 3:
        return _resample(np.array([[0.0, 0.0], [10.0, 0.0]]), 0.5)
    path = _resample(ordered, 0.5, closed=False)
    path = _smooth(path, k=3, closed=closed)
    return _resample(path, 0.5, closed=closed)
