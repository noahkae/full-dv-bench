"""
Tracks: cones, timing lines and trackside objects, in the ENU frame relative
to the car's spawn point (x forward at spawn, y left).

Builtin events follow the Formula Student Rules 2026 (FSG) and the FS
Driverless Specification 2026 (DS):

    acceleration   D 5.1 / D 5.2 / DS Fig. 2
    skidpad        D 4.1 / D 4.2 / DS Fig. 3
    autocross      D 6.1 / D 6.2 (one lap of a closed handling course)
    trackdrive     D 8.1 (same circuit as autocross, 10 laps)

Marking (DS 1.1): blue cones on the left, yellow on the right, small orange
for entry/exit lanes, big orange before and after start/finish lines, at
most 5 m apart and closer in corners.

Also reads the FSDS custom-map CSV format unchanged
(tag,x,y,direction,x_variance,y_variance,xy_covariance). For CSV tracks the
start/finish line comes from the big orange cones, as in FSDS.

Trackside objects (`Obstacle`) give the LiDAR something to see besides
cones: concrete barriers, tyre stacks, light poles, timekeeping equipment
and stacked spare cones. DS 1.4 lists the last two explicitly as things a
perception stack must not mistake for track cones.
"""
import csv
import math
import os
from dataclasses import dataclass

import numpy as np

# Cone shape as a right circular cone: (base radius, height) in metres.
# DS Table 1: small cones 228 x 228 x 325 mm, big orange 285 x 285 x 505 mm.
CONE_SHAPES = {
    'blue': (0.114, 0.325),
    'yellow': (0.114, 0.325),
    'orange': (0.114, 0.325),
    'big_orange': (0.1425, 0.505),
    'unknown': (0.114, 0.325),
}

TRACKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'tracks')
TRACK_WIDTH = 3.0          # D 4.1.5 / D 5.1.1 / D 6.1.1 minimum
MAX_CONE_GAP = 5.0         # DS 1.1
MIN_RADIUS = 5.5           # centreline; D 6.1.1 asks for >= 9 m turning diameter


@dataclass
class TimingLine:
    """A line the car's centre must cross in the `forward` direction."""
    a: tuple
    b: tuple
    forward: tuple             # unit vector: the direction of travel across the line

    def crossed(self, p0, p1):
        """True if the segment p0 -> p1 crosses this line going forward."""
        ax, ay = self.a
        bx, by = self.b
        d1 = (bx - ax) * (p0[1] - ay) - (by - ay) * (p0[0] - ax)
        d2 = (bx - ax) * (p1[1] - ay) - (by - ay) * (p1[0] - ax)
        if d1 * d2 > 0 or d1 == d2:
            return False
        t = d1 / (d1 - d2)
        x, y = p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])
        seg = (bx - ax) * (x - ax) + (by - ay) * (y - ay)
        if not 0.0 <= seg <= (bx - ax) ** 2 + (by - ay) ** 2:
            return False
        return (p1[0] - p0[0]) * self.forward[0] + (p1[1] - p0[1]) * self.forward[1] > 0


@dataclass
class Obstacle:
    kind: str                  # 'box' | 'cylinder' | 'cone'
    x: float
    y: float
    height: float
    yaw: float = 0.0           # box only
    half_x: float = 0.0        # box only: half length along yaw
    half_y: float = 0.0        # box only: half width
    radius: float = 0.0        # cylinder / cone base radius
    label: str = ''            # barrier | tyres | pole | timekeeping | spare_cones
    color: str = ''            # cone obstacles: cone colour

    @property
    def bound_radius(self):
        return math.hypot(self.half_x, self.half_y) if self.kind == 'box' else self.radius


class Track(object):
    def __init__(self, name, positions, colors, start_line=None, finish_line=None,
                 mission='trackdrive', laps=1, obstacles=None, centreline=None):
        self.name = name
        self.positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        self.colors = list(colors)
        shapes = np.array([CONE_SHAPES.get(c, CONE_SHAPES['unknown']) for c in self.colors],
                          dtype=np.float64).reshape(-1, 2)
        self.radii = shapes[:, 0]
        self.heights = shapes[:, 1]
        self.start_line = start_line
        self.finish_line = finish_line or start_line
        self.mission = mission
        self.laps = laps                    # timed laps that complete the mission
        self.centreline = centreline        # (N, 2) or None
        self.obstacles = list(obstacles or [])
        self._pack_obstacles()

    def __len__(self):
        return len(self.colors)

    def _pack_obstacles(self):
        """Numpy arrays per obstacle kind, for the LiDAR and collision checks."""
        obs = self.obstacles
        self.boxes = np.array([(o.x, o.y, o.yaw, o.half_x, o.half_y, o.height)
                               for o in obs if o.kind == 'box'], dtype=np.float64).reshape(-1, 6)
        self.cylinders = np.array([(o.x, o.y, o.radius, o.height)
                                   for o in obs if o.kind == 'cylinder'], dtype=np.float64).reshape(-1, 4)
        self.cone_obstacles = [o for o in obs if o.kind == 'cone']
        self.cone_obs = np.array([(o.x, o.y, o.radius, o.height) for o in self.cone_obstacles],
                                 dtype=np.float64).reshape(-1, 4)

    def set_obstacles(self, obstacles):
        self.obstacles = list(obstacles)
        self._pack_obstacles()

    @classmethod
    def from_csv(cls, path):
        positions, colors = [], []
        with open(path) as f:
            for row in csv.reader(f):
                if not row or row[0].strip().startswith('#') or row[0].strip() == 'tag':
                    continue
                colors.append(row[0].strip())
                positions.append((float(row[1]), float(row[2])))
        track = cls(os.path.splitext(os.path.basename(path))[0], positions, colors)
        track.start_line = track.finish_line = _line_from_big_orange(track)
        return track


def _line_from_big_orange(track):
    """FSDS convention: 2 or 4 big orange cones mark the start/finish line."""
    big = track.positions[[c == 'big_orange' for c in track.colors]]
    if len(big) not in (2, 4):
        return None
    if len(big) == 4:
        # pair each cone with its nearest neighbour (same side of the track)
        d = np.linalg.norm(big[:, None] - big[None], axis=2) + np.eye(4) * 1e9
        partner = d.argmin(axis=1)
        sides = sorted({tuple(sorted((i, int(partner[i])))) for i in range(4)})
        if len(sides) != 2:
            return None
        a, b = (big[list(s)].mean(axis=0) for s in sides)
    else:
        a, b = big
    across = b - a
    forward = np.array([across[1], -across[0]]) / (np.linalg.norm(across) or 1.0)
    if forward[0] < 0:                 # the car spawns facing +x
        forward = -forward
    return TimingLine(tuple(a), tuple(b), tuple(forward))


# ---- builtin events ---------------------------------------------------------

def _cone_row(start, end, spacing):
    """Points from start to end (inclusive) at most `spacing` apart."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    n = max(1, int(math.ceil(np.linalg.norm(end - start) / spacing)))
    return [tuple(start + (end - start) * k / n) for k in range(n + 1)]


def acceleration_track(car_length=1.8):
    """D 5.1: 75 m straight, 3 m wide, cones every ~5 m; D 5.2.3: car staged
    0.30 m behind the start line; D 5.2.4: stop within 75 m after the finish."""
    half = TRACK_WIDTH / 2.0
    x0 = 0.30 + car_length / 2.0                     # start line, car centre at the origin
    x1 = x0 + 75.0                                   # finish line
    pos, col = [], []

    def add(points, tag):
        for p in points:
            pos.append(p)
            col.append(tag)
    for side, tag in ((half, 'blue'), (-half, 'yellow')):
        add(_cone_row((x0 + 5.0, side), (x1 - 5.0, side), 5.0), tag)        # timed section
        add([(x0 - 5.0, side), (x0 - 10.0, side)], tag)                     # start position
        add([(x - 0.5, side) for x in (x0, x1)] + [(x + 0.5, side) for x in (x0, x1)],
            'big_orange')                                                    # DS 1.1
        add(_cone_row((x1 + 5.0, side), (x1 + 75.0, side), 5.0), 'orange')  # stop area
    add(_cone_row((x1 + 75.0, -half), (x1 + 75.0, half), 1.0)[1:-1], 'orange')  # end of stop area
    start = TimingLine((x0, -half), (x0, half), (1.0, 0.0))
    finish = TimingLine((x1, -half), (x1, half), (1.0, 0.0))
    centre = np.array([(x, 0.0) for x in np.arange(-12.0, x1 + 76.0, 1.0)])
    return Track('acceleration', pos, col, start, finish, mission='acceleration', laps=1,
                 centreline=centre)


def skidpad_track(car_length=1.8):
    """D 4.1: two concentric circle pairs in a figure of eight, centres 18.25 m
    apart, inner diameter 15.25 m, outer 21.25 m, 17 cones inside each inner
    circle and 13 outside each outer; start/finish on the line between the
    centres. D 4.2.4: car staged 15 m before that line; D 4.2.5: stop within
    25 m after it. Run: right circle twice (2nd timed), left twice (4th timed)."""
    r_in, r_out, half_gap = 15.25 / 2.0, 21.25 / 2.0, 18.25 / 2.0
    xc = 15.0 + car_length / 2.0                     # start/finish line
    right, left = np.array([xc, -half_gap]), np.array([xc, half_gap])
    pos, col = [], []

    def ring(centre, radius, n, phase):
        a = phase + 2.0 * math.pi * np.arange(n) / n
        return list(map(tuple, centre + radius * np.column_stack([np.cos(a), np.sin(a)])))

    # inner circles: the car circles the right one clockwise, so its inner
    # cones are on the car's right (yellow); the left one anticlockwise (blue)
    for centre, tag in ((right, 'yellow'), (left, 'blue')):
        for p in ring(centre, r_in, 17, math.pi / 17):   # no cone right at the lane
            pos.append(p)
            col.append(tag)
    # outer circles: 13 cones each on the arc that isn't inside the other circle
    for centre, other, tag in ((right, left, 'blue'), (left, right, 'yellow')):
        a = np.linspace(0.0, 2.0 * math.pi, 3600, endpoint=False)
        pts = centre + r_out * np.column_stack([np.cos(a), np.sin(a)])
        keep = pts[np.linalg.norm(pts - other, axis=1) > r_out + 0.5]
        # the kept arc is contiguous; spread 13 cones evenly along it
        ang = np.unwrap(np.sort(np.arctan2(keep[:, 1] - centre[1], keep[:, 0] - centre[0]) % (2 * math.pi)))
        gap = np.argmax(np.diff(np.concatenate([ang, ang[:1] + 2 * math.pi])))
        a0 = ang[(gap + 1) % len(ang)]
        a1 = ang[gap] + (2 * math.pi if gap + 1 < len(ang) else 0.0)
        for t in np.linspace(a0, a1, 13):
            pos.append(tuple(centre + r_out * np.array([math.cos(t), math.sin(t)])))
            col.append(tag)
    # big orange before and after the start/finish line (DS 1.1)
    for dx in (-1.0, 1.0):
        for y in (-1.45, 1.45):
            pos.append((xc + dx, y))
            col.append('big_orange')
    # entry and exit lanes (small orange), up to where the circles begin
    lane_end = math.sqrt((r_out + 0.5) ** 2 - (half_gap - TRACK_WIDTH / 2.0) ** 2)
    for y in (-TRACK_WIDTH / 2.0, TRACK_WIDTH / 2.0):
        for p in _cone_row((-2.0, y), (xc - lane_end, y), 5.0) + _cone_row((xc + lane_end, y), (xc + 26.0, y), 5.0):
            pos.append(p)
            col.append('orange')
    for p in _cone_row((xc + 26.0, -1.5), (xc + 26.0, 1.5), 1.0)[1:-1]:
        pos.append(p)
        col.append('orange')                         # end of the stop area
    line = TimingLine(tuple(right), tuple(left), (1.0, 0.0))
    # centreline: entry lane, both driving circles, exit lane (keeps scenery clear)
    r_mid = (r_in + r_out) / 2.0
    loop = np.linspace(0.0, 2.0 * math.pi, 120, endpoint=False)
    circle = r_mid * np.column_stack([np.cos(loop), np.sin(loop)])
    pts = [np.column_stack([np.linspace(-5.0, xc + 26.0, 40), np.zeros(40)]),
           right + circle, left + circle]
    track = Track('skidpad', pos, col, line, line, mission='skidpad', laps=4,
                  centreline=np.vstack(pts))
    track.skidpad_centres = (tuple(right), tuple(left))
    return track


# Autocross / trackdrive circuit: a closed Catmull-Rom spline through these
# control points (metres), driven anticlockwise. It contains every D 6.1.1
# feature: a ~70 m main straight (<= 80 m), a slalom (cone gates 10 m apart,
# 7.5-12 m), a hairpin, a chicane, and a decreasing-radius turn. The lap is
# ~520 m: under 1.5 km (D 6.1.2) and inside 200-500 m-ish for trackdrive (D 8.1.2).
_CIRCUIT = [
    (0, 0), (25, 0), (50, 0), (70, 0),                      # main straight
    (84, 5), (90, 18),                                      # sweeper onto the back
    (88, 30), (92, 40), (88, 50), (90, 60),                 # slalom, gates 10 m apart
    (88.5, 70), (87.6, 75.3), (82, 78.5), (76.4, 75.3), (75.5, 70),   # hairpin, R 6.5 m
    (74, 62), (66, 56), (58, 58), (52, 64), (44, 64), (38, 58),       # chicane
    (26, 52), (12, 54), (2, 50),                            # run west
    (-6, 42), (-10, 32), (-12, 22), (-12, 12), (-9, 4),     # decreasing-radius left
]


def _catmull_rom_closed(points, step=0.25):
    p = np.asarray(points, dtype=float)
    out = []
    n = len(p)
    for i in range(n):
        p0, p1, p2, p3 = p[i - 1], p[i], p[(i + 1) % n], p[(i + 2) % n]
        seg = max(2, int(np.linalg.norm(p2 - p1) / step))
        t = np.linspace(0.0, 1.0, seg, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out)


def _resample(path, spacing, closed=True):
    pts = np.vstack([path, path[:1]]) if closed else path
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(s[-1] / spacing)
    targets = np.linspace(0.0, s[-1], n, endpoint=not closed)
    return np.column_stack([np.interp(targets, s, pts[:, 0]), np.interp(targets, s, pts[:, 1])])


def _curvature(path):
    """Signed curvature of a closed polyline (positive = turning left)."""
    d1 = (np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)) / 2.0
    d2 = np.roll(path, -1, axis=0) - 2 * path + np.roll(path, 1, axis=0)
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    return num / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-9)


def _relax_tight_corners(path, min_radius, max_iter=3000):
    """Smooth only the stretches tighter than `min_radius`, leaving the rest alone.

    Catmull-Rom splines kink between unevenly spaced control points; this
    irons those kinks out until the whole closed path meets the radius limit
    (D 6.1.1: 9 m minimum turning diameter).
    """
    path = path.copy()
    for _ in range(max_iter):
        radius = 1.0 / np.maximum(np.abs(_curvature(path)), 1e-9)
        tight = radius < min_radius
        if not tight.any():
            break
        grow = np.zeros_like(tight)
        for k in range(-12, 13):                           # relax a neighbourhood, not a point
            grow |= np.roll(tight, k)
        avg = 0.5 * (np.roll(path, 1, axis=0) + np.roll(path, -1, axis=0))
        path[grow] = 0.5 * path[grow] + 0.5 * avg[grow]
        path = _resample(path, 0.25)
    return path


def _boundary_cones(boundary, curvature, closed=True):
    """Walk a boundary polyline placing cones: <= 5 m on straights, tighter in corners."""
    pts = np.vstack([boundary, boundary[:1]]) if closed else boundary
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cones, run = [pts[0]], 0.0
    for i, length in enumerate(seg):
        radius = 1.0 / max(abs(curvature[i % len(curvature)]), 1e-6)
        # DS 1.1: <= 5 m (minus one sampling step of slack), closer in corners
        spacing = float(np.clip(radius * 0.25, 2.5, MAX_CONE_GAP - 0.3))
        run += length
        if run >= spacing:
            cones.append(pts[i + 1])
            run = 0.0
    if closed and np.linalg.norm(cones[-1] - cones[0]) < 1.5:
        cones.pop()
    return np.array(cones)


def circuit_track(name='autocross', laps=1, width=3.5, car_lf=0.8):
    """Closed handling course to D 6.1 / D 8.1, car staged per D 6.2.3 (front
    wheels 6 m before the start line), start = finish line."""
    centre = _relax_tight_corners(_resample(_catmull_rom_closed(_CIRCUIT), 0.25), MIN_RADIUS)
    kappa = _curvature(centre)
    tangent = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])          # points left
    left = centre + normal * width / 2.0
    right = centre - normal * width / 2.0

    # put the start line 25 m down the main straight and the car behind it
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))])
    i_line = int(np.searchsorted(s, 25.0))
    i_car = int(np.searchsorted(s, 25.0 - 6.0 - car_lf))
    origin, heading = centre[i_car], math.atan2(tangent[i_car, 1], tangent[i_car, 0])
    c, sn = math.cos(-heading), math.sin(-heading)
    rot = np.array([[c, -sn], [sn, c]])

    def to_car(p):
        return (np.asarray(p) - origin) @ rot.T

    pos, col = [], []
    line_a, line_b = right[i_line], left[i_line]
    for boundary, tag in ((left, 'blue'), (right, 'yellow')):
        for p in _boundary_cones(boundary, kappa):
            near_line = min(np.linalg.norm(p - line_a), np.linalg.norm(p - line_b)) < 1.2
            if not near_line:
                pos.append(p)
                col.append(tag)
    for k in (-2, 2):                                                    # big orange, DS 1.1
        j = int(np.searchsorted(s, 25.0 + 0.5 * k))
        for p in (left[j], right[j]):
            pos.append(p)
            col.append('big_orange')
    pos = to_car(np.array(pos))
    fwd = tangent[i_line] @ rot.T
    line = TimingLine(tuple(to_car(line_a)), tuple(to_car(line_b)), tuple(fwd))
    mission = 'trackdrive' if laps > 1 else 'autocross'
    return Track(name, pos, col, line, line, mission=mission, laps=laps,
                 centreline=to_car(centre))


# ---- trackside objects -------------------------------------------------------

def add_scenery(track, seed=None):
    """Populate the surroundings with things for the LiDAR to hit.

    Deterministic per track name. Every object keeps >= 3 m from cones and the
    centreline, so it never blocks the racing line.
    """
    rng = np.random.default_rng(seed if seed is not None else sum(map(ord, track.name)))
    keep_out = track.positions
    if track.centreline is not None:
        keep_out = np.vstack([keep_out, track.centreline])
    lo, hi = keep_out.min(axis=0), keep_out.max(axis=0)
    obstacles = []

    def clear(x, y, r, margin=3.0):
        d = np.min(np.hypot(keep_out[:, 0] - x, keep_out[:, 1] - y))
        if d < r + margin:
            return False
        return all(math.hypot(o.x - x, o.y - y) > o.bound_radius + r + 1.0 for o in obstacles)

    # concrete barriers around the area, 6 m blocks with some gaps
    pad = 14.0
    x0, y0, x1, y1 = lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad
    for (ax, ay), (bx, by) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                               ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = math.hypot(bx - ax, by - ay)
        yaw = math.atan2(by - ay, bx - ax)
        for k in range(int(length // 6.0)):
            if rng.random() < 0.2:
                continue
            t = (k + 0.5) * 6.0 / length
            obstacles.append(Obstacle('box', ax + (bx - ax) * t, ay + (by - ay) * t, 0.9,
                                      yaw=yaw, half_x=2.9, half_y=0.3, label='barrier'))
    # light poles just inside the barrier corners
    for x, y in ((x0 + 3, y0 + 3), (x1 - 3, y0 + 3), (x1 - 3, y1 - 3), (x0 + 3, y1 - 3)):
        if clear(x, y, 0.12):
            obstacles.append(Obstacle('cylinder', x, y, 6.0, radius=0.12, label='pole'))
    # timekeeping equipment beside the timing lines (DS 1.4)
    for line in {id(l): l for l in (track.start_line, track.finish_line) if l}.values():
        a, b = np.array(line.a), np.array(line.b)
        out = (b - a) / (np.linalg.norm(b - a) or 1.0)
        for p in (a - out * 2.0, b + out * 2.0):
            if clear(p[0], p[1], 0.25, margin=1.2):
                obstacles.append(Obstacle('box', p[0], p[1], 1.1, yaw=0.0, half_x=0.2,
                                          half_y=0.2, label='timekeeping'))
    # tyre stacks and spare cone stacks dotted 4-10 m off the track
    tries = 0
    want = {'tyres': 10, 'spare_cones': 4}
    while any(want.values()) and tries < 4000:
        tries += 1
        anchor = keep_out[rng.integers(len(keep_out))]
        ang = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(4.0, 10.0)
        x, y = anchor[0] + dist * math.cos(ang), anchor[1] + dist * math.sin(ang)
        if not (x0 + 1 < x < x1 - 1 and y0 + 1 < y < y1 - 1):
            continue
        kind = 'tyres' if want['tyres'] else 'spare_cones'
        if kind == 'tyres' and clear(x, y, 0.35):
            obstacles.append(Obstacle('cylinder', x, y, 0.8, radius=0.35, label='tyres'))
            want['tyres'] -= 1
        elif kind == 'spare_cones' and clear(x, y, 0.15, margin=2.5):
            n = int(rng.integers(3, 7))                       # cones per stack
            color = ['blue', 'yellow', 'orange'][int(rng.integers(3))]
            obstacles.append(Obstacle('cone', x, y, 0.325 + 0.06 * (n - 1), radius=0.114,
                                      label='spare_cones', color=color))
            want['spare_cones'] -= 1
    track.set_obstacles(obstacles)
    return track


BUILTIN = {
    'acceleration': acceleration_track,
    'skidpad': skidpad_track,
    'autocross': lambda: circuit_track('autocross', laps=1),
    'trackdrive': lambda: circuit_track('trackdrive', laps=10),
}


def load_track(spec, scenery=True):
    """`spec` is a builtin name, a CSV path, or the name of a CSV in sim/tracks/."""
    if spec in BUILTIN:
        track = BUILTIN[spec]()
    elif os.path.isfile(spec):
        track = Track.from_csv(spec)
    else:
        bundled = os.path.join(TRACKS_DIR, spec if spec.endswith('.csv') else spec + '.csv')
        if not os.path.isfile(bundled):
            available = sorted(list(BUILTIN) + [os.path.splitext(f)[0] for f in os.listdir(TRACKS_DIR)
                                                if f.endswith('.csv')])
            raise ValueError('Unknown track {!r}. Use a CSV path or one of: {}'.format(
                spec, ', '.join(available)))
        track = Track.from_csv(bundled)
    return add_scenery(track) if scenery else track
