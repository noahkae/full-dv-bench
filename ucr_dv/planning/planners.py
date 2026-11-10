"""
Path planners: cones in the car frame -> centre path in the car frame.

    greedy_pairing   ucr-dv-timeless' CTU-style planner: pair the nearest blue
                     and yellow cones ahead of the last centre point, step by
                     step. Needs colours; fills a missing side 3.5 m across.
    gates            fsds_ds_stack's colour-blind gate pairing: any two cones a
                     plausible track width apart, across the heading.
    delaunay         Delaunay triangulation of the cones; midpoints of edges
                     that cross the track (blue-yellow, or width-plausible if
                     colours are unknown), then a forward search through them
                     with heading and step constraints (AMZ 2019, KIT 2022,
                     IIT Bombay 2024 - the standard competitive choice).
    track_template   follow a known layout (skidpad / acceleration rules
                     geometry) with odometry - what teams do for the static
                     events. Falls back to `delaunay` on unknown tracks.

Every planner returns a path starting at the car, resampled and (optionally)
smoothed, so speed profiles and controllers see the same thing.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.registry import Component, P, register
from ..core.types import BIG_ORANGE, BLUE, ORANGE, UNKNOWN, YELLOW, to_car
from .pathutil import anchor_to_car, resample, smooth

try:
    from scipy.spatial import Delaunay as _ScipyDelaunay
    HAVE_SCIPY = True
except Exception:                       # pragma: no cover
    HAVE_SCIPY = False

POST = [P('spacing', 0.5, 0.1, 2.0, 0.1, 'm', 'output point spacing'),
        P('smoothing', 0.25, 0.0, 1.0, 0.05, '', 'Laplacian smoothing weight (0 = off)'),
        P('max_length', 30.0, 5.0, 100.0, 1.0, 'm', 'truncate the path beyond this')]


def _post(path, p):
    path = anchor_to_car(path)
    if len(path) < 2:
        return np.array([[0.0, 0.0], [1.0, 0.0]])
    path = resample(path, p['spacing'])
    if p['smoothing'] > 0:
        path = smooth(path, p['smoothing'], passes=15)
        path = resample(path, p['spacing'])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])
    return path[s <= p['max_length']]


def _sides(cones):
    """Treat orange cones as either side; big orange too (they mark lines)."""
    c = cones
    blue = c[c[:, 2] == BLUE, :2]
    yellow = c[c[:, 2] == YELLOW, :2]
    other = c[(c[:, 2] == UNKNOWN) | (c[:, 2] == ORANGE) | (c[:, 2] == BIG_ORANGE), :2]
    return blue, yellow, other


# ---- greedy pairing (ported from ucr-dv-timeless helpers/path_planning.py) ----

@register('planner', 'greedy_pairing', label='Greedy blue/yellow pairing (timeless)',
          params=[P('n_steps', 20, 2, 60, 1, '', 'centre points to chain'),
                  P('fill_distance', 3.5, 2.0, 6.0, 0.1, 'm', 'offset for a missing side')] + POST,
          refs=['ucr-dv-timeless PathPlanner (CTU)'], needs=['colors'])
class GreedyPairingPlanner(Component):
    """Walk forward pairing the nearest blue and yellow cone each step. Needs colours."""

    def plan(self, cones, ctx):
        blue, yellow, other = _sides(cones)
        n_steps = self.params['n_steps']
        if len(blue) == 0 and len(yellow) == 0:
            return _post(np.array([[0.0, 0.0], [2.0, 0.0]]), self.params)
        if len(blue) == 0 or len(yellow) == 0:
            full = yellow if len(yellow) else blue
            full = full[np.argsort(full[:, 0])]
            sign = 1.0 if len(yellow) else -1.0          # yellow is right: blue is +left of it
            fill = self.params['fill_distance']
            filled = []
            for i in range(len(full)):
                j = min(i + 1, len(full) - 1)
                k = max(j - 1, 0)
                t = full[j] - full[k] if len(full) > 1 else np.array([1.0, 0.0])
                t = t / max(np.linalg.norm(t), 1e-9)
                nrm = np.array([-t[1], t[0]]) * sign
                filled.append(full[i] + nrm * fill)
            filled = np.array(filled)
            blue, yellow = (filled, full) if len(yellow) else (full, filled)
        centres = [np.array([0.0, 0.0])]
        prev_dir = np.array([1.0, 0.0])
        used_b, used_y = set(), set()
        for _ in range(n_steps):
            cur = centres[-1]
            def ahead(pts, used):
                rel = pts - cur
                ok = (rel @ prev_dir) > 0.2
                idx = [i for i in np.nonzero(ok)[0] if i not in used]
                if not idx:
                    return None
                d = np.hypot(rel[idx, 0], rel[idx, 1])
                return idx[int(np.argmin(d))]
            bi, yi = ahead(blue, used_b), ahead(yellow, used_y)
            if bi is None or yi is None:
                break
            used_b.add(bi)
            used_y.add(yi)
            c = 0.5 * (blue[bi] + yellow[yi])
            step = c - cur
            if np.linalg.norm(step) > 8.0:
                break
            if np.linalg.norm(step) > 0.3:
                prev_dir = step / np.linalg.norm(step)
            centres.append(c)
        return _post(np.array(centres), self.params)


# ---- gates (ported from fsds_ds_stack planning/gates.py) ---------------------

def find_gates(cones_xy, min_width=2.0, max_width=5.0, max_angle_deg=60.0):
    cones = np.asarray(cones_xy, dtype=float).reshape(-1, 2)
    n = len(cones)
    if n < 2:
        return []
    i, j = np.triu_indices(n, k=1)
    d = cones[j] - cones[i]
    width = np.hypot(d[:, 0], d[:, 1])
    across = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))
    ok = (width >= min_width) & (width <= max_width) & (across <= max_angle_deg)
    gates, used = [], set()
    for k in np.flatnonzero(ok)[np.argsort(width[ok], kind='stable')]:
        a, b = int(i[k]), int(j[k])
        if a in used or b in used:
            continue
        used.update((a, b))
        gates.append((cones[a], cones[b]))
    return gates


@register('planner', 'gates', label='Gate midpoints (fsds_ds_stack)',
          params=[P('min_width', 2.0, 1.0, 4.0, 0.1, 'm', 'narrowest gate'),
                  P('max_width', 5.0, 2.5, 8.0, 0.1, 'm', 'widest gate'),
                  P('max_angle', 60.0, 10.0, 89.0, 1.0, 'deg', 'gate line vs the car\'s y axis')] + POST,
          refs=['fsds_ds_stack planning_node'])
class GatePlanner(Component):
    """Colour-blind: pair cones a track-width apart, sort midpoints by x."""

    def plan(self, cones, ctx):
        p = self.params
        ahead = cones[cones[:, 0] > -1.0]
        mids = [0.5 * (a + b) for a, b in find_gates(ahead[:, :2], p['min_width'], p['max_width'], p['max_angle'])]
        if not mids:
            return _post(np.array([[0.0, 0.0], [2.0, 0.0]]), p)
        mids = np.array(sorted(mids, key=lambda m: m[0]))
        mids = mids[mids[:, 0] > 0]
        return _post(np.vstack([[0.0, 0.0], mids]) if len(mids) else np.array([[0.0, 0.0], [2.0, 0.0]]), p)


# ---- Delaunay -----------------------------------------------------------------

def delaunay_edges(xy):
    """Unique undirected edges of the Delaunay triangulation of (N, 2) points."""
    n = len(xy)
    if n < 3:
        return np.array([[0, 1]]) if n == 2 else np.zeros((0, 2), dtype=int)
    tris = None
    if HAVE_SCIPY:
        try:
            tris = _ScipyDelaunay(xy).simplices
        except Exception:
            tris = None
    if tris is None:
        tris = bowyer_watson(xy)
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [0, 2]]])
    e.sort(axis=1)
    return np.unique(e, axis=0)


def bowyer_watson(xy):
    """Plain-numpy Delaunay (O(n^2)) - used when scipy is missing."""
    pts = np.asarray(xy, dtype=float)
    n = len(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = max(hi - lo) * 20 + 1.0
    mid = (lo + hi) / 2
    sup = np.array([[mid[0] - span, mid[1] - span], [mid[0] + span, mid[1] - span], [mid[0], mid[1] + span]])
    allp = np.vstack([pts, sup])
    tris = [(n, n + 1, n + 2)]

    def circ(t):
        (ax, ay), (bx, by), (cx, cy) = allp[t[0]], allp[t[1]], allp[t[2]]
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            return (0.0, 0.0, float('inf'))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        return (ux, uy, (ax - ux) ** 2 + (ay - uy) ** 2)

    cache = {tris[0]: circ(tris[0])}
    for i in range(n):
        px, py = pts[i]
        bad = [t for t in tris if (px - cache[t][0]) ** 2 + (py - cache[t][1]) ** 2 < cache[t][2]]
        edges = {}
        for t in bad:
            for e in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2])):
                k = tuple(sorted(e))
                edges[k] = edges.get(k, 0) + 1
        bset = set(bad)
        tris = [t for t in tris if t not in bset]
        for (a, b), cnt in edges.items():
            if cnt == 1:
                t = (a, b, i)
                tris.append(t)
                cache[t] = circ(t)
    out = [t for t in tris if max(t) < n]
    return np.array(out, dtype=int).reshape(-1, 3)


@register('planner', 'delaunay', label='Delaunay midline',
          params=[P('min_width', 2.0, 1.0, 4.0, 0.1, 'm', 'narrowest crossing edge'),
                  P('max_width', 6.0, 3.0, 10.0, 0.1, 'm', 'widest crossing edge'),
                  P('max_step', 6.0, 2.0, 12.0, 0.25, 'm', 'longest jump between midpoints'),
                  P('max_turn', 50.0, 10.0, 90.0, 1.0, 'deg', 'largest heading change per step'),
                  P('turn_weight', 2.0, 0.0, 10.0, 0.1, 'm/rad', 'cost of turning vs distance'),
                  P('consistency', 1.0, 0.0, 10.0, 0.1, '', 'cost per metre away from the previous path '
                                                              '(stops the path flickering between ticks)'),
                  P('max_skew', 50.0, 10.0, 90.0, 1.0, 'deg', 'a crossing edge may be at most this far from '
                                                           'perpendicular to the direction of travel'),
                  P('use_colors', True, help='only blue-yellow edges when both colours are seen'),
                  P('max_cone_range', 18.0, 5.0, 60.0, 0.5, 'm', 'ignore cones further than this'),
                  P('side_inference', True, help='label colourless cones left/right of the previous path '
                                                 '(lets a LiDAR-only rig use the coloured logic)'),
                  P('virtual_mids', True, help='coloured cones with no partner across the track get a '
                                               'midpoint half a track width inwards'),
                  P('half_width', 1.6, 1.0, 3.0, 0.05, 'm', 'assumed half track width for virtual midpoints')] + POST,
          refs=['AMZ 2019 §5', 'KIT 2022 (triangulation + spline)', 'IIT Bombay 2024'])
class DelaunayPlanner(Component):
    """Triangulate the cones, keep edges that cross the track, chain their midpoints."""

    def __init__(self, params, **context):
        super(DelaunayPlanner, self).__init__(params, **context)
        self.reset()

    def reset(self):
        self._prev_world = None

    def _prev_in_car(self, ctx):
        pose = ctx.get('pose')
        if self._prev_world is None or pose is None:
            return None
        return to_car(self._prev_world, pose)

    def plan(self, cones, ctx):
        p = self.params
        c = cones[(cones[:, 0] > -3.0) & (np.hypot(cones[:, 0], cones[:, 1]) < p['max_cone_range'])]
        if len(c) < 2:
            return _post(np.array([[0.0, 0.0], [2.0, 0.0]]), p)
        xy = c[:, :2]
        col = c[:, 2].astype(int)
        if p['side_inference']:
            col = self._infer_sides(xy, col, ctx)
        e = delaunay_edges(xy)
        a, b = xy[e[:, 0]], xy[e[:, 1]]
        w = np.hypot(*(b - a).T)
        ok = (w >= p['min_width']) & (w <= p['max_width'])
        ca, cb = col[e[:, 0]], col[e[:, 1]]
        have_colors = (col == BLUE).any() and (col == YELLOW).any()
        if p['use_colors'] and have_colors:
            same_side = ((ca == BLUE) & (cb == BLUE)) | ((ca == YELLOW) & (cb == YELLOW))
            ok &= ~same_side
        mids = 0.5 * (a[ok] + b[ok])
        dirs = (b[ok] - a[ok]) / np.maximum(w[ok], 1e-9)[:, None]
        penalty = np.zeros(len(mids))
        if p['virtual_mids']:
            used = np.zeros(len(xy), dtype=bool)
            used[e[ok, 0]] = True
            used[e[ok, 1]] = True
            vm, vd = self._virtual(xy, col, used)
            if len(vm):
                mids = np.vstack([mids, vm])
                dirs = np.vstack([dirs, vd])
                penalty = np.concatenate([penalty, np.full(len(vm), 2.0)])
        if not len(mids):
            return _post(np.array([[0.0, 0.0], [2.0, 0.0]]), p)
        prev = self._prev_in_car(ctx)
        if prev is not None and len(prev) > 1 and p['consistency'] > 0:
            dprev = np.min(np.hypot(mids[:, None, 0] - prev[None, :, 0], mids[:, None, 1] - prev[None, :, 1]), axis=1)
            penalty = penalty + p['consistency'] * np.minimum(dprev, 3.0)
        path = self._chain(mids, ctx, penalty, dirs)
        self.last_debug = {'edges': np.stack([a[ok], b[ok]], axis=1), 'mids': mids}
        out = _post(path, p)
        pose = ctx.get('pose')
        if pose is not None and len(out) > 2:
            from ..core.types import to_world
            self._prev_world = to_world(out, pose)
        return out

    def _infer_sides(self, xy, col, ctx):
        """Colourless cones -> BLUE (left) / YELLOW (right) of the previous path."""
        unk = (col == UNKNOWN) | (col == ORANGE) | (col == BIG_ORANGE)
        # a coloured cone well over on the other side of the path is a colour
        # error (or a false detection): geometry wins
        override = self.params.get('geometry_override', 1.2)
        ref = self._prev_in_car(ctx)
        if ref is None or len(ref) < 2 or np.hypot(*ref[0]) > 3.0:
            ref = np.array([[-5.0, 0.0], [40.0, 0.0]])
        else:
            d0 = ref[1] - ref[0]
            d1 = ref[-1] - ref[-2]
            d0 /= max(np.linalg.norm(d0), 1e-9)
            d1 /= max(np.linalg.norm(d1), 1e-9)
            ref = np.vstack([ref[0] - 5.0 * d0, ref, ref[-1] + 25.0 * d1])
        out = col.copy()
        a, b = ref[:-1], ref[1:]
        seg = b - a
        L2 = np.maximum((seg ** 2).sum(axis=1), 1e-9)
        for i in range(len(xy)):
            t = np.clip(((xy[i] - a) * seg).sum(axis=1) / L2, 0.0, 1.0)
            proj = a + seg * t[:, None]
            dist = np.hypot(*(xy[i] - proj).T)
            k = int(np.argmin(dist))
            lat = seg[k, 0] * (xy[i, 1] - a[k, 1]) - seg[k, 1] * (xy[i, 0] - a[k, 0])
            side = BLUE if lat > 0 else YELLOW
            if unk[i]:
                if dist[k] > 0.5:
                    out[i] = side
            elif override and dist[k] > override and col[i] != side and col[i] in (BLUE, YELLOW):
                out[i] = side
        return out

    def _virtual(self, xy, col, used):
        """Midpoints for unpaired coloured cones: offset inwards along the boundary normal."""
        out = []
        hw = self.params['half_width']
        for side, sign in ((BLUE, -1.0), (YELLOW, 1.0)):       # inside is right of blue, left of yellow
            idx = np.nonzero(col == side)[0]
            if len(idx) < 2:
                continue
            pts = xy[idx]
            for k in np.nonzero(~used[idx])[0]:
                d = np.hypot(pts[:, 0] - pts[k, 0], pts[:, 1] - pts[k, 1])
                d[k] = np.inf
                j = int(np.argmin(d))
                if d[j] > 7.0:
                    continue
                t = pts[j] - pts[k]
                # orient the boundary tangent away from the car (direction of travel)
                if np.dot(t, pts[k] + pts[j]) < 0:
                    t = -t
                t = t / np.linalg.norm(t)
                nrm = np.array([-t[1], t[0]]) * sign
                out.append(np.concatenate([pts[k] + nrm * hw, nrm]))
        arr = np.array(out).reshape(-1, 4)
        return arr[:, :2], arr[:, 2:]

    def _chain(self, mids, ctx, penalty=None, dirs=None):
        """Greedy forward search: cheapest (distance + turn) reachable midpoint each step."""
        p = self.params
        cur = np.array([0.0, 0.0])
        heading = 0.0
        out = [cur]
        alive = np.ones(len(mids), dtype=bool)
        max_turn = math.radians(p['max_turn'])
        for step in range(60):
            rel = mids - cur
            d = np.hypot(rel[:, 0], rel[:, 1])
            alive &= d > 0.4                       # passed / reached
            if not alive.any():
                break
            ang = np.arctan2(rel[:, 1], rel[:, 0])
            turn = np.abs((ang - heading + math.pi) % (2 * math.pi) - math.pi)
            if step == 0:                          # the car may not be aligned with the track yet
                valid = alive & (d <= p['max_step'] * 1.5) & (turn <= math.radians(80))
            else:
                valid = alive & (d <= p['max_step']) & (turn <= max_turn)
            skew = np.zeros(len(mids))
            if dirs is not None:
                # |cos| between the crossing edge and the way we'd travel to it:
                # 0 = edge straight across the track (good), 1 = edge along it (a same-side pair)
                u = rel / np.maximum(d, 1e-9)[:, None]
                travel = np.column_stack([np.cos(heading), np.sin(heading)]) if step else u
                skew = np.abs(np.sum(dirs * (0.5 * (u + travel)), axis=1))
                valid &= skew <= math.cos(math.radians(90.0 - p['max_skew']))
            if not valid.any():
                break
            cost = np.where(valid, d + p['turn_weight'] * turn + 2.0 * skew
                            + (penalty if penalty is not None else 0.0), np.inf)
            k = int(np.argmin(cost))
            nxt = mids[k]
            heading = math.atan2(nxt[1] - cur[1], nxt[0] - cur[0])
            # everything behind the new point (relative to the step) is used up
            behind = (rel @ np.array([math.cos(heading), math.sin(heading)])) < (d[k] - 0.5)
            alive &= ~(behind & (d < d[k] + 0.5))
            alive[k] = False
            cur = nxt
            out.append(cur)
        return np.array(out)


@register('planner', 'track_template', label='Known layout + odometry (skidpad/accel)',
          params=[P('lookahead_points', 60, 10, 400, 5, '', 'template points handed on'),
                  P('fallback', 'boundaries', choices=['boundaries', 'delaunay', 'gates', 'greedy_pairing'],
                    help='planner for tracks without a known template')] + POST,
          refs=['IIT Bombay 2024 (skidpad template)'])
class TemplatePlanner(Component):
    """Static events have a rule-defined layout: follow it with the pose estimate."""

    def __init__(self, params, **context):
        super(TemplatePlanner, self).__init__(params, **context)
        self._fallback = None
        self._idx = 0

    def reset(self):
        self._idx = 0

    def plan(self, cones, ctx):
        tmpl = ctx.get('template')
        if tmpl is None or not len(tmpl):
            if self._fallback is None:
                from ..core.registry import get
                self._fallback = get('planner', self.params['fallback']).create({})
            return self._fallback.plan(cones, ctx)
        pose = ctx['pose']
        n = len(tmpl)
        lo, hi = self._idx, min(n, self._idx + 80)
        d = np.hypot(tmpl[lo:hi, 0] - pose.x, tmpl[lo:hi, 1] - pose.y)
        self._idx = lo + int(np.argmin(d))
        seg = tmpl[self._idx:self._idx + int(self.params['lookahead_points'])]
        if len(seg) < 2:
            seg = tmpl[-2:]
        return _post(to_car(seg, pose), dict(self.params, smoothing=0.0))


# ---- boundary chains --------------------------------------------------------------

def _walk(points, start_dir=(1.0, 0.0), max_gap=6.0, max_turn_deg=65.0, origin=(0.0, 0.0)):
    """Order boundary cones by walking forward from the car: nearest cone ahead
    within `max_gap`, turning at most `max_turn_deg` per step."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(pts):
        return pts
    cur = np.asarray(origin, dtype=float)
    d = np.asarray(start_dir, dtype=float)
    alive = np.ones(len(pts), dtype=bool)
    out = []
    max_turn = math.radians(max_turn_deg)
    for step in range(len(pts)):
        rel = pts - cur
        dist = np.hypot(rel[:, 0], rel[:, 1])
        ahead = rel @ d
        ang = np.abs(np.arctan2(rel @ np.array([-d[1], d[0]]), ahead))
        lim = max_gap * (1.6 if step == 0 else 1.0)
        valid = alive & (dist > 0.3) & (dist < lim) & (ang < (math.radians(80) if step == 0 else max_turn))
        if not valid.any():
            break
        k = int(np.argmin(np.where(valid, dist + 2.0 * ang, np.inf)))
        if step > 0:
            d = rel[k] / dist[k]
        out.append(pts[k])
        cur = pts[k]
        alive[k] = False
        alive &= ~((rel @ d) < 0.0) | (dist > lim)      # drop cones left behind
    return np.array(out).reshape(-1, 2)


@register('planner', 'boundaries', label='Boundary chains + centreline',
          params=[P('max_gap', 6.0, 2.0, 12.0, 0.25, 'm', 'largest gap between cones of one boundary'),
                  P('max_turn', 65.0, 10.0, 90.0, 1.0, 'deg', 'largest boundary heading change per cone'),
                  P('half_width', 1.6, 1.0, 3.0, 0.05, 'm', 'offset used when only one boundary is seen'),
                  P('max_cone_range', 20.0, 5.0, 60.0, 0.5, 'm', 'ignore cones further than this'),
                  P('side_inference', True, help='label colourless cones by their side of the car / last path'),
                  P('geometry_override', 1.2, 0.0, 5.0, 0.1, 'm', 'a coloured cone this far on the wrong side of '
                                                                  'the last path is relabelled (0 = trust colours)')] + POST,
          refs=['AMZ 2019 (boundary estimation)', 'ucr-dv-timeless fill_missing (offset one side)'],
          needs=[])
class BoundaryPlanner(Component):
    """Chain the left (blue) and right (yellow) cones separately, then take the
    midline between the chains - or offset the one chain that is visible."""

    def __init__(self, params, **context):
        super(BoundaryPlanner, self).__init__(params, **context)
        from ..core.registry import get
        self._helper = get('planner', 'delaunay').create({}, **context)

    def reset(self):
        self._helper.reset()

    def plan(self, cones, ctx):
        p = self.params
        c = cones[(cones[:, 0] > -2.0) & (np.hypot(cones[:, 0], cones[:, 1]) < p['max_cone_range'])]
        if len(c) == 0:
            return _post(np.array([[0.0, 0.0], [2.0, 0.0]]), p)
        col = c[:, 2].astype(int)
        if p['side_inference']:
            self._helper.params['geometry_override'] = p['geometry_override']
            col = self._helper._infer_sides(c[:, :2], col, ctx)
        left = _walk(c[col == BLUE, :2], max_gap=p['max_gap'], max_turn_deg=p['max_turn'])
        right = _walk(c[col == YELLOW, :2], max_gap=p['max_gap'], max_turn_deg=p['max_turn'])
        hw = p['half_width']
        if len(left) >= 2 and len(right) >= 2:
            L, R = resample(np.vstack([[left[0][0] - 1.0, left[0][1]], left]), 0.5), \
                resample(np.vstack([[right[0][0] - 1.0, right[0][1]], right]), 0.5)
            # pair each point of the longer chain with the nearest of the other
            A, B = (L, R) if len(L) >= len(R) else (R, L)
            j = np.argmin(np.hypot(A[:, None, 0] - B[None, :, 0], A[:, None, 1] - B[None, :, 1]), axis=1)
            mids = 0.5 * (A + B[j])
            # stop where the shorter chain ends (beyond it the pairing is meaningless)
            mids = mids[np.hypot(*(A - B[j]).T) < 2.0 * hw + 2.5]
        elif len(left) >= 2 or len(right) >= 2:
            chain, sign = (left, -1.0) if len(left) >= 2 else (right, 1.0)
            ch = resample(chain, 0.5)
            t = np.gradient(ch, axis=0)
            t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
            nrm = np.column_stack([-t[:, 1], t[:, 0]]) * sign
            mids = ch + nrm * hw
        else:
            return self._helper.plan(cones, ctx)
        mids = mids[mids[:, 0] > 0.3] if len(mids) else mids
        if len(mids) < 2:
            return self._helper.plan(cones, ctx)
        path = _post(np.vstack([[0.0, 0.0], mids]), p)
        pose = ctx.get('pose')
        if pose is not None and len(path) > 2:
            from ..core.types import to_world
            self._helper._prev_world = to_world(path, pose)
        self.last_debug = {'mids': mids, 'edges': np.zeros((0, 2, 2)),
                           'left': left, 'right': right}
        return path
