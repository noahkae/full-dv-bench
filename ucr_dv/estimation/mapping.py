"""
Cone mapping: car-frame detections + pose -> a cone map in the world frame.

    none         no map; planning works on the current frame's detections only
    cone_map     landmark map with nearest-neighbour association, running-mean
                 positions, per-colour votes and an observation count; a cone
                 is confirmed after `min_observations` sightings. This is what
                 fsds_ds_stack's mapping_node does, plus colour voting and
                 confirmation (KIT 2022 / AMZ 2019 use the same association,
                 inside GraphSLAM / FastSLAM).

The planner gets the confirmed map transformed back into the car frame,
cropped to `local_radius`, merged with the latest detections - so a cone
seen once a few metres back still bounds the path after it leaves the FOV.
"""
from __future__ import annotations

import numpy as np

from ..core.registry import Component, P, register
from ..core.types import N_COLORS, UNKNOWN, empty_cones, to_car, to_world


@register('mapping', 'none', label='No map (frame-by-frame)', params=[])
class NoMap(Component):
    """Plan on the current detections only."""

    def update(self, cones_car, pose):
        return empty_cones()

    def local(self, cones_car, pose):
        return cones_car


@register('mapping', 'cone_map', label='Landmark cone map',
          params=[P('assoc_radius', 1.0, 0.2, 3.0, 0.05, 'm', 'match a detection to a landmark within this'),
                  P('min_observations', 2, 1, 20, 1, '', 'sightings before a landmark is trusted'),
                  P('local_radius', 25.0, 5.0, 80.0, 1.0, 'm', 'map cones handed to the planner'),
                  P('include_behind', 4.0, 0.0, 20.0, 0.5, 'm', 'keep map cones this far behind the car'),
                  P('max_landmarks', 2000, 100, 20000, 100, '', 'memory cap'),
                  P('localize', True, help='correct odometry drift by aligning each frame to the map (ICP) - '
                                           'this is what makes it SLAM rather than mapping'),
                  P('icp_range', 15.0, 3.0, 40.0, 1.0, 'm', 'use landmarks within this range for alignment'),
                  P('icp_gain', 0.5, 0.05, 1.0, 0.05, '', 'fraction of the ICP correction applied per frame')],
          refs=['fsds_ds_stack mapping_node', 'KIT 2022 (NN association)', 'AMZ 2019 (FastSLAM)'])
class ConeMap(Component):
    """Merge repeated observations of each cone; vote on its colour."""

    def __init__(self, params, **context):
        super(ConeMap, self).__init__(params, **context)
        self.reset()

    def reset(self):
        self.xy = np.zeros((0, 2))
        self.n = np.zeros(0)
        self.w = np.zeros(0)
        self.votes = np.zeros((0, N_COLORS))

    def update(self, cones_car, pose):
        if len(cones_car):
            w = to_world(cones_car[:, :2], pose)
            rng = np.hypot(cones_car[:, 0], cones_car[:, 1])
            # sightings from close up are far more accurate: weight ~ 1 / range^2
            weight = 1.0 / (0.25 + (rng / 8.0) ** 2)
            base = self.params['assoc_radius']
            for k in range(len(w)):
                col = int(cones_car[k, 2])
                conf = float(cones_car[k, 3])
                r = base * (1.0 + rng[k] / 20.0)            # far sightings are less certain
                if len(self.xy):
                    d = np.hypot(self.xy[:, 0] - w[k, 0], self.xy[:, 1] - w[k, 1])
                    j = int(np.argmin(d))
                    if d[j] < r:
                        self.n[j] += 1
                        wk = weight[k]
                        self.xy[j] += (w[k] - self.xy[j]) * wk / (self.w[j] + wk)
                        self.w[j] = min(self.w[j] + wk, 200.0)
                        self.votes[j, col] += (conf if col != UNKNOWN else 0.1 * conf) * min(1.0, wk)
                        continue
                if len(self.xy) < self.params['max_landmarks']:
                    self.xy = np.vstack([self.xy, w[k]])
                    self.n = np.append(self.n, 1.0)
                    self.w = np.append(self.w, weight[k])
                    v = np.zeros(N_COLORS)
                    v[col] = conf if col != UNKNOWN else 0.1 * conf
                    self.votes = np.vstack([self.votes, v])
        return self.confirmed()

    def localize(self, cones_car, pose):
        """Pose corrected by a damped 2D ICP of the detections onto the confirmed map."""
        from ..core.types import Pose2D
        p = self.params
        if not p['localize'] or len(cones_car) < 3:
            return pose
        ok = self.n >= p['min_observations']
        m = self.xy[ok]
        if len(m) < 3:
            return pose
        near = np.hypot(cones_car[:, 0], cones_car[:, 1]) < p['icp_range']
        src_car = cones_car[near, :2]
        if len(src_car) < 3:
            return pose
        x, y, yaw = pose.x, pose.y, pose.yaw
        for _ in range(3):
            w = to_world(src_car, Pose2D(x, y, yaw))
            d = np.hypot(w[:, None, 0] - m[None, :, 0], w[:, None, 1] - m[None, :, 1])
            j = np.argmin(d, axis=1)
            good = d[np.arange(len(w)), j] < p['assoc_radius']
            if good.sum() < 3:
                return pose if _ == 0 else Pose2D(x, y, yaw)
            a, b = w[good], m[j[good]]
            ma, mb = a.mean(axis=0), b.mean(axis=0)
            H = (a - ma).T @ (b - mb)
            th = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
            th = float(np.clip(th, -0.03, 0.03)) * p['icp_gain']
            c, s_ = np.cos(th), np.sin(th)
            R = np.array([[c, -s_], [s_, c]])
            # damped rigid fix about the matched centroid: rotate by th, shift by gain x offset
            shift = np.clip((mb - ma) * p['icp_gain'], -0.25, 0.25)
            nx, ny = R @ (np.array([x, y]) - ma) + ma + shift
            x, y, yaw = float(nx), float(ny), float(yaw + th)
        return Pose2D(x, y, yaw)

    def colors(self):
        if not len(self.votes):
            return np.zeros(0)
        known = self.votes[:, :UNKNOWN]
        col = np.argmax(known, axis=1).astype(float)
        col[known.max(axis=1) <= 0] = UNKNOWN
        return col

    def confirmed(self):
        ok = self.n >= self.params['min_observations']
        out = np.zeros((int(ok.sum()), 4))
        out[:, :2] = self.xy[ok]
        out[:, 2] = self.colors()[ok]
        out[:, 3] = np.minimum(1.0, self.n[ok] / 5.0)
        return out

    def local(self, cones_car, pose):
        """Confirmed map near the car (car frame), merged with fresh detections."""
        m = self.confirmed()
        if not len(m):
            return cones_car
        c = m.copy()
        c[:, :2] = to_car(m[:, :2], pose)
        keep = (np.hypot(c[:, 0], c[:, 1]) < self.params['local_radius']) & (c[:, 0] > -self.params['include_behind'])
        c = c[keep]
        if not len(cones_car):
            return c
        # fresh detections win where they overlap the map (they're current)
        d = np.min(np.hypot(c[:, None, 0] - cones_car[None, :, 0], c[:, None, 1] - cones_car[None, :, 1]), axis=1) \
            if len(c) else np.zeros(0)
        merged = np.vstack([cones_car, c[d > self.params['assoc_radius']]])
        # carry the map's (voted) colour onto uncoloured fresh detections
        if len(c):
            unk = merged[:, 2] == UNKNOWN
            if unk.any():
                dd = np.hypot(merged[unk, None, 0] - c[None, :, 0], merged[unk, None, 1] - c[None, :, 1])
                j = np.argmin(dd, axis=1)
                close = dd[np.arange(len(j)), j] < self.params['assoc_radius']
                idx = np.nonzero(unk)[0][close]
                merged[idx, 2] = c[j[close], 2]
        return merged
