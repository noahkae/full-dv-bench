"""
Sensor fusion: cones from each sensor (car frame) -> one cone list.

    concat           stack everything, then merge duplicates closer than
                     `merge_radius` (keeps the best-confidence colour)
    nearest_assoc    associate across sensors by distance; position is the
                     inverse-covariance (or confidence) weighted mean, colour
                     comes from whichever sensor saw colour (cameras), so a
                     LiDAR + camera rig gets LiDAR range with camera colour
                     (AMZ 2019/2020, Heriot-Watt 2025 weighted average)
    priority         take one sensor's cones as-is and only add the others'
                     cones where it saw nothing (simple, robust fallback)
"""
from __future__ import annotations

import numpy as np

from ..core.registry import Component, P, register
from ..core.types import N_COLORS, UNKNOWN, empty_cones


try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components as _cc
except Exception:                           # pragma: no cover
    _cc = None


def _cluster_by_distance(xy, radius, sensor_id=None, cov=None, chi2=None):
    """Single-link groups of points closer than `radius` (group id per point).

    With `sensor_id`, only detections from DIFFERENT sensors are linked: one
    sensor's two detections are two objects by that detector's own judgement
    (e.g. a big orange cone right next to a blue one at the start line).
    """
    n = len(xy)
    if n == 0:
        return np.zeros(0, dtype=int)
    diff = xy[:, None, :] - xy[None, :, :]
    d = np.hypot(diff[..., 0], diff[..., 1])
    adj = d < radius
    if cov is not None and chi2 is not None:
        # Mahalanobis gate with the pair's combined covariance: a camera's long
        # range uncertainty lets it pair with the LiDAR cone straight down its
        # bearing, but not with the neighbour beside it
        S = cov[:, None] + cov[None, :]
        det = S[..., 0, 0] * S[..., 1, 1] - S[..., 0, 1] * S[..., 1, 0]
        det = np.where(det > 1e-12, det, 1e-12)
        m2 = (S[..., 1, 1] * diff[..., 0] ** 2 - 2 * S[..., 0, 1] * diff[..., 0] * diff[..., 1]
              + S[..., 0, 0] * diff[..., 1] ** 2) / det
        adj = (m2 < chi2) & (d < radius)
    if sensor_id is None:
        if _cc is not None:
            return _cc(csr_matrix(adj), directed=False)[1]
        sensor_id = np.arange(n)            # fall through: every point its own "sensor"
    # greedy association, closest pairs first: a group holds at most one
    # detection per sensor, so one sensor's detection can't chain two of
    # another sensor's objects together
    score = m2 if (cov is not None and chi2 is not None) else d
    ii, jj = np.nonzero(np.triu(adj & (sensor_id[:, None] != sensor_id[None, :]), 1))
    order = np.argsort(score[ii, jj], kind='stable')
    parent = list(range(n))
    members = [{int(sensor_id[k])} for k in range(n)]

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for k in order:
        a, b = find(int(ii[k])), find(int(jj[k]))
        if a == b or members[a] & members[b]:
            continue
        parent[b] = a
        members[a] |= members[b]
    roots = np.array([find(k) for k in range(n)])
    _, group = np.unique(roots, return_inverse=True)
    return np.asarray(group).reshape(-1)


def _merge(all_cones, all_cov, radius, use_cov, sensor_id=None, chi2=None):
    """Merge each group into one cone: weighted mean position, voted colour."""
    if not len(all_cones):
        return empty_cones()
    group = _cluster_by_distance(all_cones[:, :2], radius, sensor_id,
                                 all_cov if (use_cov and chi2) else None, chi2)
    G = int(group.max()) + 1
    w = np.maximum(all_cones[:, 3], 1e-3)
    if use_cov and all_cov is not None:
        info = np.linalg.inv(all_cov + np.eye(2)[None] * 1e-6)              # (N, 2, 2)
        S = np.zeros((G, 2, 2))
        np.add.at(S, group, info)
        b = np.zeros((G, 2))
        np.add.at(b, group, np.einsum('nij,nj->ni', info, all_cones[:, :2]))
        pos = np.linalg.solve(S, b[..., None])[..., 0]
    else:
        num = np.zeros((G, 2))
        np.add.at(num, group, all_cones[:, :2] * w[:, None])
        den = np.zeros(G)
        np.add.at(den, group, w)
        pos = num / den[:, None]
    votes = np.zeros((G, N_COLORS))
    col = all_cones[:, 2].astype(int)
    known = col != UNKNOWN
    np.add.at(votes, (group[known], col[known]), w[known])
    color = np.where(votes[:, :UNKNOWN].max(axis=1) > 0, np.argmax(votes[:, :UNKNOWN], axis=1), UNKNOWN)
    count = np.bincount(group, minlength=G)
    best = np.zeros(G)
    np.maximum.at(best, group, all_cones[:, 3])
    conf = np.minimum(1.0, best * (1.0 + 0.2 * (count - 1)))
    return np.column_stack([pos, color, conf])


def _stack(cones_by_sensor, cov_by_sensor):
    cones, covs, sids, have_cov = [], [], [], True
    for k, (name, c) in enumerate(cones_by_sensor.items()):
        if not len(c):
            continue
        cones.append(c)
        sids.append(np.full(len(c), k))
        cv = cov_by_sensor.get(name)
        if cv is None or len(cv) != len(c):
            have_cov = False
            cv = np.repeat((np.eye(2) * 0.05)[None], len(c), axis=0)
        covs.append(cv)
    if not cones:
        return empty_cones(), None, False, np.zeros(0, dtype=int)
    return np.vstack(cones), np.vstack(covs), have_cov, np.concatenate(sids)


@register('fusion', 'concat', label='Concatenate + merge duplicates',
          params=[P('merge_radius', 0.6, 0.1, 2.0, 0.05, 'm', 'detections closer than this are one cone'),
                  P('min_confidence', 0.0, 0.0, 1.0, 0.05, '', 'drop detections below this confidence')])
class ConcatFusion(Component):
    """All sensors' cones in one list; near-duplicates merged."""

    def fuse(self, cones_by_sensor, cov_by_sensor):
        cones, cov, _, sid = _stack(cones_by_sensor, cov_by_sensor)
        if not len(cones):
            return cones
        keep = cones[:, 3] >= self.params['min_confidence']
        return _merge(cones[keep], None, self.params['merge_radius'], False, sid[keep])


@register('fusion', 'nearest_assoc', label='Nearest association (covariance weighted)',
          params=[P('gate', 2.5, 0.1, 5.0, 0.05, 'm', 'never associate detections further apart than this'),
                  P('use_covariance', True, help='Mahalanobis gating + inverse-covariance weighting'),
                  P('chi2', 9.21, 1.0, 30.0, 0.5, '', 'Mahalanobis gate (9.21 = 99% for 2 dof)'),
                  P('min_confidence', 0.1, 0.0, 1.0, 0.05, '', 'drop detections below this confidence'),
                  P('require_color', False, help='drop fused cones nobody could colour'),
                  P('lidar_confirm_range', 25.0, 0.0, 50.0, 0.5, 'm', 'inside this range a camera-only cone must '
                                                                     'be confirmed by a LiDAR (0 = off)')],
          refs=['AMZ 2019/2020 camera-LiDAR fusion', 'Heriot-Watt 2025 confidence-weighted average'])
class NearestFusion(Component):
    """LiDAR range + camera colour: associate by distance, weight by covariance."""

    def fuse(self, cones_by_sensor, cov_by_sensor):
        cones, cov, have_cov, sid = _stack(cones_by_sensor, cov_by_sensor)
        if not len(cones):
            return cones
        keep = cones[:, 3] >= self.params['min_confidence']
        cones, cov, sid = cones[keep], cov[keep], sid[keep]
        use = self.params['use_covariance'] and have_cov
        names = [n for n, c in cones_by_sensor.items() if len(c)]
        types = self.context.get('sensor_types', {})
        lidar_ids = {k for k, n in enumerate(names) if types.get(n) == 'lidar'}
        conf_r = self.params['lidar_confirm_range']
        if lidar_ids and conf_r > 0:
            out = self._confirmed(cones, cov, sid, use, lidar_ids, conf_r)
        else:
            out = _merge(cones, cov, self.params['gate'], use, sid, self.params['chi2'] if use else None)
        if not use and not lidar_ids:
            out = _merge(cones, cov, min(self.params['gate'], 0.8), False, sid)
        if self.params['require_color'] and len(out):
            out = out[out[:, 2] != UNKNOWN]
        return out


    def _confirmed(self, cones, cov, sid, use, lidar_ids, conf_r):
        """Merge, then drop camera-only cones close enough that a LiDAR should have seen them."""
        group = _cluster_by_distance(cones[:, :2], self.params['gate'], sid,
                                     cov if use else None, self.params['chi2'] if use else None)
        has_lidar = np.zeros(group.max() + 1, dtype=bool)
        is_lidar = np.isin(sid, list(lidar_ids))
        has_lidar[group[is_lidar]] = True
        r = np.hypot(cones[:, 0], cones[:, 1])
        keep = has_lidar[group] | (r > conf_r)
        return _merge(cones[keep], cov[keep] if cov is not None else None, self.params['gate'], use,
                      sid[keep], self.params['chi2'] if use else None)


@register('fusion', 'priority', label='Primary sensor + fill-in',
          params=[P('primary', '', help='sensor name to trust first (blank = first sensor)'),
                  P('fill_radius', 1.0, 0.1, 3.0, 0.05, 'm', 'add other sensors\' cones only this far from primary ones')])
class PriorityFusion(Component):
    """Trust one sensor; add the others' cones only where it saw nothing."""

    def fuse(self, cones_by_sensor, cov_by_sensor):
        names = list(cones_by_sensor)
        if not names:
            return empty_cones()
        primary = self.params['primary'] if self.params['primary'] in cones_by_sensor else names[0]
        base = cones_by_sensor[primary]
        out = [base] if len(base) else []
        for n in names:
            if n == primary or not len(cones_by_sensor[n]):
                continue
            extra = cones_by_sensor[n]
            if len(base):
                d = np.min(np.hypot(extra[:, None, 0] - base[None, :, 0], extra[:, None, 1] - base[None, :, 1]), axis=1)
                extra = extra[d > self.params['fill_radius']]
            if len(extra):
                out.append(extra)
        return np.vstack(out) if out else empty_cones()
