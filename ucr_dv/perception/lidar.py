"""
LiDAR cone detection: point cloud (car frame) -> cones (car frame, UNKNOWN colour).

All detectors share the same three steps, each with selectable methods:
    1. ground removal   plane_fit (iterative least squares, from fsds_ds_stack)
                        ransac    (random 3-point planes, best inlier count;
                                   IIT Bombay / Heriot-Watt)
                        flat      (z threshold; only for perfectly flat sims)
    2. clustering       grid      (8-connected occupied cells; fsds_ds_stack)
                        dbscan    (eps / min_samples; Heriot-Watt 2025 uses 0.5 / 5)
    3. cone validation  point count, cluster width, highest return - rejects
                        tyre stacks, barriers, poles and timekeeping boxes.
"""
from __future__ import annotations

import numpy as np

from ..core.registry import Component, P, register
from ..core.types import BIG_ORANGE, UNKNOWN, empty_cones

try:                                      # optional speed-ups
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    HAVE_SCIPY = True
except Exception:                         # pragma: no cover - numpy fallback
    HAVE_SCIPY = False

COMMON = [
    P('max_range', 30.0, 5.0, 120.0, 1.0, 'm', 'ignore returns further than this'),
    P('ground', 'plane_fit', choices=['plane_fit', 'ransac', 'flat'],
      help='ground removal method'),
    P('min_height', 0.04, 0.0, 0.3, 0.01, 'm', 'keep points this far above the ground'),
    P('max_height', 0.60, 0.3, 1.5, 0.05, 'm', 'returns above this mark a cluster as not-a-cone'),
    P('min_cluster_points', 3, 1, 50, 1, 'pts', 'smallest accepted cluster'),
    P('max_cone_width', 0.5, 0.2, 1.5, 0.05, 'm', 'cluster bounding-box diagonal limit'),
    P('big_cone_height', 0.42, 0.3, 0.6, 0.01, 'm', 'clusters taller than this are big orange cones '
                                                      '(0.505 m vs 0.325 m) - LiDAR can see that'),
]


def fit_ground_plane(xyz, z0=0.0, bands=(0.30, 0.10, 0.05)):
    """Plane z = a x + b y + c through ground returns (car frame, ground near z0)."""
    plane = np.array([0.0, 0.0, z0])
    for band in bands:
        resid = xyz[:, 2] - (xyz[:, 0] * plane[0] + xyz[:, 1] * plane[1] + plane[2])
        g = xyz[np.abs(resid) < band]
        if len(g) < 20:
            break
        A = np.column_stack([g[:, 0], g[:, 1], np.ones(len(g))])
        plane = np.linalg.lstsq(A, g[:, 2], rcond=None)[0]
    return plane


def ransac_ground(xyz, rng, iters=40, tol=0.04, z0=0.0):
    """Best-supported plane among random 3-point samples of the low points."""
    low = xyz[xyz[:, 2] < z0 + 0.3]
    if len(low) < 20:
        return np.array([0.0, 0.0, z0])
    best, best_n = np.array([0.0, 0.0, z0]), -1
    idx = rng.integers(0, len(low), size=(iters, 3))
    for i, j, k in idx:
        p1, p2, p3 = low[i], low[j], low[k]
        n = np.cross(p2 - p1, p3 - p1)
        if abs(n[2]) < 1e-6:
            continue
        a, b = -n[0] / n[2], -n[1] / n[2]
        if abs(a) > 0.2 or abs(b) > 0.2:          # > ~11 deg: not the ground
            continue
        c = p1[2] - a * p1[0] - b * p1[1]
        count = int(np.sum(np.abs(low[:, 2] - (a * low[:, 0] + b * low[:, 1] + c)) < tol))
        if count > best_n:
            best, best_n = np.array([a, b, c]), count
    # refine on the inliers
    inl = low[np.abs(low[:, 2] - (best[0] * low[:, 0] + best[1] * low[:, 1] + best[2])) < tol]
    if len(inl) >= 20:
        A = np.column_stack([inl[:, 0], inl[:, 1], np.ones(len(inl))])
        best = np.linalg.lstsq(A, inl[:, 2], rcond=None)[0]
    return best


def cluster_grid(xy, cell):
    """Connected components of occupied grid cells (8-connected)."""
    if len(xy) == 0:
        return np.zeros(0, dtype=np.int64)
    ij = np.floor(xy / cell).astype(np.int64)
    cells, inverse = np.unique(ij, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(cells)}
    label = np.full(len(cells), -1, dtype=np.int64)
    n = 0
    for start in range(len(cells)):
        if label[start] >= 0:
            continue
        label[start] = n
        stack = [start]
        while stack:
            k = stack.pop()
            ci, cj = int(cells[k][0]), int(cells[k][1])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = index.get((ci + di, cj + dj))
                    if nb is not None and label[nb] < 0:
                        label[nb] = n
                        stack.append(nb)
        n += 1
    return label[inverse]


def dbscan(xy, eps, min_samples):
    """DBSCAN labels (-1 = noise). KD-tree + sparse graph when scipy is present."""
    n = len(xy)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if HAVE_SCIPY:
        tree = cKDTree(xy)
        pairs = tree.query_pairs(eps, output_type='ndarray')
        deg = np.bincount(pairs.reshape(-1), minlength=n) + 1 if len(pairs) else np.ones(n, dtype=int)
        core = deg >= min_samples
        # connect core-core pairs; border points join a neighbouring core's cluster
        if len(pairs):
            cc = core[pairs[:, 0]] & core[pairs[:, 1]]
            p = pairs[cc]
            g = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
            _, lab = connected_components(g, directed=False)
        else:
            lab = np.arange(n)
        labels = np.where(core, lab, -1)
        if len(pairs):
            for a, b in ((0, 1), (1, 0)):
                m = core[pairs[:, a]] & ~core[pairs[:, b]]
                labels[pairs[m, b]] = lab[pairs[m, a]]
        return labels
    # fallback: grid-hashed neighbour search
    lab = cluster_grid(xy, eps)
    counts = np.bincount(lab)
    return np.where(counts[lab] >= min_samples, lab, -1)


def validate_clusters(xyz, h, labels, p):
    """Cluster -> cone centre, applying the shape tests. Returns (cones (M,4), stats)."""
    ok = labels >= 0
    if not ok.any():
        return empty_cones(), {'clusters': 0, 'rejected': 0}
    labels, xyz, h = labels[ok], xyz[ok], h[ok]
    order = np.argsort(labels, kind='stable')
    labels, xyz, h = labels[order], xyz[order], h[order]
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:], len(labels)]
    out, rejected = [], 0
    for s, e in zip(starts, ends):
        if e - s < p['min_cluster_points'] or h[s:e].max() > p['max_height']:
            rejected += 1
            continue
        pts = xyz[s:e, :2]
        span = pts.max(axis=0) - pts.min(axis=0)
        if np.hypot(span[0], span[1]) > p['max_cone_width']:
            rejected += 1
            continue
        c = pts.mean(axis=0)
        # only the near face reflects: push the centre ~ one radius away from the sensor
        r = np.hypot(c[0], c[1])
        if r > 1e-3:
            c = c + c / r * 0.06
        conf = min(1.0, (e - s) / (3.0 * p['min_cluster_points']))
        col = BIG_ORANGE if h[s:e].max() > p['big_cone_height'] else UNKNOWN
        out.append((c[0], c[1], col, conf))
    cones = np.array(out).reshape(-1, 4) if out else empty_cones()
    return cones, {'clusters': len(starts), 'rejected': rejected}


class _LidarBase(Component):
    def __init__(self, params, **context):
        super(_LidarBase, self).__init__(params, **context)
        self.rng = np.random.default_rng(context.get('seed', 0))

    def preprocess(self, cloud):
        xyz = np.asarray(cloud.points, dtype=np.float64).reshape(-1, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        xyz = xyz[np.hypot(xyz[:, 0], xyz[:, 1]) < self.params['max_range']]
        if not len(xyz):
            return xyz, xyz[:, 2]
        g = self.params['ground']
        if g == 'ransac':
            a, b, c = ransac_ground(xyz, self.rng)
        elif g == 'plane_fit':
            a, b, c = fit_ground_plane(xyz)
        else:
            a = b = c = 0.0
        h = xyz[:, 2] - (a * xyz[:, 0] + b * xyz[:, 1] + c)
        keep = h > self.params['min_height']
        return xyz[keep], h[keep]

    def detect(self, cloud):
        xyz, h = self.preprocess(cloud)
        if not len(xyz):
            return empty_cones(), {'points': 0}
        labels = self.cluster(xyz[:, :2])
        cones, stats = validate_clusters(xyz, h, labels, self.params)
        stats['points'] = int(len(xyz))
        return cones, stats


@register('lidar', 'lidar_grid', label='Grid clustering (fsds_ds_stack)',
          params=COMMON + [P('cell_size', 0.25, 0.05, 1.0, 0.05, 'm', 'grid cell for connected components')],
          refs=['fsds_ds_stack cone_detector', 'KIT 2022 (2D clustering)'])
class GridLidarDetector(_LidarBase):
    """Ground plane + occupied-grid connected components + shape tests. Linear time."""

    def cluster(self, xy):
        return cluster_grid(xy, self.params['cell_size'])


@register('lidar', 'lidar_dbscan', label='DBSCAN clustering',
          params=COMMON + [P('eps', 0.5, 0.05, 2.0, 0.05, 'm', 'neighbourhood radius'),
                           P('min_samples', 3, 1, 30, 1, 'pts', 'points to be a core point')],
          refs=['Heriot-Watt 2025 (eps 0.5, min 5)', 'IIT Bombay 2024 (RANSAC + DBSCAN)'])
class DbscanLidarDetector(_LidarBase):
    """RANSAC/plane ground removal, then DBSCAN. Robust to uneven point density."""

    def cluster(self, xy):
        return dbscan(xy, self.params['eps'], self.params['min_samples'])
