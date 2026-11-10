"""Path helpers shared by planners, speed profiles and controllers (car frame)."""
from __future__ import annotations

import math

import numpy as np


def arc_length(path):
    if len(path) < 2:
        return np.zeros(len(path))
    seg = np.hypot(*np.diff(path, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample(path, spacing=0.5):
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) < 2:
        return path
    s = arc_length(path)
    if s[-1] < 1e-6:
        return path[:1]
    n = max(2, int(math.ceil(s[-1] / spacing)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])])


def smooth(path, weight=0.5, passes=20, fix_start=True):
    """Laplacian smoothing that keeps the endpoints (cheap stand-in for a spline)."""
    p = np.asarray(path, dtype=float).copy()
    if len(p) < 3 or weight <= 0:
        return p
    for _ in range(passes):
        avg = 0.5 * (p[:-2] + p[2:])
        p[1:-1] += weight * (avg - p[1:-1])
    if not fix_start:
        p[0] += weight * (p[1] - p[0])
    return p


def curvature(path):
    """Signed curvature at each point (+ = turning left), from finite differences."""
    p = np.asarray(path, dtype=float)
    n = len(p)
    if n < 3:
        return np.zeros(n)
    s = arc_length(p)
    ds = np.gradient(s)
    ds = np.where(ds > 1e-6, ds, 1e-6)
    dx, dy = np.gradient(p[:, 0]) / ds, np.gradient(p[:, 1]) / ds
    ddx, ddy = np.gradient(dx) / ds, np.gradient(dy) / ds
    k = (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-9)
    return k


def point_at(path, dist):
    """Point `dist` metres along the path (clamped to its ends)."""
    if len(path) == 0:
        return np.array([0.0, 0.0])
    if len(path) == 1:
        return path[0].copy()
    s = arc_length(path)
    d = min(max(dist, 0.0), s[-1])
    return np.array([np.interp(d, s, path[:, 0]), np.interp(d, s, path[:, 1])])


def nearest(path, xy=(0.0, 0.0)):
    """(index, distance) of the path point nearest `xy`."""
    d = np.hypot(path[:, 0] - xy[0], path[:, 1] - xy[1])
    k = int(np.argmin(d))
    return k, float(d[k])


def anchor_to_car(path, max_gap=4.0):
    """Start the path at the car (origin) and drop points behind it."""
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if not len(path):
        return path
    k, _ = nearest(path)
    path = path[k:]
    if len(path) and path[0][0] < 0:
        path = path[1:] if len(path) > 1 else path
    if not len(path):
        return np.zeros((1, 2))
    if np.hypot(*path[0]) > 0.3:
        path = np.vstack([[0.0, 0.0], path])
    return path


def cross_track(path, xy=(0.0, 0.0)):
    """Signed distance of `xy` from the path (+ = point is left of the path) and the
    path tangent heading at the nearest point."""
    if len(path) < 2:
        return 0.0, 0.0
    k, _ = nearest(path, xy)
    j = min(k + 1, len(path) - 1)
    i = j - 1
    t = path[j] - path[i]
    heading = math.atan2(t[1], t[0])
    tn = t / max(np.linalg.norm(t), 1e-9)
    rel = np.asarray(xy) - path[i]
    return float(tn[0] * rel[1] - tn[1] * rel[0]), heading
