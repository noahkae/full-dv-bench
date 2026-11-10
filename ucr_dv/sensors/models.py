"""
Detection-level sensor models: ground-truth cones -> plausible detections.

These make the sensor choice matter even on the 2D bench: a 4-channel LiDAR
really does lose cones past ~10 m, a wide camera really does mis-range far
cones, two sensors really do cover more than one. Each model is physics-
first and cheap (vectorised numpy, no ray casting):

LiDAR  - points on a cone = (laser rings crossing the cone's height at that
         range) x (azimuth columns across its width). Detected if that is at
         least the clustering algorithm's `min_cluster_points`. Range noise is
         the datasheet's; the centroid sits slightly towards the sensor (only
         the near face is hit). No colour: LiDAR alone reports UNKNOWN.
Camera - box height in pixels = f * h / r. Detected above the detector's
         `min_box_px`; colour right with `color_accuracy` (worse on tiny boxes).
         Bearing error = pixel_noise / f. Mono range from box size:
         sigma_r = r * pixel_noise / box_px  (grows with r^2).
Stereo - as camera, but range from disparity: sigma_r = r^2 * d_noise / (f B).
Both   - horizontal + vertical FOV (a high camera can't see cones right under
         its nose), max range, occlusion by nearer cones, false positives.

`timeless_classic` reproduces the original ucr-dv-timeless detector model
(including its noise parameters), with its x/y-swap bug fixed.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..core.types import BIG_ORANGE, UNKNOWN, ConeDetections, empty_cones
from .rig import SensorSpec

CONE_H = 0.325
BIG_H = 0.505
CONE_W = 0.228          # base width; the visible mid-height width is about half


def _sensor_frame(cones_car, spec: SensorSpec):
    """Cones relative to the sensor origin, rotated by its yaw: (dx, dy, r, bearing)."""
    m = spec.mount
    dx = cones_car[:, 0] - m.x
    dy = cones_car[:, 1] - m.y
    yaw = math.radians(m.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    sx, sy = c * dx + s * dy, -s * dx + c * dy
    r = np.hypot(sx, sy)
    bearing = np.arctan2(sy, sx)
    return sx, sy, r, bearing


def _occluded(r, bearing, half_ang, strength=1.0):
    """True where a nearer cone covers this cone's bearing (from the sensor)."""
    n = len(r)
    if n < 2:
        return np.zeros(n, dtype=bool)
    db = np.abs((bearing[None, :] - bearing[:, None] + math.pi) % (2 * math.pi) - math.pi)
    # row i = occluded cone, column j = candidate occluder
    nearer = r[None, :] < r[:, None] - 0.05
    covers = db < (half_ang[None, :] * strength)
    return np.any(nearer & covers, axis=1)


def _drop_occluded(ok, r, bearing, strength):
    idx = np.nonzero(ok)[0]
    if len(idx) > 1:
        occ = _occluded(r[idx], bearing[idx], (CONE_W / 2.0) / np.maximum(r[idx], 0.3), strength)
        ok = ok.copy()
        ok[idx[occ]] = False
    return ok


def _in_hfov(bearing, hfov_deg):
    if hfov_deg >= 359.9:
        return np.ones_like(bearing, dtype=bool)
    return np.abs(bearing) <= math.radians(hfov_deg) / 2.0


def _false_positives(rng, rate, max_range, hfov_deg, spec, color=UNKNOWN):
    k = rng.poisson(rate)
    if not k:
        return empty_cones()
    r = rng.uniform(2.0, max(3.0, min(max_range, 20.0)), k)
    half = math.radians(min(hfov_deg, 359.9)) / 2.0
    b = rng.uniform(-half, half, k) + math.radians(spec.mount.yaw)
    out = np.empty((k, 4))
    out[:, 0] = spec.mount.x + r * np.cos(b)
    out[:, 1] = spec.mount.y + r * np.sin(b)
    out[:, 2] = color if color is not None else rng.integers(0, 4, k)
    out[:, 3] = 0.3
    return out


class DetectionModel(object):
    """Base: `detect(cones_car, heights, rng, t, algo_params)` -> ConeDetections."""

    def __init__(self, spec: SensorSpec):
        self.spec = spec

    def visible(self, cones_car, heights):
        """Mask of cones geometrically visible (FOV, range, occlusion) - for the rig preview."""
        raise NotImplementedError

    def detect(self, cones_car, heights, rng, t, algo_params=None) -> ConeDetections:
        raise NotImplementedError


class LidarModel(DetectionModel):
    def expected_points(self, r, heights):
        p = self.spec
        z = p.mount.z
        ch = int(p.p('channels'))
        lo, hi = math.radians(p.p('vfov_lower')), math.radians(p.p('vfov_upper'))
        el = np.linspace(hi, lo, ch) if ch > 1 else np.array([hi])
        r = np.maximum(r, 0.3)
        # elevation span of the cone body seen from the sensor
        e0 = np.arctan2(-z, r)                   # cone base
        e1 = np.arctan2(heights - z, r)          # cone apex
        rows = ((el[None, :] >= e0[:, None]) & (el[None, :] <= e1[:, None])).sum(axis=1)
        width = CONE_W * 0.6                     # average visible width over the height
        cols = width / r / math.radians(float(p.p('h_res_deg')))
        return rows * np.maximum(cols, 0.0)

    def visible(self, cones_car, heights):
        sx, sy, r, b = _sensor_frame(cones_car, self.spec)
        ok = _in_hfov(b, self.spec.p('hfov')) & (r <= self.spec.p('max_range')) & (r > 0.3)
        return _drop_occluded(ok, r, b, strength=0.8)

    def detect(self, cones_car, heights, rng, t, algo_params=None) -> ConeDetections:
        spec = self.spec
        algo_params = algo_params or {}
        min_pts = float(algo_params.get('min_cluster_points', 3))
        sx, sy, r, b = _sensor_frame(cones_car, spec)
        vis = self.visible(cones_car, heights)
        n_exp = self.expected_points(r, heights)
        n = rng.poisson(np.maximum(n_exp, 0.0))
        hit = vis & (n >= min_pts)
        r_hit, b_hit = r[hit], b[hit]
        sigma_r = float(spec.p('range_noise')) + 0.01
        # the near face is what reflects: centroid biased ~1/3 radius towards the sensor
        r_meas = r_hit - 0.04 + rng.normal(0.0, sigma_r, len(r_hit))
        ang_sigma = math.radians(float(spec.p('h_res_deg'))) / 2.0
        b_meas = b_hit + rng.normal(0.0, ang_sigma, len(b_hit))
        out = self._to_car(r_meas, b_meas, conf=np.clip(n[hit] / (3.0 * min_pts), 0.2, 1.0))
        # no colour from LiDAR - but a big orange cone is tall enough to tell by
        # its highest return, if a ring reaches above the small-cone height
        out[:, 2] = UNKNOWN
        big_h = float(algo_params.get('big_cone_height', 0.42))
        z, ch = spec.mount.z, int(spec.p('channels'))
        lo, hi = math.radians(spec.p('vfov_lower')), math.radians(spec.p('vfov_upper'))
        el = np.linspace(hi, lo, ch) if ch > 1 else np.array([hi])
        hh = heights[hit]
        e_lo = np.arctan2(big_h - z, r_hit)
        e_hi = np.arctan2(hh - z, r_hit)
        sees_top = ((el[None, :] >= e_lo[:, None]) & (el[None, :] <= e_hi[:, None])).any(axis=1)
        out[sees_top & (hh > big_h), 2] = BIG_ORANGE
        fp = _false_positives(rng, float(spec.p('false_positive_rate')), float(spec.p('max_range')),
                              float(spec.p('hfov')), spec)
        cones = np.vstack([out, fp])
        cov = self._cov(r_meas, b_meas, sigma_r, ang_sigma, len(fp))
        return ConeDetections(spec.name, t, cones, cov)

    def _to_car(self, r, b, conf):
        m = self.spec.mount
        yaw = math.radians(m.yaw)
        out = np.empty((len(r), 4))
        out[:, 0] = m.x + r * np.cos(b + yaw)
        out[:, 1] = m.y + r * np.sin(b + yaw)
        out[:, 3] = conf
        return out

    def _cov(self, r, b, sigma_r, sigma_b, n_fp):
        """Per-cone 2x2 covariance from (range, bearing) noise, car frame."""
        yaw = math.radians(self.spec.mount.yaw)
        sr = np.broadcast_to(np.asarray(sigma_r, dtype=float), r.shape)
        cov = np.zeros((len(r) + n_fp, 2, 2))
        c, s = np.cos(b + yaw), np.sin(b + yaw)
        vr, vt = sr ** 2, (r * sigma_b) ** 2
        cov[:len(r), 0, 0] = c * c * vr + s * s * vt
        cov[:len(r), 1, 1] = s * s * vr + c * c * vt
        cov[:len(r), 0, 1] = cov[:len(r), 1, 0] = c * s * (vr - vt)
        cov[len(r):] = np.eye(2) * 1.0
        return cov


class CameraModel(LidarModel):
    """Mono camera (and the base for stereo)."""

    def box_px(self, r, heights, algo_scale=1.0):
        return self.spec.focal_px() * heights / np.maximum(r, 0.3) * algo_scale

    def visible(self, cones_car, heights):
        spec = self.spec
        sx, sy, r, b = _sensor_frame(cones_car, spec)
        m = spec.mount
        ok = _in_hfov(b, spec.p('hfov')) & (r <= spec.p('max_range')) & (sx > 0.2)
        # vertical FOV: the cone's base must be inside the bottom edge of the image
        half_v = math.radians(spec.vfov()) / 2.0
        depression = np.arctan2(m.z, np.maximum(r, 0.3)) - math.radians(m.pitch)
        ok &= depression <= half_v
        # cameras mounted high look over near cones; low ones get blocked
        return _drop_occluded(ok, r, b, strength=0.5) if m.z < 0.6 else ok

    def range_sigma(self, r, px):
        """Mono ranging: the better of box-size (r * sigma_px / box_px) and
        ground-plane intersection of the box bottom (r^2 sigma_theta / h, with
        sigma_theta from pixel noise plus ~0.1 deg of unmodelled body pitch)."""
        spec = self.spec
        box = r * float(spec.p('pixel_noise')) / np.maximum(px, 1.0)
        h = max(spec.mount.z, 0.05)
        s_theta = math.hypot(float(spec.p('pixel_noise')) / spec.focal_px(), math.radians(0.1))
        ground = r * r * s_theta / h
        return np.minimum(box, ground) + 0.02

    def detect(self, cones_car, heights, rng, t, algo_params=None) -> ConeDetections:
        spec = self.spec
        algo_params = algo_params or {}
        min_px = float(algo_params.get('min_box_px', spec.p('min_box_px')))
        sx, sy, r, b = _sensor_frame(cones_car, spec)
        vis = self.visible(cones_car, heights)
        px = self.box_px(r, heights)
        # soft threshold: 50 % at min_px, ~100 % at 1.5x
        p_det = np.clip((px - 0.5 * min_px) / min_px, 0.0, 1.0) * 0.99
        hit = vis & (rng.random(len(r)) < p_det)
        r_h, b_h, px_h = r[hit], b[hit], px[hit]
        sig_r = self.range_sigma(r_h, px_h)
        f = spec.focal_px()
        sig_b = float(spec.p('pixel_noise')) / f
        r_meas = np.maximum(0.3, r_h + rng.normal(0.0, 1.0, len(r_h)) * sig_r)
        b_meas = b_h + rng.normal(0.0, sig_b, len(b_h))
        out = self._to_car(r_meas, b_meas, conf=np.clip(px_h / (3 * min_px), 0.3, 1.0))
        # colour: right with color_accuracy (lower on boxes < 2x the threshold)
        true_col = cones_car[hit, 2].astype(int)
        acc = np.where(px_h > 2 * min_px, float(spec.p('color_accuracy')),
                       float(spec.p('color_accuracy')) * 0.85)
        wrong = rng.random(len(true_col)) > acc
        flip = rng.integers(0, 4, len(true_col))
        col = np.where(wrong, flip, true_col)
        # big orange and orange are told apart by size: only flip between them rarely
        out[:, 2] = col
        fp = _false_positives(rng, float(spec.p('false_positive_rate')), float(spec.p('max_range')),
                              float(spec.p('hfov')), spec, color=None)
        cones = np.vstack([out, fp])
        cov = self._cov(r_meas, b_meas, sig_r, sig_b, len(fp))
        return ConeDetections(spec.name, t, cones, cov)


class StereoModel(CameraModel):
    def range_sigma(self, r, px):
        f, base = self.spec.focal_px(), float(self.spec.p('baseline'))
        return r * r * float(self.spec.p('disparity_noise')) / max(f * base, 1e-6) + 0.01


class TimelessClassicModel(DetectionModel):
    """The ucr-dv-timeless detector: GT + skew-normal/normal noise, class flips,
    misses and spurious cones, inside a 110 deg wedge and a 21 x 20 m box."""

    def visible(self, cones_car, heights):
        p = self.spec.p
        x, y = cones_car[:, 0], cones_car[:, 1]
        ok = np.hypot(x, y) < p('max_dist')
        t = math.tan(math.radians(p('fov')) / 2.0)
        ok &= (y < t * x) & (y > -t * x)
        ok &= (x >= p('box_back')) & (x <= p('box_forward')) & (np.abs(y) <= p('box_half_width'))
        return ok

    def detect(self, cones_car, heights, rng, t, algo_params=None) -> ConeDetections:
        p = self.spec.p
        c = cones_car[self.visible(cones_car, heights)].copy()
        n = len(c)
        a = float(p('skew_x'))
        delta = a / math.sqrt(1.0 + a * a)
        u0, u1 = np.abs(rng.normal(size=n)), rng.normal(size=n)
        c[:, 0] += float(p('sigma_x')) * (delta * u0 + math.sqrt(1.0 - delta * delta) * u1)
        c[:, 1] += rng.normal(0.0, float(p('sigma_y')), n)
        flip = rng.random(n) < float(p('class_flip'))
        c[flip, 2] = rng.integers(0, 4, int(flip.sum()))
        fn = np.where(c[:, 2] == BIG_ORANGE, float(p('false_negative_big')), float(p('false_negative')))
        c = c[rng.random(n) >= fn]
        extra = []
        while rng.random() < float(p('spurious_prob')):
            extra.append([rng.uniform(p('box_back'), p('box_forward')),
                          rng.uniform(-p('box_half_width'), p('box_half_width')), rng.integers(0, 4), 0.5])
        if extra:
            c = np.vstack([c, np.array(extra)])
        c[:, 3] = np.where(c[:, 3] > 0, c[:, 3], 1.0)
        return ConeDetections(self.spec.name, t, c, None)


MODELS = {'lidar': LidarModel, 'camera': CameraModel, 'stereo': StereoModel,
          'timeless_classic': TimelessClassicModel}


def make_model(spec: SensorSpec) -> Optional[DetectionModel]:
    cls = MODELS.get(spec.type)
    return cls(spec) if cls else None


def coverage_grid(spec: SensorSpec, extent=30.0, step=0.5, height=CONE_H, algo_params=None):
    """Detection probability of an isolated cone on a grid around the car (rig preview).

    Returns (xs, ys, P) with P shaped (len(ys), len(xs)).
    """
    xs = np.arange(-extent / 2, extent + step, step)
    ys = np.arange(-extent, extent + step, step)
    gx, gy = np.meshgrid(xs, ys)
    cones = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size), np.ones(gx.size)])
    model = make_model(spec)
    if model is None:
        return xs, ys, np.zeros(gx.shape)
    h = np.full(len(cones), height)
    sx, sy, r, b = _sensor_frame(cones, spec)
    algo_params = algo_params or {}
    if isinstance(model, TimelessClassicModel):
        prob = model.visible(cones, h).astype(float)
    elif isinstance(model, CameraModel):
        min_px = float(algo_params.get('min_box_px', spec.p('min_box_px')))
        px = model.box_px(r, h)
        vis = _in_hfov(b, spec.p('hfov')) & (r <= spec.p('max_range')) & (sx > 0.2)
        half_v = math.radians(spec.vfov()) / 2.0
        vis &= (np.arctan2(spec.mount.z, np.maximum(r, 0.3)) - math.radians(spec.mount.pitch)) <= half_v
        prob = vis * np.clip((px - 0.5 * min_px) / min_px, 0.0, 1.0)
    else:
        min_pts = float(algo_params.get('min_cluster_points', 3))
        n = model.expected_points(r, h)
        vis = _in_hfov(b, spec.p('hfov')) & (r <= spec.p('max_range')) & (r > 0.3)
        # P(Poisson(n) >= min_pts) ~ smooth ramp
        prob = vis * np.clip((n - 0.5 * min_pts) / max(min_pts, 1.0), 0.0, 1.0)
    return xs, ys, prob.reshape(gx.shape)
