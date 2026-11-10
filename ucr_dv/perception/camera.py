"""
Camera cone detection: BGR image -> cones (car frame, coloured).

`camera_color_blob` is a classical detector - a stand-in for a YOLO model
that runs anywhere with no weights and no GPU:
    1. HSV thresholds per cone colour (blue, yellow, orange)
    2. connected components (scipy.ndimage when present)
    3. merge blobs split by the cone's stripes (same colour, stacked)
    4. shape tests (aspect ratio, minimum box height in pixels)
    5. ranging: depth image (stereo/depth sensor) -> ground-plane
       intersection of the box's bottom centre -> box-size fallback
    6. orange vs big orange from the box's physical height

Swapping in a learned detector later means registering another 'camera'
algorithm with the same `detect(image) -> cones` contract.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.registry import Component, P, register
from ..core.types import BIG_ORANGE, BLUE, ORANGE, YELLOW, empty_cones

try:
    from scipy import ndimage
    HAVE_NDIMAGE = True
except Exception:                          # pragma: no cover
    HAVE_NDIMAGE = False

CONE_H, BIG_H = 0.325, 0.505


def bgr_to_hsv(img):
    """uint8 BGR -> (h deg 0..360, s 0..1, v 0..1), vectorised."""
    f = img.astype(np.float32) / 255.0
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    dd = np.where(d > 1e-6, d, 1.0)
    h = np.where(mx == r, ((g - b) / dd) % 6.0,
                 np.where(mx == g, (b - r) / dd + 2.0, (r - g) / dd + 4.0)) * 60.0
    h = np.where(d > 1e-6, h, 0.0)
    s = np.where(mx > 1e-6, d / np.where(mx > 1e-6, mx, 1.0), 0.0)
    return h, s, mx


def _label(mask):
    if HAVE_NDIMAGE:
        lab, n = ndimage.label(mask)
        return lab, n
    from .lidar import cluster_grid                      # slow fallback
    ys, xs = np.nonzero(mask)
    lab = np.zeros(mask.shape, dtype=np.int64)
    if len(xs):
        l = cluster_grid(np.column_stack([ys, xs]).astype(float), 1.0) + 1
        lab[ys, xs] = l
        return lab, int(l.max())
    return lab, 0


def _boxes(lab, n):
    if n == 0:
        return []
    if HAVE_NDIMAGE:
        sl = ndimage.find_objects(lab)
        counts = np.bincount(lab.ravel(), minlength=n + 1)
        return [(s[1].start, s[0].start, s[1].stop, s[0].stop, int(counts[i + 1]))
                for i, s in enumerate(sl) if s is not None]
    out = []
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab == k)
        out.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1, len(xs)))
    return out


def _merge_stacked(boxes, gap_ratio=0.6):
    """Merge boxes that overlap horizontally and are close vertically (stripes)."""
    boxes = sorted(boxes, key=lambda b: b[1])
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        out = []
        used = [False] * len(boxes)
        for i, a in enumerate(boxes):
            if used[i]:
                continue
            x0, y0, x1, y1, n = a
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                bx0, by0, bx1, by1, bn = boxes[j]
                overlap = min(x1, bx1) - max(x0, bx0)
                narrow = max(1, min(x1 - x0, bx1 - bx0))
                gap = max(by0 - y1, y0 - by1)
                height = max(y1, by1) - min(y0, by0)
                if overlap > 0.4 * narrow and gap < gap_ratio * height * 0.5 + 1:
                    x0, y0, x1, y1, n = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1), n + bn
                    used[j] = True
                    merged = True
            out.append((x0, y0, x1, y1, n))
        boxes = out
    return boxes


def _rot(roll, pitch, yaw):
    """Same composition as the fsds_lite renderer: Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
    rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def pixel_ray(u, v, intr, mount):
    """Unit ray (car frame) through pixel (u, v), from the camera origin."""
    fx, fy, cx, cy = intr
    d_cam = np.array([1.0, -(u - cx) / fx, -(v - cy) / fy])      # x fwd, y left, z up
    r = _rot(mount[3], mount[4], mount[5])
    d = r @ d_cam
    return d / np.linalg.norm(d)


def project_points(xyz_car, intr, mount):
    """Car-frame points -> pixel (u, v) and depth along the optical axis."""
    fx, fy, cx, cy = intr
    r = _rot(mount[3], mount[4], mount[5])
    rel = (np.asarray(xyz_car, dtype=float).reshape(-1, 3) - np.array(mount[:3])) @ r   # into camera axes
    depth = rel[:, 0]
    safe = np.where(depth > 1e-3, depth, 1e-3)
    u = cx - fx * rel[:, 1] / safe
    v = cy - fy * rel[:, 2] / safe
    return u, v, depth


COLOR_BANDS = {
    # name: (hue lo, hue hi, class)
    'orange': (8.0, 32.0, ORANGE),
    'yellow': (38.0, 70.0, YELLOW),
    'blue': (200.0, 250.0, BLUE),
}


@register('camera', 'camera_color_blob', label='Colour-blob detector (classical)',
          params=[
              P('downscale', 1, 1, 4, 1, 'x', 'process the image at 1/n resolution'),
              P('min_saturation', 0.45, 0.1, 1.0, 0.05, '', 'HSV saturation threshold'),
              P('min_value', 0.20, 0.0, 1.0, 0.05, '', 'HSV value (brightness) threshold'),
              P('min_box_px', 8.0, 2.0, 60.0, 1.0, 'px', 'smallest box height (full resolution) accepted'),
              P('min_aspect', 0.7, 0.2, 3.0, 0.1, 'h/w', 'smallest box height/width ratio'),
              P('max_aspect', 3.5, 1.0, 8.0, 0.1, 'h/w', 'largest box height/width ratio'),
              P('ranging', 'auto', choices=['auto', 'ground_plane', 'box_size', 'depth'],
                help='how a box becomes a range'),
              P('max_range', 40.0, 5.0, 100.0, 1.0, 'm', 'drop detections further than this'),
          ],
          refs=['IIT Bombay 2024 (mono/stereo ranging)', 'stand-in for YOLOv5/v8/v11'])
class ColorBlobDetector(Component):
    """HSV colour segmentation + connected components + geometric ranging."""

    def detect(self, frame):
        p = self.params
        img = frame.image
        k = max(1, int(p['downscale']))
        small = img[::k, ::k] if k > 1 else img
        h, s, v = bgr_to_hsv(small)
        sat_ok = (s >= p['min_saturation']) & (v >= p['min_value'])
        fx, fy, cx, cy = frame.intrinsics
        boxes_out, cones = [], []
        for name, (lo, hi, cls) in COLOR_BANDS.items():
            mask = sat_ok & (h >= lo) & (h <= hi)
            if not mask.any():
                continue
            lab, n = _label(mask)
            boxes = _merge_stacked(_boxes(lab, n))
            for (x0, y0, x1, y1, npx) in boxes:
                x0, y0, x1, y1 = x0 * k, y0 * k, x1 * k, y1 * k
                bw, bh = x1 - x0, y1 - y0
                if bh < p['min_box_px'] or bw < 2:
                    continue
                aspect = bh / float(bw)
                if aspect < p['min_aspect'] or aspect > p['max_aspect']:
                    continue
                # a cone's base is always below the horizon: reject sky-side blobs
                horizon = cy - fy * math.tan(frame.mount[4])
                if y1 < horizon:
                    continue
                cone = self._range(frame, x0, y0, x1, y1, cls)
                if cone is None:
                    continue
                if math.hypot(cone[0], cone[1]) > p['max_range']:
                    continue
                cones.append(cone)
                boxes_out.append((x0, y0, x1, y1, int(cone[2])))
        arr = np.array(cones).reshape(-1, 4) if cones else empty_cones()
        return arr, {'boxes': boxes_out}

    def _range(self, frame, x0, y0, x1, y1, cls):
        fx, fy, cx, cy = frame.intrinsics
        mount = frame.mount
        u = 0.5 * (x0 + x1)
        method = self.params['ranging']
        depth = frame.depth if frame.depth is not None else (frame.image if frame.is_depth else None)
        origin = np.array(mount[:3])
        r_box = None
        point = None
        if method in ('auto', 'depth') and depth is not None:
            ys = slice(int(y0 + 0.3 * (y1 - y0)), max(int(y0 + 0.3 * (y1 - y0)) + 1, int(y1 - 0.1 * (y1 - y0))))
            xs = slice(int(x0 + 0.3 * (x1 - x0)), max(int(x0 + 0.3 * (x1 - x0)) + 1, int(x1 - 0.3 * (x1 - x0))))
            patch = depth[ys, xs]
            patch = patch[np.isfinite(patch)]
            if len(patch):
                d_axis = float(np.median(patch))
                ray = pixel_ray(u, 0.5 * (y0 + y1), frame.intrinsics, mount)
                axis = _rot(mount[3], mount[4], mount[5])[:, 0]
                t = d_axis / max(float(np.dot(ray, axis)), 1e-3)
                point = origin + ray * t
        if point is None and method in ('auto', 'ground_plane', 'depth'):
            ray = pixel_ray(u, y1, frame.intrinsics, mount)
            if ray[2] < -1e-3:
                t = -origin[2] / ray[2]
                point = origin + ray * t
        if point is None:
            # box size: assume a small cone
            r_box = fy * CONE_H / max(y1 - y0, 1.0)
            ray = pixel_ray(u, 0.5 * (y0 + y1), frame.intrinsics, mount)
            point = origin + ray * r_box
        rng = math.hypot(point[0] - origin[0], point[1] - origin[1])
        # physical box height tells big orange from small orange
        if cls == ORANGE:
            phys_h = (y1 - y0) * rng / fy
            if phys_h > 0.5 * (CONE_H + BIG_H):
                cls = BIG_ORANGE
        # bottom-centre / mid-depth points sit on the cone's near face: step back to its axis
        xy = np.array(point[:2])
        dirn = xy - origin[:2]
        nrm = np.linalg.norm(dirn)
        if nrm > 1e-6:
            xy = xy + dirn / nrm * 0.08
        conf = float(np.clip((y1 - y0) / (3.0 * self.params['min_box_px']), 0.3, 1.0))
        return (xy[0], xy[1], cls, conf)
