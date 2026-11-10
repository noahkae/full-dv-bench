"""
Procedural meshes as interleaved float32 vertex arrays: position(3) normal(3) color(3).
The car (a box) is in the body frame (ENU, origin on the ground under the CoG).
No asset files - everything is generated here so the sim stays self-contained.
"""
import math

import numpy as np

# FSG cone colours (body, stripe)
CONE_PAINT = {
    'blue': ((0.05, 0.20, 0.85), (0.95, 0.95, 0.95)),
    'yellow': ((0.98, 0.80, 0.05), (0.08, 0.08, 0.08)),
    'orange': ((1.00, 0.42, 0.02), (0.95, 0.95, 0.95)),
    'big_orange': ((1.00, 0.36, 0.02), (0.95, 0.95, 0.95)),
    'unknown': ((0.6, 0.6, 0.6), (0.3, 0.3, 0.3)),
}

CAR_PAINT = (0.78, 0.07, 0.10)
TYRE = (0.06, 0.06, 0.06)
TYRE_MARK = (0.30, 0.30, 0.30)
RIM = (0.62, 0.62, 0.66)
OBSTACLE_PAINT = {
    'barrier': (0.66, 0.66, 0.62),      # concrete
    'tyres': (0.07, 0.07, 0.07),
    'pole': (0.55, 0.57, 0.60),
    'timekeeping': (0.14, 0.14, 0.16),
}


def box(x0, x1, y0, y1, z0, z1, color):
    """Axis-aligned box, outward normals, 12 triangles."""
    c = color
    faces = [
        ((1, 0, 0), [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]),
        ((-1, 0, 0), [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)]),
        ((0, 1, 0), [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)]),
        ((0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),
        ((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),
        ((0, 0, -1), [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)]),
    ]
    out = []
    for n, (a, b, cc, d) in faces:
        for v in (a, b, cc, a, cc, d):
            out.append(v + n + c)
    return out


def rotated_box(x, y, yaw, half_x, half_y, height, color):
    """A box standing on the ground, turned by `yaw`, as triangle rows."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for v in box(-half_x, half_x, -half_y, half_y, 0.0, height, color):
        px, py, pz, nx, ny, nz = v[:6]
        out.append((x + c * px - s * py, y + s * px + c * py, pz,
                    c * nx - s * ny, s * nx + c * ny, nz) + tuple(v[6:]))
    return out


def cylinder(x, y, radius, height, color, segments=16):
    """Vertical cylinder standing on the ground (side + top), as triangle rows."""
    out = []
    for i in range(segments):
        a0, a1 = 2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments
        am = (a0 + a1) / 2
        n = (math.cos(am), math.sin(am), 0.0)
        p0 = (x + radius * math.cos(a0), y + radius * math.sin(a0))
        p1 = (x + radius * math.cos(a1), y + radius * math.sin(a1))
        quad = [(p0[0], p0[1], 0.0), (p1[0], p1[1], 0.0), (p1[0], p1[1], height), (p0[0], p0[1], height)]
        for k in (0, 1, 2, 0, 2, 3):
            out.append(quad[k] + n + color)
        for v in ((x, y, height), quad[3], quad[2]):
            out.append(v + (0.0, 0.0, 1.0) + color)
    return out


def obstacle_mesh(obstacles):
    """One static mesh (world coordinates) for every box and cylinder obstacle."""
    tris = []
    for o in obstacles:
        color = OBSTACLE_PAINT.get(o.label, (0.5, 0.5, 0.5))
        if o.kind == 'box':
            tris += rotated_box(o.x, o.y, o.yaw, o.half_x, o.half_y, o.height, color)
        elif o.kind == 'cylinder':
            tris += cylinder(o.x, o.y, o.radius, o.height, color)
    return np.array(tris, dtype=np.float32).reshape(-1, 9)


def car_box(params, height=0.5, clearance=0.12):
    """The car body as a plain box: the FSDS collision box footprint (length x
    width), from `clearance` up to `height`, centred on the pawn centre. The
    front face is a lighter shade so heading is readable at a glance."""
    hl, hw = params.length / 2.0, params.width / 2.0
    tris = box(-hl, hl, -hw, hw, clearance, height, CAR_PAINT)
    front = tuple(min(1.0, c * 1.6 + 0.25) for c in CAR_PAINT)
    return np.array([t[:6] + (front if t[3] > 0.5 else t[6:]) for t in tris], dtype=np.float32)


def wheel(radius, width, segments=24):
    """Tyre: a cylinder along y centred at the origin, with light tread marks
    every few segments so rotation is visible, and grey rims on both faces."""
    out = []
    hw = width / 2.0
    for i in range(segments):
        a0, a1 = 2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments
        col = TYRE_MARK if i % 6 == 0 else TYRE
        p0 = (radius * math.cos(a0), radius * math.sin(a0))
        p1 = (radius * math.cos(a1), radius * math.sin(a1))
        am = (a0 + a1) / 2
        n = (math.cos(am), 0.0, math.sin(am))
        q = [(p0[0], -hw, p0[1]), (p1[0], -hw, p1[1]), (p1[0], hw, p1[1]), (p0[0], hw, p0[1])]
        for v in (q[0], q[3], q[2], q[0], q[2], q[1]):
            out.append(v + n + col)
        for sgn in (-1.0, 1.0):                                   # side walls with a rim
            for r_in, r_out, c in ((0.0, 0.6 * radius, RIM), (0.6 * radius, radius, TYRE)):
                a = (r_in * math.cos(a0), sgn * hw, r_in * math.sin(a0))
                b = (r_out * math.cos(a0), sgn * hw, r_out * math.sin(a0))
                c1 = (r_out * math.cos(a1), sgn * hw, r_out * math.sin(a1))
                d = (r_in * math.cos(a1), sgn * hw, r_in * math.sin(a1))
                for v in (a, b, c1, a, c1, d):
                    out.append(v + (0.0, sgn, 0.0) + c)
    return np.array(out, dtype=np.float32)


def cone(segments=20, top_ratio=0.16):
    """Unit traffic cone: base radius 1 at z=0, height 1, plus a square base plate.

    Vertex layout: position(3) normal(3) h(1). h is the height fraction of
    the band's midpoint, the same for every vertex in the band, so stripe
    colours have crisp edges; h = -1 marks the base plate.
    """
    out = []
    slope = 1.0 - top_ratio                       # radius drop over unit height
    for i in range(segments):
        a0, a1 = 2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments
        am = (a0 + a1) / 2
        n = (math.cos(am), math.sin(am), slope)
        ln = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        n = (n[0] / ln, n[1] / ln, n[2] / ln)
        # split vertically so stripes have crisp edges
        bands = [0.0, 0.3, 0.42, 0.58, 0.7, 1.0]
        for z0, z1 in zip(bands[:-1], bands[1:]):
            r0, r1 = 1.0 - slope * z0, 1.0 - slope * z1
            mid = (z0 + z1) / 2.0
            v = [(r0 * math.cos(a0), r0 * math.sin(a0), z0),
                 (r0 * math.cos(a1), r0 * math.sin(a1), z0),
                 (r1 * math.cos(a1), r1 * math.sin(a1), z1),
                 (r1 * math.cos(a0), r1 * math.sin(a0), z1)]
            for k in (0, 1, 2, 0, 2, 3):
                out.append(v[k] + n + (mid,))
        # top cap
        for v in ((0.0, 0.0, 1.0), (top_ratio * math.cos(a0), top_ratio * math.sin(a0), 1.0),
                  (top_ratio * math.cos(a1), top_ratio * math.sin(a1), 1.0)):
            out.append(v + (0.0, 0.0, 1.0) + (0.99,))
    # square base plate, slightly wider than the cone, 2 cm thick
    s, t = 1.15, 0.06
    for tri in box(-s, s, -s, s, 0.0, t, (0.0, 0.0, 0.0)):
        out.append(tri[:6] + (-1.0,))
    return np.array(out, dtype=np.float32)
