"""
Scene renderer. Works on any moderngl context (the Qt widget's, or a
standalone one for camera sensors) and holds no Qt or ROS state.

The scene is a sky gradient, a ground grid with distance fog, instanced cones
(one draw call; knocked cones lie on their side), the car as a box body (it
follows roll/pitch) on four wheels (fronts steer, all spin, they stay level
with the ground), and a blob shadow. Lighting is one
directional sun plus ambient.
"""
import math

import numpy as np

from . import meshes

SKY_TOP = (0.36, 0.55, 0.80)
SKY_HORIZON = (0.78, 0.84, 0.90)
SUN_DIR = (-0.35, 0.45, 0.82)   # towards the sun

_LIT_VS = """
#version 330
uniform mat4 u_viewproj;
uniform mat4 u_model;
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_color;
out vec3 v_normal;
out vec3 v_color;
out vec3 v_world;
void main() {
    vec4 w = u_model * vec4(in_pos, 1.0);
    v_world = w.xyz;
    v_normal = mat3(u_model) * in_normal;
    v_color = in_color;
    gl_Position = u_viewproj * w;
}
"""

_CONE_VS = """
#version 330
uniform mat4 u_viewproj;
in vec3 in_pos;
in vec3 in_normal;
in float in_h;
in vec2 in_offset;
in vec2 in_size;      // base radius, height
in vec3 in_body;
in vec3 in_stripe;
in vec2 in_knocked;   // (knocked 0/1, direction it fell)
out vec3 v_normal;
out vec3 v_color;
out vec3 v_world;
void main() {
    vec3 p = in_pos * vec3(in_size.x, in_size.x, in_size.y);
    vec3 n = normalize(in_normal / vec3(in_size.x, in_size.x, in_size.y));
    if (in_knocked.x > 0.5) {
        // tip over onto its side (rotate about y), then turn to the fall direction
        p = vec3(p.z, p.y, -p.x + in_size.x);
        n = vec3(n.z, n.y, -n.x);
        float c = cos(in_knocked.y), s = sin(in_knocked.y);
        p = vec3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
        n = vec3(c * n.x - s * n.y, s * n.x + c * n.y, n.z);
    }
    vec3 w = p + vec3(in_offset, 0.0);
    v_world = w;
    v_normal = n;
    bool big = in_size.y > 0.4;
    bool stripe = big ? ((in_h > 0.3 && in_h < 0.42) || (in_h > 0.58 && in_h < 0.7))
                      : (in_h > 0.42 && in_h < 0.58);
    v_color = in_h < 0.0 ? in_body * 0.75 : (stripe ? in_stripe : in_body);
    gl_Position = u_viewproj * vec4(w, 1.0);
}
"""

_LIT_FS = """
#version 330
uniform vec3 u_sun;
uniform vec3 u_eye;
uniform vec3 u_fog_color;
uniform float u_fog_density;
uniform bool u_bgr;
in vec3 v_normal;
in vec3 v_color;
in vec3 v_world;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;
    float diffuse = max(dot(n, u_sun), 0.0);
    vec3 view = normalize(u_eye - v_world);
    float spec = pow(max(dot(reflect(-u_sun, n), view), 0.0), 24.0) * 0.25;
    vec3 c = v_color * (0.38 + 0.68 * diffuse) + vec3(spec);
    float fog = 1.0 - exp(-u_fog_density * length(u_eye - v_world));
    c = mix(c, u_fog_color, fog);
    f_color = vec4(u_bgr ? c.bgr : c, 1.0);
}
"""

_GROUND_VS = """
#version 330
uniform mat4 u_viewproj;
in vec2 in_pos;
out vec2 v_xy;
void main() {
    v_xy = in_pos;
    gl_Position = u_viewproj * vec4(in_pos, 0.0, 1.0);
}
"""

_GROUND_FS = """
#version 330
uniform vec3 u_eye;
uniform vec3 u_sun;
uniform vec3 u_fog_color;
uniform float u_fog_density;
uniform bool u_bgr;
in vec2 v_xy;
out vec4 f_color;
float grid(vec2 p, float spacing, float width) {
    vec2 g = abs(fract(p / spacing - 0.5) - 0.5) * spacing;
    vec2 aa = fwidth(p) * 1.2;
    vec2 line = 1.0 - smoothstep(vec2(width) - aa, vec2(width) + aa, g);
    return max(line.x, line.y);
}
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
    vec3 asphalt = vec3(0.30, 0.31, 0.33) * (0.94 + 0.06 * hash(floor(v_xy * 4.0)));
    vec3 c = mix(asphalt, vec3(0.42, 0.43, 0.45), grid(v_xy, 5.0, 0.03) * 0.6);
    c = mix(c, vec3(0.55, 0.55, 0.57), grid(v_xy, 50.0, 0.08) * 0.6);
    c *= 0.45 + 0.6 * u_sun.z;
    float fog = 1.0 - exp(-u_fog_density * length(u_eye - vec3(v_xy, 0.0)));
    c = mix(c, u_fog_color, fog);
    f_color = vec4(u_bgr ? c.bgr : c, 1.0);
}
"""

_SKY_VS = """
#version 330
in vec2 in_pos;
out float v_y;
void main() { v_y = in_pos.y; gl_Position = vec4(in_pos, 0.9999, 1.0); }
"""

_SKY_FS = """
#version 330
uniform vec3 u_top;
uniform vec3 u_horizon;
uniform float u_horizon_y;   // screen-space y of the horizon
uniform float u_up;          // -1 when the projection is flipped for readback
uniform bool u_bgr;
in float v_y;
out vec4 f_color;
void main() {
    float t = clamp((v_y - u_horizon_y) * u_up * 1.3, 0.0, 1.0);
    vec3 c = mix(u_horizon, u_top, sqrt(t));
    f_color = vec4(u_bgr ? c.bgr : c, 1.0);
}
"""

_SHADOW_VS = """
#version 330
uniform mat4 u_viewproj;
uniform mat4 u_model;
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos;
    gl_Position = u_viewproj * u_model * vec4(in_pos, 0.005, 1.0);
}
"""

_SHADOW_FS = """
#version 330
in vec2 v_uv;
out vec4 f_color;
void main() {
    float d = length(v_uv);
    f_color = vec4(0.0, 0.0, 0.0, 0.45 * (1.0 - smoothstep(0.55, 1.0, d)));
}
"""


# --- matrix helpers (row-major numpy; use gl_bytes() to upload) ------------

def gl_bytes(m):
    """GLSL expects column-major; numpy is row-major."""
    return np.ascontiguousarray(m.T, dtype=np.float32).tobytes()


def perspective(fovy, aspect, near, far):
    f = 1.0 / math.tan(fovy / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-9:              # looking straight down
        s = np.cross(f, (0.0, 1.0, 0.0))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def translate(x, y, z):
    m = np.identity(4)
    m[:3, 3] = (x, y, z)
    return m


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4)
    m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def body_matrix(x, y, yaw, roll, pitch, pivot_z=0.0):
    """World-from-body for the sprung mass.

    The simulator tilts sensors about the body origin (on the ground), so
    everything here uses pivot_z = 0 to match; at ~1 degree the visual
    difference from pivoting at the CoG is millimetres.
    """
    return (translate(x, y, pivot_z) @ rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
            @ translate(0.0, 0.0, -pivot_z))


class Renderer(object):
    def __init__(self, ctx, track, params, sensor_mounts=(), fog_density=0.004):
        self.ctx = ctx
        self.track = track
        self.p = params
        self.fog_density = fog_density
        sun = np.array(SUN_DIR)
        self.sun = tuple(sun / np.linalg.norm(sun))

        self.lit = ctx.program(vertex_shader=_LIT_VS, fragment_shader=_LIT_FS)
        self.cone_prog = ctx.program(vertex_shader=_CONE_VS, fragment_shader=_LIT_FS)
        self.ground_prog = ctx.program(vertex_shader=_GROUND_VS, fragment_shader=_GROUND_FS)
        self.sky_prog = ctx.program(vertex_shader=_SKY_VS, fragment_shader=_SKY_FS)
        self.shadow_prog = ctx.program(vertex_shader=_SHADOW_VS, fragment_shader=_SHADOW_FS)

        # sky: one full-screen triangle
        tri = np.array([-1, -1, 3, -1, -1, 3], dtype=np.float32)
        self.sky_vao = ctx.vertex_array(self.sky_prog, [(ctx.buffer(tri), '2f', 'in_pos')])

        # ground: a big quad centred on the track
        if len(track):
            centre = track.positions.mean(axis=0)
            extent = np.abs(track.positions - centre).max() + 300.0
        else:
            centre, extent = np.zeros(2), 300.0
        cx, cy = centre
        quad = np.array([cx - extent, cy - extent, cx + extent, cy - extent, cx + extent, cy + extent,
                         cx - extent, cy - extent, cx + extent, cy + extent, cx - extent, cy + extent],
                        dtype=np.float32)
        self.ground_vao = ctx.vertex_array(self.ground_prog, [(ctx.buffer(quad), '2f', 'in_pos')])

        # shadow: unit disc-ish quad scaled under the car
        sq = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1], dtype=np.float32)
        self.shadow_vao = ctx.vertex_array(self.shadow_prog, [(ctx.buffer(sq), '2f', 'in_pos')])

        # car: a box body on four wheels (sensor_mounts is accepted for API compatibility, unused)
        self.body_vao = self._lit_vao(meshes.car_box(params))
        self.wheel_vao = self._lit_vao(meshes.wheel(params.wheel_radius, 0.2))

        # trackside objects (barriers, tyres, poles, timekeeping): one static mesh
        obs = meshes.obstacle_mesh(getattr(track, 'obstacles', []))
        self.obstacle_vao = self._lit_vao(obs) if len(obs) else None

        # cones: one mesh, per-instance attributes. Track cones first, then
        # stacked spare cones (drawn as taller cones; they can be knocked over too)
        self.cone_vbo = ctx.buffer(meshes.cone())
        spares = list(getattr(track, 'cone_obstacles', []))
        self.n_track_cones = len(track)
        n = self.n_track_cones + len(spares)
        self._inst = np.zeros((max(n, 1), 12), dtype=np.float32)
        rows = [((x, y), r, h, c) for (x, y), r, h, c in
                zip(track.positions, track.radii, track.heights, track.colors)]
        rows += [((o.x, o.y), o.radius, o.height, o.color) for o in spares]
        for i, (xy, r, h, c) in enumerate(rows):
            body, stripe = meshes.CONE_PAINT.get(c, meshes.CONE_PAINT['unknown'])
            self._inst[i, 0:2] = xy
            self._inst[i, 2:4] = (r, h)
            self._inst[i, 4:7] = body
            self._inst[i, 7:10] = stripe
        self._fall_dir = np.random.default_rng(1).uniform(0, 2 * math.pi, max(n, 1)).astype(np.float32)
        self.inst_vbo = ctx.buffer(self._inst.tobytes(), dynamic=True)
        self.n_cones = n
        self.cone_vao = ctx.vertex_array(self.cone_prog, [
            (self.cone_vbo, '3f 3f 1f', 'in_pos', 'in_normal', 'in_h'),
            (self.inst_vbo, '2f 2f 3f 3f 2f/i', 'in_offset', 'in_size', 'in_body', 'in_stripe',
             'in_knocked'),
        ])
        self._last_hit = None

    def _lit_vao(self, data):
        return self.ctx.vertex_array(self.lit, [(self.ctx.buffer(data), '3f 3f 3f',
                                                 'in_pos', 'in_normal', 'in_color')])

    def update_cones(self, cones_hit, spare_hit=None):
        """Lay knocked cones (track cones, then spare-cone stacks) on their side."""
        hit = np.zeros(self.n_cones, dtype=bool)
        hit[:len(cones_hit)] = cones_hit
        if spare_hit is not None and len(spare_hit):
            hit[self.n_track_cones:self.n_track_cones + len(spare_hit)] = spare_hit
        key = hit.tobytes()
        if key == self._last_hit or not self.n_cones:
            return
        self._last_hit = key
        self._inst[:self.n_cones, 10] = hit.astype(np.float32)
        self._inst[:self.n_cones, 11] = self._fall_dir[:self.n_cones]
        self.inst_vbo.write(self._inst.tobytes())

    def render(self, view, proj, eye, car, fbo, draw_car=True, readback=False):
        """Draw one frame into `fbo`.

        car: dict with x, y, yaw, roll, pitch, delta, and wheel_angle (per
        wheel, FL FR RL RR) or wheel_spin (one angle for all four).
        readback=True renders upside down and in BGR, so glReadPixels returns
        rows top-first in ROS bgr8 order and the CPU does no per-pixel work.
        """
        ctx = self.ctx
        if readback:
            proj = np.diag([1.0, -1.0, 1.0, 1.0]) @ proj
        fbo.use()
        ctx.disable(ctx.CULL_FACE)
        fog_color = SKY_HORIZON
        viewproj = gl_bytes(proj @ view)

        # sky - horizon height from where a far point ahead lands on screen
        clear = SKY_HORIZON[::-1] if readback else SKY_HORIZON
        fbo.clear(*clear, 1.0, depth=1.0)
        ctx.disable(ctx.DEPTH_TEST)
        fwd = -view[2, :3]
        horizon_y = -1.0
        if math.hypot(fwd[0], fwd[1]) > 1e-3:
            far_pt = np.append(np.asarray(eye) + np.array([fwd[0], fwd[1], 0.0]) * 1e4, 1.0)
            clip = proj @ view @ far_pt
            if clip[3] > 1e-6:
                horizon_y = float(clip[1] / clip[3])
        self.sky_prog['u_horizon_y'].value = horizon_y
        self.sky_prog['u_top'].value = SKY_TOP
        self.sky_prog['u_horizon'].value = SKY_HORIZON
        self.sky_prog['u_up'].value = -1.0 if readback else 1.0
        self.sky_prog['u_bgr'].value = readback
        self.sky_vao.render()

        ctx.enable(ctx.DEPTH_TEST)
        for prog in (self.lit, self.cone_prog, self.ground_prog):
            prog['u_viewproj'].write(viewproj)
            prog['u_eye'].value = tuple(float(v) for v in eye)
            prog['u_sun'].value = self.sun
            prog['u_fog_color'].value = fog_color
            prog['u_fog_density'].value = self.fog_density
            prog['u_bgr'].value = readback
        self.ground_vao.render()

        if draw_car:
            p = self.p
            self.shadow_prog['u_viewproj'].write(viewproj)
            shadow = (translate(car['x'], car['y'], 0.0) @ rot_z(car['yaw'])
                      @ np.diag([p.length * 0.75, p.width * 0.75, 1.0, 1.0]))
            self.shadow_prog['u_model'].write(gl_bytes(shadow))
            ctx.enable(ctx.BLEND)
            ctx.depth_mask = False
            self.shadow_vao.render()
            ctx.depth_mask = True
            ctx.disable(ctx.BLEND)

        if self.n_cones:
            self.cone_vao.render(instances=self.n_cones)
        if self.obstacle_vao is not None:
            self.lit['u_model'].write(gl_bytes(np.identity(4)))
            self.obstacle_vao.render()

        if draw_car:
            p = self.p
            body = body_matrix(car['x'], car['y'], car['yaw'], car['roll'], car['pitch'])
            self.lit['u_model'].write(gl_bytes(body))
            self.body_vao.render()
            # wheels hang off the unsprung chassis: level with the ground, not the body
            chassis = translate(car['x'], car['y'], 0.0) @ rot_z(car['yaw'])
            half = p.track_width / 2.0
            angles = car.get('wheel_angle') or [car.get('wheel_spin', 0.0)] * 4
            for k, (wx, wy, steers) in enumerate(((p.lf, half, True), (p.lf, -half, True),
                                                  (-p.lr, half, False), (-p.lr, -half, False))):
                m = chassis @ translate(wx, wy, p.wheel_radius)
                if steers:
                    m = m @ rot_z(car['delta'])
                m = m @ rot_y(angles[k])
                self.lit['u_model'].write(gl_bytes(m))
                self.wheel_vao.render()


def camera_view(car_pose, cam):
    """(view matrix, eye) for a body-mounted camera (sensors.CameraConfig)."""
    body = body_matrix(car_pose.x, car_pose.y, car_pose.yaw, car_pose.roll, car_pose.pitch)
    mount = (translate(cam.x, cam.y, cam.z) @ rot_z(math.radians(cam.yaw))
             @ rot_y(math.radians(cam.pitch)) @ rot_x(math.radians(cam.roll)))
    world = body @ mount
    eye = world[:3, 3]
    fwd, up = world[:3, 0], world[:3, 2]
    return look_at(eye, eye + fwd, up), eye, world


def camera_projection(cam, near=0.05, far=300.0):
    aspect = cam.width / float(cam.height)
    fovy = 2.0 * math.atan(math.tan(math.radians(cam.fov_deg) / 2.0) / aspect)
    return perspective(fovy, aspect, near, far)
