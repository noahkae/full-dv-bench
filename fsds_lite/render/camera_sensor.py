"""
Camera sensor: renders the configured cameras offscreen on a dedicated thread.

A GL context belongs to the thread that created it, so this thread owns a
standalone (headless EGL/GLX) moderngl context for its whole life. The
physics thread only enqueues CameraRequests. If the GPU can't keep up,
requests are dropped instead of queued, so images stay current rather than
lagging further behind.

Images are BGR uint8, top row first - the same layout the FSDS bridge
publishes as bgr8. The GPU produces that layout directly, so the CPU only
copies the pixels once.

Readback is asynchronous and the CPU does no per-pixel work. A plain
glReadPixels stalls 4-10 ms per frame while holding Python's GIL, which
stutters the physics and UI threads, and drivers only take the fast async
path for 4-byte RGBA pixels. So:
  1. the scene renders BGR, upside down (Renderer readback=True) into a texture
  2. a tiny pack pass writes those 3-byte pixels tightly packed into an RGBA
     texture - its raw bytes ARE the bgr8 image
  3. that texture is copied into a pixel buffer object (async, ~0.03 ms)
  4. the buffer is collected when the next frame is rendered (~0.4 ms)
Each image keeps its own capture stamp; it just arrives one camera period
later, like a real camera's latency.

Depth cameras (settings.json ImageType 1 = DepthPlanar, 2 = DepthPerspective,
as in AirSim) skip MSAA, render into a depth texture, and a second small pass
converts it to float32 metres (planar: distance along the optical axis;
perspective: along each pixel's ray). That comes back through the same async
path. The FSDS bridge then degrades the image: a 7x7 box blur, and depth
rounded up to steps that grow with range (1 m steps up close, 10 m from 10 m
out). FSDS's C++ rounding has a float bug (`(x + m - 1) / m * m` is just
x + m - 1), so we implement the intent - round up to a multiple of m - not
the bug. Sky has no depth and is +inf (REP 117 "too far").
"""
import logging
import queue
import threading

import numpy as np

from ..simulator import CameraSample
from .renderer import Renderer, camera_projection, camera_view

log = logging.getLogger('fsds_lite.camera')

PACK_WIDTH = 1024

_PACK_VS = """
#version 330
in vec2 in_pos;
void main() { gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

# Output texel i holds bytes 4i..4i+3 of the tightly packed 3-byte-per-pixel image.
_PACK_FS = """
#version 330
uniform sampler2D u_src;
uniform int u_width;
uniform int u_total;      // width * height * 3
out vec4 f_color;
float byte_at(int b) {
    if (b >= u_total) return 0.0;
    int p = b / 3;
    return texelFetch(u_src, ivec2(p % u_width, p / u_width), 0)[b % 3];
}
void main() {
    int i = int(gl_FragCoord.y) * PACK_WIDTH + int(gl_FragCoord.x);
    int b = 4 * i;
    f_color = vec4(byte_at(b), byte_at(b + 1), byte_at(b + 2), byte_at(b + 3));
}
""".replace('PACK_WIDTH', str(PACK_WIDTH))

# Depth buffer -> metres. Rows are already top-first (readback render).
_DEPTH_FS = """
#version 330
uniform sampler2D u_depth;
uniform float u_near;
uniform float u_far;
uniform vec4 u_intr;          // fx, fy, cx, cy
uniform int u_perspective;
out float f_depth;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    float d = texelFetch(u_depth, p, 0).r;
    if (d >= 1.0) { f_depth = -1.0; return; }          // sky
    float z = 2.0 * d - 1.0;
    float planar = 2.0 * u_near * u_far / (u_far + u_near - z * (u_far - u_near));
    if (u_perspective == 1) {
        float u = (float(p.x) + 0.5 - u_intr.z) / u_intr.x;
        float v = (float(p.y) + 0.5 - u_intr.w) / u_intr.y;
        planar *= sqrt(1.0 + u * u + v * v);
    }
    f_depth = planar;
}
"""

DEPTH_NEAR, DEPTH_FAR = 0.05, 300.0


def noisify_depth(depth):
    """FSDS bridge depth degradation: 7x7 box blur, then range-dependent rounding."""
    sky = ~np.isfinite(depth)
    d = np.where(sky, DEPTH_FAR, depth).astype(np.float64)
    pad = np.pad(d, 3, mode='reflect')                     # OpenCV BORDER_REFLECT_101
    integral = np.pad(pad.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    h, w = d.shape
    blur = (integral[7:7 + h, 7:7 + w] - integral[:h, 7:7 + w]
            - integral[7:7 + h, :w] + integral[:h, :w]) / 49.0
    step = np.ceil(np.clip(blur, 1.0, 10.0))
    out = (np.ceil(blur / step) * step).astype(np.float32)
    out[sky] = np.inf
    return out


def _quat_from_matrix(m):
    """(x, y, z, w) from a 3x3 rotation matrix."""
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return ((m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s, 0.25 / s)
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return tuple(q)


class CameraRenderer(object):
    def __init__(self, sim, cameras, samples=4, sensor_mounts=()):
        self.sim = sim
        self.cameras = list(cameras)
        self.samples = samples
        self.sensor_mounts = sensor_mounts
        self.queue = queue.Queue(maxsize=2 * max(1, len(self.cameras)))
        self.thread = threading.Thread(target=self._run, name='camera-render', daemon=True)
        self.ready = threading.Event()
        self._stopping = threading.Event()
        self.error = None
        self.dropped = 0
        self.rendered = 0

    def start(self, timeout=10.0):
        self.thread.start()
        if not self.ready.wait(timeout):
            raise RuntimeError('camera renderer did not start within {:.0f}s'.format(timeout))
        if self.error:
            raise RuntimeError(self.error)
        self.sim.camera_sink = self.submit
        log.info('rendering cameras %s offscreen on %s',
                 ', '.join('{} {}x{}'.format(c.name, c.width, c.height) for c in self.cameras),
                 self.gl_renderer)
        return self

    def stop(self, timeout=5.0):
        """Stop rendering and release the GL context before returning.

        The queue is usually full (the GPU runs behind and requests get
        dropped), so the stop sentinel must not be dropped too: a lost sentinel
        left this thread rendering and its EGL context alive, with GPU work
        and readbacks in flight, when the process exited.
        """
        self.sim.camera_sink = None
        self._stopping.set()
        try:
            while True:
                self.queue.get_nowait()      # drop pending frames; the sentinel always fits
        except queue.Empty:
            pass
        try:
            self.queue.put(None, timeout=1.0)
        except queue.Full:
            pass                             # _stopping still ends the loop
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            log.warning('camera render thread did not stop within %.0fs', timeout)

    def submit(self, request):
        try:
            self.queue.put_nowait(request)
        except queue.Full:
            self.dropped += 1

    # ---- render thread ---------------------------------------------------
    def _run(self):
        try:
            import moderngl
            try:
                # EGL first: the default (GLX) backend can hang on Wayland / in containers
                ctx = moderngl.create_standalone_context(require=330, backend='egl')
            except Exception:
                ctx = moderngl.create_standalone_context(require=330)
            self.gl_renderer = ctx.info.get('GL_RENDERER', 'unknown GPU')
            renderer = Renderer(ctx, self.sim.track, self.sim.vehicle.p, self.sensor_mounts)
            self._pack_prog = ctx.program(vertex_shader=_PACK_VS, fragment_shader=_PACK_FS)
            tri = np.array([-1, -1, 3, -1, -1, 3], dtype=np.float32)
            tri_buf = ctx.buffer(tri)
            self._pack_vao = ctx.vertex_array(self._pack_prog, [(tri_buf, '2f', 'in_pos')])
            self._depth_prog = ctx.program(vertex_shader=_PACK_VS, fragment_shader=_DEPTH_FS)
            self._depth_vao = ctx.vertex_array(self._depth_prog, [(tri_buf, '2f', 'in_pos')])
            # keyed by name; each entry carries this renderer's own config (size, FOV)
            targets = {c.name: (c,) + self._make_targets(ctx, c) for c in self.cameras}
        except Exception as e:
            self.error = 'camera renderer unavailable: {}: {}'.format(type(e).__name__, e)
            self.ready.set()
            return
        self.ready.set()

        try:
            while not self._stopping.is_set():
                req = self.queue.get()
                if req is None or self._stopping.is_set():
                    break
                entry = targets.get(req.camera.name)
                if entry is None:                 # not one of ours
                    continue
                try:
                    done = self._render(ctx, renderer, entry, req)
                    if done is not None:
                        self.sim.emit(done)
                        self.rendered += 1
                except Exception:
                    log.exception('camera %s render failed', req.camera.name)
        finally:
            # Let queued GPU work and the async readbacks complete, then free the
            # buffers and context ourselves rather than leaving it to process exit.
            try:
                ctx.finish()
            except Exception:
                pass
            try:
                ctx.release()
            except Exception:
                log.exception('releasing the camera GL context failed')

    def _make_targets(self, ctx, cam):
        size = (cam.width, cam.height)
        color = ctx.texture(size, 4)
        color.filter = (ctx.NEAREST, ctx.NEAREST)
        if cam.is_depth:
            depth = ctx.depth_texture(size)
            depth.compare_func = ''                        # sample raw depth values
            depth.filter = (ctx.NEAREST, ctx.NEAREST)
            resolve = ctx.framebuffer(color, depth)
            metres = ctx.framebuffer(ctx.texture(size, 1, dtype='f4'))
            pbos = [ctx.buffer(reserve=cam.width * cam.height * 4) for _ in range(2)]
            state = {'next': 0, 'pending': None, 'depth': depth, 'metres': metres,
                     'nbytes': cam.width * cam.height * 4}
            return None, resolve, pbos, state
        resolve = ctx.framebuffer(color, ctx.depth_renderbuffer(size))
        msaa = None
        if self.samples > 1:
            try:
                msaa = ctx.framebuffer(ctx.renderbuffer(size, samples=self.samples),
                                       ctx.depth_renderbuffer(size, samples=self.samples))
            except Exception:
                log.warning('MSAA unavailable for camera %s; rendering without it', cam.name)
        nbytes = cam.width * cam.height * 3
        rows = -(-nbytes // (4 * PACK_WIDTH))
        packed = ctx.framebuffer(ctx.renderbuffer((PACK_WIDTH, rows)))
        pbos = [ctx.buffer(reserve=PACK_WIDTH * rows * 4) for _ in range(2)]
        state = {'next': 0, 'pending': None, 'color': color, 'packed': packed, 'nbytes': nbytes}
        return msaa, resolve, pbos, state

    def _render(self, ctx, renderer, targets, req):
        """Render `req`, start its async readback, and return the PREVIOUS frame (or None)."""
        cam, msaa, resolve, pbos, state = targets
        b = req.body
        view, eye, world = camera_view(b, cam)
        renderer.update_cones(req.cones_hit, req.spare_hit)
        with self.sim.lock:
            s = self.sim.vehicle.state
            car = {'x': b.x, 'y': b.y, 'yaw': b.yaw, 'roll': b.roll, 'pitch': b.pitch,
                   'delta': s.delta, 'wheel_angle': list(s.wheel_angle)}
        target = msaa or resolve
        ctx.viewport = (0, 0, cam.width, cam.height)
        renderer.render(view, camera_projection(cam), eye, car, target, readback=True)
        if msaa is not None:
            ctx.copy_framebuffer(resolve, msaa)
        if cam.is_depth:
            return self._depth_readback(ctx, cam, req, eye, world, pbos, state)
        # pack the BGR pixels into RGBA texels, then start the async copy
        packed = state['packed']
        packed.use()
        ctx.viewport = (0, 0) + packed.size
        state['color'].use(location=0)
        self._pack_prog['u_src'].value = 0
        self._pack_prog['u_width'].value = cam.width
        self._pack_prog['u_total'].value = state['nbytes']
        self._pack_vao.render()
        i = state['next']
        packed.read_into(pbos[i], components=4)
        state['next'] = 1 - i
        meta = CameraSample(cam.name, req.stamp, None, cam.intrinsics(), tuple(eye),
                            _quat_from_matrix(world[:3, :3]))
        previous, state['pending'] = state['pending'], (i, meta)
        if previous is None:
            return None
        j, sample = previous
        raw = pbos[j].read(size=state['nbytes'])
        sample.image = np.frombuffer(raw, dtype=np.uint8).reshape(cam.height, cam.width, 3)
        return sample

    def _depth_readback(self, ctx, cam, req, eye, world, pbos, state):
        metres = state['metres']
        metres.use()
        ctx.viewport = (0, 0, cam.width, cam.height)
        state['depth'].use(location=0)
        prog = self._depth_prog
        prog['u_depth'].value = 0
        prog['u_near'].value = DEPTH_NEAR
        prog['u_far'].value = DEPTH_FAR
        prog['u_intr'].value = tuple(float(v) for v in cam.intrinsics())
        prog['u_perspective'].value = 1 if cam.image_type == 2 else 0
        self._depth_vao.render()
        i = state['next']
        metres.read_into(pbos[i], components=1, dtype='f4')
        state['next'] = 1 - i
        meta = CameraSample(cam.name, req.stamp, None, cam.intrinsics(), tuple(eye),
                            _quat_from_matrix(world[:3, :3]))
        previous, state['pending'] = state['pending'], (i, meta)
        if previous is None:
            return None
        j, sample = previous
        depth = np.frombuffer(pbos[j].read(size=state['nbytes']), dtype=np.float32)
        depth = depth.reshape(cam.height, cam.width).copy()
        depth[depth < 0.0] = np.inf                       # sky
        sample.image = noisify_depth(depth) if cam.depth_noise else depth
        return sample
