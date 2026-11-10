"""
Interactive 3D view: a QOpenGLWidget drawing with the shared Renderer.

Camera modes (C cycles; dragging the mouse jumps to orbit):
    chase    smoothed follow camera behind the car
    orbit    drag to rotate around the car, wheel to zoom
    top      overhead, follows the car (wheel zooms)
    onboard  the first configured sensor camera's viewpoint

Needs an OpenGL 3.3 core default surface format; __main__ sets it with
`configure_surface_format()` before creating the QApplication.
"""
import logging
import math
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..simulator import BodyPose
from .renderer import Renderer, camera_projection, camera_view, look_at, perspective

log = logging.getLogger('fsds_lite.viewer3d')

MODES = ('chase', 'orbit', 'top', 'onboard')


class _QtGLLoader(object):
    """Lets moderngl call GL through Qt's own context, whatever it is (EGL on
    native Wayland, GLX on X11/XWayland, WSLg), instead of guessing the
    platform. glcontext's EGL backend can't attach to an existing context,
    so this is what makes the 3D view work on native Wayland."""

    def __init__(self, qt_context):
        self.qt_context = qt_context

    def load_opengl_function(self, name):
        ptr = self.qt_context.getProcAddress(name.encode())
        return int(ptr) if ptr else 0

    def release(self):
        pass


def attach_moderngl(qt_context):
    """moderngl Context for the Qt GL context that is current right now."""
    import moderngl
    if hasattr(moderngl, 'init_context'):                      # moderngl >= 5.9
        moderngl.init_context(_QtGLLoader(qt_context))
        ctx = moderngl.get_context()
        if ctx.version_code < 330:
            raise RuntimeError('OpenGL 3.3 required, got {}'.format(ctx.version_code))
        return ctx
    return moderngl.create_context(require=330)                 # older: GLX detection


def configure_surface_format(samples=4):
    fmt = QtGui.QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(samples)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)


class SimView3D(QtWidgets.QOpenGLWidget):
    failed = QtCore.pyqtSignal(str)

    def __init__(self, sim, sensors, parent=None, fps=60):
        super().__init__(parent)
        self.sim = sim
        self.cameras = list(sensors.cameras or [])
        self.mounts = [(l.x, l.y, l.z) for l in sensors.lidars]
        self.mode = 'chase'
        self.ctx = None
        self.renderer = None
        self.orbit_az, self.orbit_el, self.orbit_dist = math.radians(200), math.radians(25), 8.0
        self.top_height = 40.0
        self._chase_yaw = None
        self._last_paint = None
        self._drag = None
        self._gl_failed = False
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setMinimumSize(320, 240)
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.update)
        self._interval = int(1000 / fps)
        self.timer.start(self._interval)

    # Only redraw while on screen: a hidden tab still costs a GL frame per tick
    # on a GPU the compositor shares.
    def showEvent(self, ev):
        super().showEvent(ev)
        if not self._gl_failed:
            self.timer.start(self._interval)

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self.timer.stop()

    # ---- GL lifecycle ----------------------------------------------------
    def initializeGL(self):
        try:
            self.ctx = attach_moderngl(self.context())
            self.renderer = Renderer(self.ctx, self.sim.track, self.sim.vehicle.p, self.mounts)
            log.info('3D view on %s (OpenGL %s)', self.ctx.info.get('GL_RENDERER'),
                     self.ctx.version_code)
        except Exception as e:
            self.renderer = None
            self._gl_failed = True
            msg = '3D view unavailable: {}: {}'.format(type(e).__name__, e)
            log.error(msg)
            self.timer.stop()
            QtCore.QTimer.singleShot(0, lambda: self.failed.emit(msg))

    def paintGL(self):
        if self.renderer is None:
            return
        s = self.sim.snapshot()
        self.renderer.update_cones(s['cones_hit'], s['spare_hit'])
        ratio = self.devicePixelRatioF()
        w, h = max(1, int(self.width() * ratio)), max(1, int(self.height() * ratio))
        view, proj, eye, draw_car = self._camera(s, w / float(h))
        fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
        self.ctx.viewport = (0, 0, w, h)
        self.renderer.render(view, proj, eye, s, fbo, draw_car=draw_car)

    # ---- cameras -----------------------------------------------------------
    def _camera(self, s, aspect):
        x, y, yaw = s['x'], s['y'], s['yaw']
        car = np.array([x, y, 0.4])
        proj = perspective(math.radians(60), aspect, 0.1, 600.0)
        if self.mode == 'onboard' and self.cameras:
            cam = self.cameras[0]
            view, eye, _ = camera_view(BodyPose(x, y, yaw, s['roll'], s['pitch']), cam)
            proj = camera_projection(cam)
            proj[0, 0] = proj[1, 1] / aspect          # keep the camera's vertical FOV, fit the widget
            return view, proj, eye, True
        if self.mode == 'orbit':
            ce = math.cos(self.orbit_el)
            eye = car + self.orbit_dist * np.array([ce * math.cos(self.orbit_az),
                                                    ce * math.sin(self.orbit_az),
                                                    math.sin(self.orbit_el)])
            return look_at(eye, car), proj, eye, True
        if self.mode == 'top':
            eye = np.array([x, y, self.top_height])
            return look_at(eye, (x, y, 0.0), up=(1.0, 0.0, 0.0)), proj, eye, True
        # chase: locked to the car's position (no lag, so the car never wobbles
        # on screen); only the heading is smoothed, with a real-time constant
        # so the feel doesn't depend on frame rate.
        now = time.monotonic()
        dt = 0.0 if self._last_paint is None else min(now - self._last_paint, 0.1)
        self._last_paint = now
        if self._chase_yaw is None:
            self._chase_yaw = yaw
        dyaw = math.atan2(math.sin(yaw - self._chase_yaw), math.cos(yaw - self._chase_yaw))
        self._chase_yaw += dyaw * (1.0 - math.exp(-dt / 0.25))
        back = np.array([math.cos(self._chase_yaw), math.sin(self._chase_yaw), 0.0])
        eye = car - 5.5 * back + np.array([0.0, 0.0, 2.2])
        target = car + 4.0 * back
        return look_at(eye, target), proj, eye, True

    def cycle_mode(self):
        modes = [m for m in MODES if m != 'onboard' or self.cameras]
        self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
        self._chase_yaw = None

    def handle_key(self, key):
        if key == QtCore.Qt.Key_C:
            self.cycle_mode()
            return True
        return False

    def status(self):
        return 'view {} (C)'.format(self.mode)

    # ---- mouse -------------------------------------------------------------
    def mousePressEvent(self, ev):
        self._drag = ev.pos()
        if self.mode != 'orbit':
            self.mode = 'orbit'

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        d = ev.pos() - self._drag
        self._drag = ev.pos()
        self.orbit_az -= d.x() * 0.008
        self.orbit_el = min(max(self.orbit_el + d.y() * 0.006, math.radians(3)), math.radians(85))

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def wheelEvent(self, ev):
        k = 0.9 if ev.angleDelta().y() > 0 else 1.1
        if self.mode == 'top':
            self.top_height = min(max(self.top_height * k, 5.0), 300.0)
        else:
            self.mode = 'orbit'
            self.orbit_dist = min(max(self.orbit_dist * k, 2.5), 150.0)
