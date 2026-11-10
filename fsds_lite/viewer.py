"""
Sim window: a 3D tab (moderngl, if available) and a 2D top-down tab
(pyqtgraph), with one HUD and keyboard driving that works on either tab.

SimView (2D) and SimView3D are plain widgets so they can later be dropped
into the workbench - that is the path to the single combined app. Both only
read Simulator.snapshot() on a timer and never block the physics thread.

Keys (manual driving works only while API control is OFF, like FSDS):
    W / Up      throttle            S / Down   brake
    A / Left    steer left          D / Right  steer right
    R  reset    P  pause            Tab  switch 3D / top-down
    3D:  C  cycle camera (chase / orbit / top / onboard), mouse drag orbits
    2D:  F  toggle follow-car       L  toggle lidar points
"""
import logging
import math
import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from .simulator import Controls

log = logging.getLogger('fsds_lite.viewer')

CONE_COLORS = {
    'blue': (40, 110, 255), 'yellow': (255, 210, 0), 'orange': (255, 130, 0),
    'big_orange': (255, 90, 0), 'unknown': (200, 200, 200),
}
HIT_COLOR = (90, 90, 90)

# Dynamic UI sizing. Sizes are in points / logical pixels, so they already
# follow the screen's DPI scale. On top of that the bars scale with the window:
# at REF_SIZE (about a full laptop screen) they look exactly as designed; a
# smaller tile (e.g. a tiling window manager split) shrinks them proportionally
# down to MIN_UI_SCALE, so the layout looks the same, just smaller, instead of
# being squeezed. They never grow beyond the designed size.
REF_SIZE = (1200.0, 800.0)
MIN_UI_SCALE = 0.6
TAB_FONT_PT, HUD_FONT_PT, AXIS_FONT_PT = 11.0, 11.0, 9.0


def ui_scale_for(width, height):
    f = min(width / REF_SIZE[0], height / REF_SIZE[1], 1.0)
    return round(max(f, MIN_UI_SCALE) * 20) / 20          # 0.05 steps: no restyle churn


def tab_style(f):
    return """
QTabBar::tab {{ padding: {py}px {px}px; font-size: {pt:.1f}pt; min-width: {mw}px; }}
QTabBar::tab:selected {{ font-weight: bold; }}
""".format(py=round(7 * f), px=round(22 * f), pt=TAB_FONT_PT * f, mw=round(80 * f))


def hud_style(f):
    return """
QLabel {{ background: #1d1f23; color: #e8e8e8; padding: {py}px {px}px; }}
""".format(py=round(7 * f), px=round(12 * f))


def hud_text(groups):
    """Join HUD readings so word-wrap can only break BETWEEN them."""
    return '   '.join(g.replace(' ', '\u00a0') for g in groups if g)

Q = QtCore.Qt
DRIVE_KEYS = {Q.Key_W, Q.Key_S, Q.Key_A, Q.Key_D, Q.Key_Up, Q.Key_Down, Q.Key_Left, Q.Key_Right}


class DriveKeys(object):
    """Keyboard -> manual Controls, ramped so keys behave like analog pedals.

    Keys are on/off, so applying them directly slams full steering lock and
    throttle in one frame. Like UE4's vehicle input rates (which FSDS's own
    keyboard driving uses), commands rise and fall at a limited rate instead.
    API commands from the stack are never ramped.
    """
    STEER_RATE = 2.5      # full lock in 0.4 s
    STEER_RETURN = 5.0    # centres twice as fast
    PEDAL_RATE = 4.0      # 0 -> 1 in 0.25 s
    THROTTLE = 0.6

    def __init__(self, sim, handle_reset=True):
        self.sim = sim
        self.handle_reset = handle_reset      # the workbench owns R (it also disarms)
        self.keys = set()
        self.cmd = Controls()
        self._last = None
        self.timer = QtCore.QTimer()
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(10)

    def press(self, k):
        if k in DRIVE_KEYS:
            self.keys.add(k)
            return True
        if k == Q.Key_R and self.handle_reset:
            self.sim.reset()
            self.cmd = Controls()
            return True
        if k == Q.Key_P:
            self.sim.set_paused(not self.sim.paused)
            return True
        return False

    def release(self, k):
        if k in self.keys:
            self.keys.discard(k)
            return True
        return False

    def clear(self):
        self.keys.clear()

    def reset_inputs(self):
        self.keys.clear()
        self.cmd = Controls()

    @staticmethod
    def _toward(value, target, up_rate, down_rate, dt):
        rate = up_rate if abs(target) > abs(value) and target * value >= 0 else down_rate
        step = rate * dt
        return target if abs(target - value) <= step else value + step * (1 if target > value else -1)

    def _tick(self):
        now = time.monotonic()
        dt = 0.0 if self._last is None else min(now - self._last, 0.1)
        self._last = now
        k = self.keys
        steer = (1.0 if (Q.Key_D in k or Q.Key_Right in k) else 0.0) - \
                (1.0 if (Q.Key_A in k or Q.Key_Left in k) else 0.0)
        throttle = self.THROTTLE if (Q.Key_W in k or Q.Key_Up in k) else 0.0
        brake = 1.0 if (Q.Key_S in k or Q.Key_Down in k) else 0.0
        c = self.cmd
        new = Controls(
            throttle=self._toward(c.throttle, throttle, self.PEDAL_RATE, 2 * self.PEDAL_RATE, dt),
            brake=self._toward(c.brake, brake, self.PEDAL_RATE, 2 * self.PEDAL_RATE, dt),
            steering=self._toward(c.steering, steer, self.STEER_RATE, self.STEER_RETURN, dt))
        if new != c:
            self.cmd = new
            self.sim.set_manual_controls(new)


class SimView(QtWidgets.QWidget):
    """2D top-down view."""

    def __init__(self, sim, parent=None, fps=30):
        super().__init__(parent)
        self.sim = sim
        self.follow = True
        self.show_lidar = True

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel('bottom', 'x (m)')
        self.plot.setLabel('left', 'y (m)')
        self.plot.setFocusPolicy(QtCore.Qt.NoFocus)

        track = sim.track
        self._base_brushes = [pg.mkBrush(*CONE_COLORS.get(c, CONE_COLORS['unknown']))
                              for c in track.colors]
        self._hit_brush = pg.mkBrush(*HIT_COLOR)
        sizes = [12 if c == 'big_orange' else 9 for c in track.colors]
        self.cones = pg.ScatterPlotItem(pos=track.positions, size=sizes, pen=None,
                                        brush=self._base_brushes)
        self.plot.addItem(self.cones)
        self._add_trackside(track)
        self.lidar = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush(255, 60, 60, 160))
        self.plot.addItem(self.lidar)
        self.car = pg.PlotDataItem(pen=pg.mkPen((230, 230, 230), width=2))
        self.plot.addItem(self.car)
        self.heading = pg.PlotDataItem(pen=pg.mkPen((80, 255, 120), width=2))
        self.plot.addItem(self.heading)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        self._last_hit = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(int(1000 / fps))
        self.plot.setRange(xRange=(-15, 15), yRange=(-15, 15))
        self.set_ui_scale(1.0)

    def _add_trackside(self, track):
        """Barriers/equipment as outlines, tyres/poles as discs, spare cones, timing lines."""
        xs, ys = [], []
        for o in track.obstacles:
            if o.kind != 'box':
                continue
            c, s = math.cos(o.yaw), math.sin(o.yaw)
            for px, py in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1)):
                lx, ly = px * o.half_x, py * o.half_y
                xs.append(o.x + c * lx - s * ly)
                ys.append(o.y + s * lx + c * ly)
            xs.append(np.nan)
            ys.append(np.nan)
        if xs:
            self.plot.addItem(pg.PlotDataItem(xs, ys, connect='finite',
                                              pen=pg.mkPen((150, 150, 145), width=1.5)))
        cyl = [o for o in track.obstacles if o.kind == 'cylinder']
        if cyl:
            self.plot.addItem(pg.ScatterPlotItem(
                pos=[(o.x, o.y) for o in cyl], size=[2 * o.radius for o in cyl], pxMode=False,
                pen=pg.mkPen((120, 120, 120)), brush=pg.mkBrush(25, 25, 25)))
        spares = track.cone_obstacles
        self._spare_brushes = [pg.mkBrush(*CONE_COLORS.get(o.color, CONE_COLORS['unknown']))
                               for o in spares]
        self.spares = pg.ScatterPlotItem(pos=[(o.x, o.y) for o in spares] or np.zeros((0, 2)),
                                         size=11, symbol='s', pen=pg.mkPen('w', width=1),
                                         brush=self._spare_brushes or None)
        self.plot.addItem(self.spares)
        self._last_spare = None
        for line in {id(l): l for l in (track.start_line, track.finish_line) if l}.values():
            self.plot.addItem(pg.PlotDataItem([line.a[0], line.b[0]], [line.a[1], line.b[1]],
                                              pen=pg.mkPen((255, 50, 50), width=3)))

    def set_ui_scale(self, f):
        """Axis labels and tick text follow the window's UI scale."""
        font = QtGui.QFont()
        font.setPointSizeF(AXIS_FONT_PT * f)
        style = {'font-size': '{:.1f}pt'.format(AXIS_FONT_PT * f)}
        for side, label in (('bottom', 'x (m)'), ('left', 'y (m)')):
            axis = self.plot.getAxis(side)
            axis.setStyle(tickFont=font)
            self.plot.setLabel(side, label, **style)

    def handle_key(self, k):
        if k == Q.Key_F:
            self.follow = not self.follow
            return True
        if k == Q.Key_L:
            self.show_lidar = not self.show_lidar
            if not self.show_lidar:
                self.lidar.clear()
            return True
        return False

    def status(self):
        return 'follow {} (F) lidar {} (L)'.format('on' if self.follow else 'off',
                                                   'on' if self.show_lidar else 'off')

    def refresh(self):
        if not self.isVisible():
            return
        s = self.sim.snapshot()
        p = self.sim.vehicle.p
        c, sn = math.cos(s['yaw']), math.sin(s['yaw'])
        hl, hw = p.length / 2, p.width / 2
        box = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw]])
        rot = np.array([[c, -sn], [sn, c]])
        pts = box @ rot.T + (s['x'], s['y'])
        self.car.setData(pts[:, 0], pts[:, 1])
        wheel = s['yaw'] + s['delta']
        self.heading.setData([s['x'], s['x'] + 2.0 * math.cos(wheel)],
                             [s['y'], s['y'] + 2.0 * math.sin(wheel)])

        hit = s['cones_hit']
        key = hit.tobytes()
        if key != self._last_hit:
            self._last_hit = key
            self.cones.setBrush([self._hit_brush if h else b
                                 for h, b in zip(hit, self._base_brushes)])

        spare_key = s['spare_hit'].tobytes()
        if len(s['spare_hit']) and spare_key != self._last_spare:
            self._last_spare = spare_key
            self.spares.setBrush([self._hit_brush if h else b
                                  for h, b in zip(s['spare_hit'], self._spare_brushes)])

        if self.show_lidar and s['lidar']:
            world = []
            for sample in s['lidar'].values():
                lx, ly, lz, lyaw = sample.pose
                pts_s = sample.points
                if not len(pts_s):
                    continue
                cl, sl = math.cos(lyaw), math.sin(lyaw)
                q = pts_s[pts_s[:, 2] > -lz + 0.05]                    # hide ground returns
                world.append(np.column_stack([lx + q[:, 0] * cl - q[:, 1] * sl,
                                              ly + q[:, 0] * sl + q[:, 1] * cl]))
            if world:
                self.lidar.setData(pos=np.vstack(world))

        if self.follow:
            vb = self.plot.getViewBox()
            (x0, x1), (y0, y1) = vb.viewRange()
            hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
            vb.setRange(xRange=(s['x'] - hx, s['x'] + hx), yRange=(s['y'] - hy, s['y'] + hy),
                        padding=0)


class SimPanel(QtWidgets.QWidget):
    """3D + top-down tabs, HUD and keyboard driving, as one embeddable widget.

    SimWindow wraps it for the standalone sim; the workbench drops it in as a
    tab (`embedded=True`: keys only act while the panel is visible, and R is
    left to the workbench so a reset also disarms).
    """

    view3d_failed = QtCore.pyqtSignal(str)

    def __init__(self, sim, sensors=None, use_3d=True, embedded=False, parent=None, fps_3d=60):
        super().__init__(parent)
        self.sim = sim
        self.embedded = embedded
        self.keys = DriveKeys(sim, handle_reset=not embedded)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tabs.tabBar().setFocusPolicy(QtCore.Qt.NoFocus)
        self.tabs.setDocumentMode(True)
        self.view3d = None
        self.view3d_error = None
        if use_3d and sensors is not None:
            try:
                from .render.viewer3d import SimView3D
                self.view3d = SimView3D(sim, sensors, fps=fps_3d)
                self.view3d.failed.connect(self._drop_3d)
                self.tabs.addTab(self.view3d, '3D')
            except ImportError as e:
                log.warning('3D view disabled (%s) - pip install moderngl', e)
        self.view2d = SimView(sim)
        self.tabs.addTab(self.view2d, 'Top-down')

        self.hud = QtWidgets.QLabel()
        self._hud_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.hud.setTextFormat(QtCore.Qt.PlainText)
        self.hud.setWordWrap(True)          # wraps between readings when the tile is narrow
        self.hud.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.hud)
        self._ui_scale = None
        self._apply_ui_scale()

        self.hud_timer = QtCore.QTimer(self)
        self.hud_timer.timeout.connect(self._update_hud)
        self.hud_timer.start(100)
        QtWidgets.QApplication.instance().installEventFilter(self)

    def reset_inputs(self):
        self.keys.reset_inputs()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._apply_ui_scale()

    def _apply_ui_scale(self):
        f = ui_scale_for(self.width(), self.height())
        if f == self._ui_scale:
            return
        self._ui_scale = f
        self.tabs.setStyleSheet(tab_style(f))
        font = QtGui.QFont(self._hud_font)
        font.setPointSizeF(HUD_FONT_PT * f)
        self.hud.setFont(font)
        self.hud.setStyleSheet(hud_style(f))
        self.view2d.set_ui_scale(f)

    def _drop_3d(self, msg):
        if self.view3d is not None:
            self.tabs.removeTab(self.tabs.indexOf(self.view3d))
            self.view3d = None
            self.view3d_error = msg
            self.view3d_failed.emit(msg)

    # keyboard: an app-level filter so keys work whatever widget has focus
    def eventFilter(self, obj, ev):
        t = ev.type()
        if t in (QtCore.QEvent.KeyPress, QtCore.QEvent.KeyRelease) and self.isActiveWindow():
            if self.embedded and not self.isVisible():
                return False                          # another workbench tab has the keys
            if isinstance(QtWidgets.QApplication.focusWidget(),
                          (QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit, QtWidgets.QComboBox)):
                return False                          # typing, not driving
            if ev.isAutoRepeat():
                return True
            k = ev.key()
            if t == QtCore.QEvent.KeyRelease:
                return self.keys.release(k)
            if k == Q.Key_Tab:
                self.tabs.setCurrentIndex((self.tabs.currentIndex() + 1) % self.tabs.count())
                return True
            if self.keys.press(k):
                return True
            view = self.tabs.currentWidget()
            return bool(view is not None and view.handle_key(k))
        if t == QtCore.QEvent.WindowDeactivate and obj is self.window():
            self.keys.clear()
        return super().eventFilter(obj, ev)

    def _update_hud(self):
        s = self.sim.snapshot()
        ctl = s['controls']
        view = self.tabs.currentWidget()
        telemetry = hud_text([
            '{:.2f} s'.format(s['time']),
            '{:5.2f} m/s'.format(s['speed']),
            'gear {}  {:5.0f} rpm'.format(s['gear'], s['rpm']),
            'thr {:.2f}  brk {:.2f}  str {:+.2f}'.format(ctl.throttle, ctl.brake, ctl.steering),
            'roll {:+.1f}°  pitch {:+.1f}°'.format(math.degrees(s['roll']), math.degrees(s['pitch'])),
            'API' if s['api_control'] else 'MANUAL',
            '[PAUSED]' if s['paused'] else '',
        ])
        ref = s['referee']
        lap_info = ['{} lap {}/{}'.format(ref['mission'], min(len(ref['laps']) + 1, ref['laps_required']),
                                          ref['laps_required'])]
        if ref['laps']:
            lap_info.append('last {:.2f} s'.format(ref['laps'][-1]))
        if ref['complete']:
            lap_info.append('DONE {:.2f} s'.format(ref['run_time']))
        lap_info.append('DOO {}'.format(ref['doo']))
        if ref['crashes']:
            lap_info.append('crashes {}'.format(ref['crashes']))
        if ref['finished']:
            lap_info.append('FINISHED')
        telemetry = telemetry + '   ' + hud_text(lap_info)
        keys = hud_text(['W/S throttle/brake', 'A/D steer', 'R reset', 'P pause',
                         'Tab switch view', view.status() if view is not None else ''])
        self.hud.setText(telemetry + '\n' + keys)


class SimWindow(QtWidgets.QMainWindow):
    """Standalone sim window (python3 -m fsds_lite)."""

    def __init__(self, sim, sensors=None, use_3d=True, title='fsds_lite'):
        super().__init__()
        self.setWindowTitle('{} - {}'.format(title, sim.track.name))
        self.panel = SimPanel(sim, sensors, use_3d=use_3d)
        self.panel.view3d_failed.connect(lambda msg: self.statusBar().showMessage(msg, 15000))
        self.setCentralWidget(self.panel)
        self.resize(int(REF_SIZE[0]), int(REF_SIZE[1]))

    # kept for callers/tests that reach through the window
    @property
    def tabs(self):
        return self.panel.tabs

    @property
    def view3d(self):
        return self.panel.view3d

    @property
    def view2d(self):
        return self.panel.view2d

    @property
    def keys(self):
        return self.panel.keys

    @property
    def hud(self):
        return self.panel.hud
