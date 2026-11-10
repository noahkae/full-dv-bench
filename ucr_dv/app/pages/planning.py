"""
Planning & control page: odometry, mapping, planner, speed profile, lateral
and longitudinal control, mission - plus an instant single-frame preview:
scrub along a track, and see what the stack plans from what the rig sees there.

Parameter edits made here while a bench is running are applied to the
running stack immediately (live tuning).
"""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from ...core.types import COLOR_RGB, ControlCommand, Pose2D
from ...tracks import list_tracks, load_track, track_reference
from .. import theme
from ..widgets import AlgoPicker, Splitter, card, color_legend, label

STAGES = ('odometry', 'mapping', 'planner', 'speed', 'lateral', 'longitudinal', 'mission')


class PlanningPage(QtWidgets.QWidget):
    def __init__(self, state, parent=None):
        super(PlanningPage, self).__init__(parent)
        self.state = state
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.addWidget(label('Planning & control', 'title'))
        outer.addWidget(label('Everything after perception. Changes to parameters apply live to a running bench; '
                              'algorithm swaps apply to the next run.', 'subtitle', True))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        host = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(host)
        scroll.setWidget(host)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pickers = {}
        for i, kind in enumerate(STAGES):
            p = AlgoPicker(kind)
            p.algo_changed.connect(lambda n, k=kind: self._algo(k, n))
            p.param_changed.connect(lambda key, v, k=kind: self._param(k, key, v))
            self.pickers[kind] = p
        self._cols = 0
        self._scroll = scroll
        scroll.viewport().installEventFilter(self)
        self._reflow(2)

        # preview
        prev = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(prev)
        pl.setContentsMargins(0, 0, 0, 0)
        ctl = card('Single-frame preview')
        cl = QtWidgets.QGridLayout(ctl)
        self.track = QtWidgets.QComboBox()
        self.track.addItems(list_tracks())
        self.track.setCurrentText('autocross')
        self.track.currentTextChanged.connect(self._load_track)
        self.pos = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pos.setRange(0, 1000)
        self.pos.valueChanged.connect(lambda _: self._timer.start(30))
        self.off = QtWidgets.QDoubleSpinBox()
        self.off.setRange(-2.0, 2.0)
        self.off.setSingleStep(0.1)
        self.off.setSuffix(' m')
        self.off.setToolTip('lateral offset of the car from the centreline')
        self.off.valueChanged.connect(lambda _: self._timer.start(30))
        self.head = QtWidgets.QDoubleSpinBox()
        self.head.setRange(-30, 30)
        self.head.setSuffix(' deg')
        self.head.setToolTip('heading error of the car relative to the centreline')
        self.head.valueChanged.connect(lambda _: self._timer.start(30))
        self.vel = QtWidgets.QDoubleSpinBox()
        self.vel.setRange(0, 25)
        self.vel.setValue(8)
        self.vel.setSuffix(' m/s')
        self.vel.valueChanged.connect(lambda _: self._timer.start(30))
        # two short rows + the scrubber: fits the pane even in a small window
        cells = [('track', self.track, 0, 0), ('speed', self.vel, 0, 2),
                 ('offset', self.off, 1, 0), ('heading', self.head, 1, 2)]
        for text, w, r, c in cells:
            cl.addWidget(label(text, 'muted'), r, c, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            cl.addWidget(w, r, c + 1)
        self.pos.setToolTip('where along the track centreline the car sits')
        cl.addWidget(label('position', 'muted'), 2, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        cl.addWidget(self.pos, 2, 1, 1, 3)
        cl.setColumnStretch(1, 1)
        cl.setColumnStretch(3, 1)
        cl.setHorizontalSpacing(10)
        pl.addWidget(ctl)
        self.view = pg.PlotWidget(title='car frame: what the planner sees')
        self.view.setAspectLocked(True)
        self.view.showGrid(x=True, y=True, alpha=0.15)
        pl.addWidget(self.view, 3)
        self.speed_plot = pg.PlotWidget(title='speed profile along the planned path')
        self.speed_plot.setLabel('bottom', 'distance along path', 'm')
        self.speed_plot.setLabel('left', 'v', 'm/s')
        self.speed_plot.showGrid(x=True, y=True, alpha=0.15)
        pl.addWidget(self.speed_plot, 1)
        self.info = label('', 'muted', True)
        pl.addWidget(self.info)
        pl.addWidget(label(color_legend() + ' &nbsp; green = path, gold = lookahead target, rings = truth',
                           'muted'))
        outer.addWidget(Splitter([(scroll, 420, 0.45), (prev, 480, 0.55)]), 1)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.preview)
        state.profile_loaded.connect(self.reload)
        state.profile_edited.connect(lambda w: self._timer.start(80))
        self.track_obj = None
        self.reload()
        self._load_track(self.track.currentText())

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Resize and obj is self._scroll.viewport():
            # two columns only when both fit; otherwise the second one would be clipped
            need = max(p.minimumSizeHint().width() for p in self.pickers.values())
            self._reflow(2 if obj.width() >= 2 * need + 40 else 1)
        return super(PlanningPage, self).eventFilter(obj, ev)

    def _reflow(self, cols):
        if cols == self._cols:
            return
        self._cols = cols
        for p in self.pickers.values():
            self.grid.removeWidget(p)
        for r in range(len(STAGES) + 1):
            self.grid.setRowStretch(r, 0)
        for i, kind in enumerate(STAGES):
            self.grid.addWidget(self.pickers[kind], i // cols, i % cols, QtCore.Qt.AlignTop)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1 if cols == 2 else 0)
        self.grid.setRowStretch((len(STAGES) + cols - 1) // cols, 1)

    def reload(self):
        for kind, p in self.pickers.items():
            st = self.state.profile.stages[kind]
            p.set(st['name'], st.get('params', {}))
        self._timer.start(50)

    def _algo(self, kind, name):
        self.state.set_stage(kind, name)
        self._timer.start(50)

    def _param(self, kind, key, value):
        self.state.set_stage_param(kind, key, value)
        self._timer.start(80)

    def _load_track(self, name):
        try:
            self.track_obj = load_track(name, scenery=False)
            self.ref = track_reference(self.track_obj)
        except Exception as e:
            self.info.setText(str(e))
            self.track_obj = None
        self._timer.start(30)

    def preview(self):
        if self.track_obj is None:
            return
        from ...backends.common import SensorSynth
        from ...core.types import GroundTruth, Observation
        from ...pipeline import Stack
        from ...tracks import track_cones
        from ...core.types import to_car
        prof = self.state.profile.copy()
        try:
            stack = Stack(prof, event=self.track_obj.mission)
        except Exception as e:
            self.info.setText('cannot build the stack: {}'.format(e))
            return
        ref = self.ref
        k = int(self.pos.value() / 1000.0 * (len(ref.xy) - 2))
        p0, p1 = ref.xy[k], ref.xy[min(k + 2, len(ref.xy) - 1)]
        yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0]) + math.radians(self.head.value())
        nx, ny = -math.sin(yaw), math.cos(yaw)
        pose = Pose2D(p0[0] + nx * self.off.value(), p0[1] + ny * self.off.value(), yaw)
        synth = SensorSynth(prof.rig, self.track_obj, seed=1,
                            algo_params={s.name: (s.algorithm or {}).get('params', {}) for s in prof.rig.perception()})
        v = self.vel.value()
        truth = {'x': pose.x, 'y': pose.y, 'yaw': yaw, 'speed': v, 'vx': v, 'vy': 0.0, 'yaw_rate': 0.0}
        cones = track_cones(self.track_obj)
        gt = GroundTruth(pose=pose, speed=v, vx=v, cones_world=cones, cones_hit=np.zeros(len(cones), bool))
        obs = Observation(t=0.0, frames=synth.frames(0.0, pose, None), ego=synth.ego(0.0, truth), truth=gt)
        # put the stack's estimate where the car is (odometry starts at the origin otherwise)
        stack.stages['odometry'].pose = pose
        stack.stages['odometry'].speed = v
        if hasattr(stack.stages['odometry'], 'x'):
            stack.stages['odometry'].x = np.array([pose.x, pose.y, pose.yaw, v])
        try:
            out = stack.step(obs, 0.0)
        except Exception as e:
            import traceback
            self.info.setText('stack error: ' + traceback.format_exc()[-600:])
            return
        self.view.clear()
        tc = cones.copy()
        tc[:, :2] = to_car(cones[:, :2], pose)
        near = (np.hypot(tc[:, 0], tc[:, 1]) < 45)
        tc = tc[near]
        self.view.addItem(pg.ScatterPlotItem(pos=tc[:, :2], size=13, brush=None,
                                             pen=[pg.mkPen(*COLOR_RGB[int(c)], width=1) for c in tc[:, 2]]))
        lc = stack.stages['mapping'].local(out.cones, out.pose)
        if len(lc):
            self.view.addItem(pg.ScatterPlotItem(pos=lc[:, :2], size=8, pen=None,
                                                 brush=[pg.mkBrush(*COLOR_RGB[int(c)]) for c in lc[:, 2]]))
        dbg = out.debug.get('planner') or {}
        if 'edges' in dbg and len(dbg['edges']):
            xs, ys = [], []
            for e in dbg['edges']:
                xs += [e[0][0], e[1][0], np.nan]
                ys += [e[0][1], e[1][1], np.nan]
            self.view.plot(xs, ys, pen=pg.mkPen((200, 80, 200, 110), width=1), connect='finite')
        if 'mids' in dbg and len(dbg['mids']):
            self.view.addItem(pg.ScatterPlotItem(pos=dbg['mids'], size=5, symbol='+', pen=pg.mkPen((220, 120, 255))))
        if len(out.path):
            self.view.plot(out.path[:, 0], out.path[:, 1], pen=pg.mkPen((60, 220, 120), width=3))
        if out.target is not None:
            self.view.addItem(pg.ScatterPlotItem(pos=[out.target], size=12, brush=pg.mkBrush(theme.ACCENT_2)))
        L = stack.vehicle
        hl, hw = L.length / 2, L.width / 2
        self.view.plot([hl, hl, -hl, -hl, hl], [hw, -hw, -hw, hw, hw], pen=pg.mkPen('w', width=2))
        d = math.degrees(out.command.steering)
        self.view.plot([L.lf, L.lf + 2.5 * math.cos(out.command.steering)],
                       [0, 2.5 * math.sin(out.command.steering)], pen=pg.mkPen(theme.ACCENT, width=3))
        self.view.setRange(xRange=(-6, 30), yRange=(-15, 15))
        self.speed_plot.clear()
        if len(out.path) > 1 and len(out.speed_profile) == len(out.path):
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(out.path, axis=0).T))])
            self.speed_plot.plot(s, out.speed_profile, pen=pg.mkPen(theme.ACCENT_2, width=2))
        self.speed_plot.addLine(y=out.command.target_speed, pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
        tm = ', '.join('{} {:.2f}'.format(k.replace('perception:', ''), v) for k, v in out.timings_ms.items())
        self.info.setText('steering {:+.1f} deg | target {:.1f} m/s | throttle {:.2f} brake {:.2f} | '
                          '{} cones seen | {} path points | ms: {}'.format(
                              d, out.command.target_speed, out.command.throttle, out.command.brake,
                              len(out.cones), len(out.path), tm))
