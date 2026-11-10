"""Reusable widgets: parameter forms, algorithm pickers, the track view, small helpers."""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from ..core import registry
from ..core.types import COLOR_NAMES, COLOR_RGB, to_world
from . import theme


def label(text, name=None, wrap=False):
    l = QtWidgets.QLabel(text)
    if name:
        l.setObjectName(name)
    l.setWordWrap(wrap)
    return l


def card(title=None):
    box = QtWidgets.QGroupBox(title) if title else QtWidgets.QFrame()
    if not title:
        box.setObjectName('card')
    return box


def button(text, primary=False, tip=None):
    b = QtWidgets.QPushButton(text)
    if primary:
        b.setObjectName('primary')
    if tip:
        b.setToolTip(tip)
    return b


class EmptyHint(QtWidgets.QLabel):
    """Centred muted text floating over an empty list / plot / table: says what will appear and how."""

    def __init__(self, host, text):
        super(EmptyHint, self).__init__(text, host)
        self.setObjectName('empty')
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setWordWrap(True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        host.installEventFilter(self)
        self._fit()
        self.raise_()

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Resize:
            self._fit()
        return False

    def _fit(self):
        self.setGeometry(self.parent().rect().adjusted(24, 24, -24, -24))


def chip(text='', color=None):
    """A small rounded status tag."""
    l = QtWidgets.QLabel()
    l.setObjectName('chip')
    set_chip(l, text, color)
    return l


def set_chip(l, text, color=None):
    l.setText(text)
    l.setStyleSheet('color: {0}; border-color: {0};'.format(color) if color else '')
    l.setVisible(bool(text))


def hbox(*widgets, stretch_last=False, margins=0):
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(margins, margins, margins, margins)
    for x in widgets:
        if x is None:
            lay.addStretch(1)
        elif isinstance(x, QtWidgets.QLayout):
            lay.addLayout(x)
        else:
            lay.addWidget(x)
    if stretch_last:
        lay.addStretch(1)
    return w


class ParamForm(QtWidgets.QWidget):
    """A form generated from a list of registry.P. Emits changed(name, value)."""
    changed = QtCore.pyqtSignal(str, object)

    def __init__(self, params, values=None, parent=None):
        super(ParamForm, self).__init__(parent)
        self.params = list(params)
        self.editors = {}
        lay = QtWidgets.QFormLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setLabelAlignment(QtCore.Qt.AlignRight)
        values = values or {}
        for p in self.params:
            v = values.get(p.name, p.default)
            ed = self._editor(p, v)
            self.editors[p.name] = ed
            lab = QtWidgets.QLabel(p.name.replace('_', ' ') + (' [{}]'.format(p.unit) if p.unit else ''))
            lab.setToolTip(p.help)
            ed.setToolTip(p.help)
            lay.addRow(lab, ed)
        if not self.params:
            lay.addRow(label('no parameters', 'muted'))

    def _editor(self, p, v):
        name = p.name
        if p.type is bool:
            ed = QtWidgets.QCheckBox()
            ed.setChecked(bool(v))
            ed.toggled.connect(lambda x, n=name: self.changed.emit(n, bool(x)))
        elif p.choices:
            ed = QtWidgets.QComboBox()
            ed.addItems(p.choices)
            ed.setCurrentText(str(v))
            ed.currentTextChanged.connect(lambda x, n=name: self.changed.emit(n, x))
        elif p.type is int:
            ed = QtWidgets.QSpinBox()
            ed.setRange(int(p.lo if p.lo is not None else -10 ** 6), int(p.hi if p.hi is not None else 10 ** 6))
            ed.setSingleStep(int(p.step or 1))
            ed.setValue(int(v))
            ed.valueChanged.connect(lambda x, n=name: self.changed.emit(n, int(x)))
        elif p.type is float:
            ed = QtWidgets.QDoubleSpinBox()
            ed.setDecimals(3 if (p.step or 1) < 0.1 else 2)
            ed.setRange(float(p.lo if p.lo is not None else -1e6), float(p.hi if p.hi is not None else 1e6))
            ed.setSingleStep(float(p.step or 0.1))
            ed.setValue(float(v))
            ed.setKeyboardTracking(False)
            ed.valueChanged.connect(lambda x, n=name: self.changed.emit(n, float(x)))
        else:
            ed = QtWidgets.QLineEdit(str(v))
            ed.editingFinished.connect(lambda n=name, e=None: self.changed.emit(n, self.editors[n].text()))
        return ed

    def values(self):
        out = {}
        for p in self.params:
            ed = self.editors[p.name]
            if isinstance(ed, QtWidgets.QCheckBox):
                out[p.name] = ed.isChecked()
            elif isinstance(ed, QtWidgets.QComboBox):
                out[p.name] = ed.currentText()
            elif isinstance(ed, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                out[p.name] = ed.value()
            else:
                out[p.name] = ed.text()
        return out


class AlgoPicker(QtWidgets.QGroupBox):
    """Choose one registered algorithm for a stage and edit its parameters.

    Emits algo_changed(name) and param_changed(name, value)."""
    algo_changed = QtCore.pyqtSignal(str)
    param_changed = QtCore.pyqtSignal(str, object)

    def __init__(self, kind, title=None, parent=None):
        super(AlgoPicker, self).__init__(title or registry.STAGE_LABELS.get(kind, kind), parent)
        self.kind = kind
        self.entries = registry.entries(kind)
        lay = QtWidgets.QVBoxLayout(self)
        self.combo = QtWidgets.QComboBox()
        for name, e in self.entries.items():
            self.combo.addItem(e.label, name)
        lay.addWidget(self.combo)
        self.help = label('', 'muted', wrap=True)
        lay.addWidget(self.help)
        self.form_holder = QtWidgets.QVBoxLayout()
        lay.addLayout(self.form_holder)
        self.form = None
        self._quiet = False
        self.combo.currentIndexChanged.connect(self._on_combo)

    def set(self, name, params):
        self._quiet = True
        i = self.combo.findData(name)
        if i >= 0:
            self.combo.setCurrentIndex(i)
        self._build(name, params)
        self._quiet = False

    def current(self):
        return self.combo.currentData()

    def _on_combo(self, _):
        name = self.combo.currentData()
        self._build(name, {})
        if not self._quiet:
            self.algo_changed.emit(name)

    def _build(self, name, params):
        e = self.entries.get(name)
        if self.form is not None:
            self.form.setParent(None)
            self.form.deleteLater()
        if e is None:
            self.form = None
            return
        refs = ('<br><i>Based on: {}</i>'.format(', '.join(e.refs))) if e.refs else ''
        self.help.setText(e.help.split('\n')[0] + refs)
        self.form = ParamForm(e.params, e.resolve(params))
        self.form.changed.connect(self.param_changed)
        self.form_holder.addWidget(self.form)


def _car_outline(x, y, yaw, length=2.6, width=1.3):
    hl, hw = length / 2, width / 2
    box = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw], [hl + 0.6, 0], [hl, -hw]])
    c, s = math.cos(yaw), math.sin(yaw)
    return box @ np.array([[c, s], [-s, c]]) + (x, y)


class TrackView(pg.PlotWidget):
    """Top-down world view: track cones, reference, trail, car, detections, map, path."""

    def __init__(self, parent=None):
        super(TrackView, self).__init__(parent)
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.15)
        self.follow = True
        self.ref = self.plot(pen=pg.mkPen((90, 95, 105), width=1, style=QtCore.Qt.DashLine))
        self.cones = pg.ScatterPlotItem(size=7, pen=None)
        self.hit = pg.ScatterPlotItem(size=11, pen=pg.mkPen(theme.BAD, width=2), brush=None, symbol='x')
        self.map = pg.ScatterPlotItem(size=12, pen=pg.mkPen((200, 200, 200, 120), width=1), brush=None)
        self.det = pg.ScatterPlotItem(size=9, symbol='+', pen=pg.mkPen('w', width=1.5))
        self.trail = self.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        self.path = self.plot(pen=pg.mkPen((60, 220, 120), width=3))
        self.target = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(theme.ACCENT_2), pen=None)
        self.car = self.plot(pen=pg.mkPen('w', width=2))
        self.est = self.plot(pen=pg.mkPen((120, 170, 255), width=1, style=QtCore.Qt.DotLine))
        for it in (self.cones, self.hit, self.map, self.det, self.target):
            self.addItem(it)
        self.vehicle = (2.6, 1.3)
        self._trail = []

    def set_track(self, track, ref=None, vehicle=None):
        cols = [COLOR_RGB.get({'yellow': 0, 'blue': 1, 'orange': 2, 'big_orange': 3}.get(c, 4)) for c in track.colors]
        self.cones.setData(pos=track.positions, brush=[pg.mkBrush(*c) for c in cols],
                           size=[10 if c == 'big_orange' else 7 for c in track.colors])
        if ref is not None:
            self.ref.setData(ref.xy[:, 0], ref.xy[:, 1])
        if vehicle is not None:
            self.vehicle = (vehicle.length, vehicle.width)
        self.track = track
        self._trail = []
        self.trail.setData([], [])
        self.hit.setData([])
        self.det.setData([])
        self.map.setData([])
        self.path.setData([], [])
        self.autoRange()

    def clear_run(self):
        self._trail = []
        for it in (self.trail, self.path, self.car, self.est):
            it.setData([], [])
        for it in (self.hit, self.det, self.map, self.target):
            it.setData([])

    def update_frame(self, frame, hit_mask=None, show_det=True, show_map=True):
        car = frame.car
        out = frame.out
        self._trail.append((car['x'], car['y']))
        if len(self._trail) > 20000:
            self._trail = self._trail[-20000:]
        tr = np.array(self._trail)
        self.trail.setData(tr[:, 0], tr[:, 1])
        box = _car_outline(car['x'], car['y'], car['yaw'], *self.vehicle)
        self.car.setData(box[:, 0], box[:, 1])
        est = _car_outline(out.pose.x, out.pose.y, out.pose.yaw, *self.vehicle)
        self.est.setData(est[:, 0], est[:, 1])
        # car-frame things are drawn using the stack's own pose estimate (what it believes)
        if len(out.path) > 1:
            pw = to_world(out.path, out.pose)
            self.path.setData(pw[:, 0], pw[:, 1])
        if out.target is not None:
            tw = to_world(np.array([out.target]), out.pose)
            self.target.setData(pos=tw)
        if show_det and len(out.cones):
            cw = to_world(out.cones[:, :2], out.pose)
            brushes = [pg.mkBrush(*COLOR_RGB.get(int(c), (200, 200, 200))) for c in out.cones[:, 2]]
            self.det.setData(pos=cw, pen=[pg.mkPen(*COLOR_RGB.get(int(c), (230, 230, 230)), width=2)
                                          for c in out.cones[:, 2]], brush=brushes)
        else:
            self.det.setData([])
        if show_map and len(out.map_world):
            self.map.setData(pos=out.map_world[:, :2])
        else:
            self.map.setData([])
        if hit_mask is not None and hasattr(self, 'track') and hit_mask.any():
            self.hit.setData(pos=self.track.positions[hit_mask])
        if self.follow:
            vb = self.getViewBox()
            (x0, x1), (y0, y1) = vb.viewRange()
            hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
            if hx > 60:
                hx = hy = 25
            vb.setRange(xRange=(car['x'] - hx, car['x'] + hx), yRange=(car['y'] - hy, car['y'] + hy), padding=0)


class Telemetry(QtWidgets.QWidget):
    """Stacked strip charts: speed vs target, steering, cross-track error, compute."""

    def __init__(self, parent=None):
        super(Telemetry, self).__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw)
        self.p_speed = self.glw.addPlot(row=0, col=0)
        self.p_speed.setLabel('left', 'speed', 'm/s')
        self.p_speed.addLegend(offset=(5, 5))
        self.c_v = self.p_speed.plot(pen=pg.mkPen('w', width=1.5), name='actual')
        self.c_vt = self.p_speed.plot(pen=pg.mkPen(theme.ACCENT_2, width=1.2), name='target')
        self.p_steer = self.glw.addPlot(row=1, col=0)
        self.p_steer.setLabel('left', 'steer', 'deg')
        self.c_sc = self.p_steer.plot(pen=pg.mkPen(theme.ACCENT, width=1.2))
        self.c_sa = self.p_steer.plot(pen=pg.mkPen('w', width=1))
        self.p_cte = self.glw.addPlot(row=2, col=0)
        self.p_cte.setLabel('left', 'CTE', 'm')
        self.c_cte = self.p_cte.plot(pen=pg.mkPen((120, 200, 255), width=1.2))
        self.p_ms = self.glw.addPlot(row=3, col=0)
        self.p_ms.setLabel('left', 'compute', 'ms')
        self.p_ms.setLabel('bottom', 't', 's')
        self.c_ms = self.p_ms.plot(pen=pg.mkPen((180, 140, 255), width=1))
        for p in (self.p_steer, self.p_cte, self.p_ms):
            p.setXLink(self.p_speed)
        for p in (self.p_speed, self.p_steer, self.p_cte, self.p_ms):
            p.showGrid(x=True, y=True, alpha=0.15)

    def set_log(self, a):
        if not len(a['t']):
            for c in (self.c_v, self.c_vt, self.c_sc, self.c_sa, self.c_cte, self.c_ms):
                c.setData([], [])
            return
        t = a['t']
        self.c_v.setData(t, a['speed'])
        self.c_vt.setData(t, a['target_speed'])
        self.c_sc.setData(t, np.degrees(a['steer_cmd']))
        self.c_sa.setData(t, np.degrees(a['steer']))
        self.c_cte.setData(t, a['cte'])
        self.c_ms.setData(t, a['compute_ms'])


def result_html(res, extra=''):
    if res is None:
        return ''
    ok = res.get('complete')
    color = theme.GOOD if ok else theme.BAD
    head = '<span style="font-size:15pt;color:{};font-weight:bold">{}</span>'.format(color, res.get('summary', ''))
    rows = [('laps', ', '.join('{:.2f}'.format(x) for x in res.get('laps', [])) or '-'),
            ('cones (DOO)', res.get('doo')), ('off course', res.get('off_course')),
            ('max / RMS CTE', '{} / {} m'.format(res.get('max_cte'), res.get('rms_cte'))),
            ('max speed', '{} m/s'.format(res.get('max_speed', '-'))),
            ('peak lat. accel', '{} g'.format(res.get('max_lat_g', '-'))),
            ('compute p50/p95', '{} / {} ms'.format(res.get('compute_ms_p50', '-'), res.get('compute_ms_p95', '-'))),
            ('sim / wall', '{} s / {} s ({}x)'.format(res.get('sim_s'), res.get('wall_s'), res.get('speedup'))),
            ('uses ground truth', 'YES (not gate-eligible)' if res.get('uses_gt') else 'no')]
    table = ''.join('<tr><td style="color:{};padding-right:12px">{}</td><td>{}</td></tr>'.format(theme.MUTED, k, v)
                    for k, v in rows)
    err = ''
    if res.get('error'):
        err = '<pre style="color:{}">{}</pre>'.format(theme.BAD, res['error'][-800:].replace('<', '&lt;'))
    return head + '<table style="margin-top:6px">' + table + '</table>' + extra + err


def color_legend():
    return ' '.join('<span style="color:rgb{}">&#9679;</span> {}'.format(COLOR_RGB[k], v)
                    for k, v in COLOR_NAMES.items())


class Splitter(QtWidgets.QSplitter):
    """A splitter that keeps every pane usable.

    panes: list of (widget, min_px, ratio). Panes never collapse, never shrink
    below `min_px`, and are laid out by `ratio` of the actual width/height
    whenever the splitter is resized - until the user drags a handle, after
    which their proportions are kept instead of fixed pixel sizes.
    """

    def __init__(self, panes, orientation=QtCore.Qt.Horizontal, parent=None):
        super(Splitter, self).__init__(orientation, parent)
        self.setChildrenCollapsible(False)
        self.setHandleWidth(6)
        self._ratios = []
        horizontal = orientation == QtCore.Qt.Horizontal
        for i, (w, min_px, ratio) in enumerate(panes):
            if horizontal:
                w.setMinimumWidth(min_px)
            else:
                w.setMinimumHeight(min_px)
            self.addWidget(w)
            self.setStretchFactor(i, max(1, int(round(ratio * 10))))
            self._ratios.append(float(ratio))
        self.splitterMoved.connect(self._remember)

    def _remember(self, *_):
        sizes = self.sizes()
        total = float(sum(sizes)) or 1.0
        self._ratios = [s / total for s in sizes]

    def _apply(self):
        total = self.width() if self.orientation() == QtCore.Qt.Horizontal else self.height()
        total -= self.handleWidth() * (self.count() - 1)
        if total <= 0:
            return
        r = sum(self._ratios) or 1.0
        self.setSizes([int(total * x / r) for x in self._ratios])

    def showEvent(self, ev):
        super(Splitter, self).showEvent(ev)
        self._apply()

    def resizeEvent(self, ev):
        super(Splitter, self).resizeEvent(ev)
        self._apply()
