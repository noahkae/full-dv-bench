"""
Sensor rig page: add/remove/mount sensors and see what they can see.

Everything here is instant because it uses the detection-level models:
    - top-down coverage heatmap (P(detect) of an isolated cone) + FOV wedges
    - side view: LiDAR laser rings vs cone heights (why 16 channels lose cones at 15 m)
    - 50 % detection range per sensor, straight ahead
"""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from ... import paths
from ...core.registry import P
from ...sensors.models import BIG_H, CONE_H, coverage_grid, make_model
from ...sensors.rig import (DEFAULT_ALGORITHM, PRESETS, TYPE_DEFAULTS, Mount, from_preset)
from .. import theme
from ..widgets import ParamForm, Splitter, button, card, label

SENSOR_COLORS = [(214, 0, 28), (255, 205, 0), (80, 170, 255), (90, 220, 140), (220, 120, 255), (255, 150, 60)]
UNITS = {'channels': '', 'vfov_upper': 'deg', 'vfov_lower': 'deg', 'hfov': 'deg', 'h_res_deg': 'deg',
         'rate_hz': 'Hz', 'max_range': 'm', 'range_noise': 'm', 'width': 'px', 'height': 'px',
         'min_box_px': 'px', 'pixel_noise': 'px', 'baseline': 'm', 'disparity_noise': 'px',
         'false_positive_rate': '/frame', 'render_width': 'px', 'noise': '', 'color_accuracy': ''}


def forward_range(spec, algo_params, height=CONE_H):
    """Largest straight-ahead distance with P(detect) >= 0.5, for a cone on the sensor's axis."""
    a = math.radians(spec.mount.yaw)
    rs = np.arange(0.5, 120.0, 0.25)
    xs = spec.mount.x + rs * math.cos(a)
    ys = spec.mount.y + rs * math.sin(a)
    model = make_model(spec)
    if model is None:
        return None
    from ...sensors.models import CameraModel, TimelessClassicModel
    cones = np.column_stack([xs, ys, np.zeros(len(xs)), np.ones(len(xs))])
    h = np.full(len(xs), height)
    if isinstance(model, TimelessClassicModel):
        ok = model.visible(cones, h)
    elif isinstance(model, CameraModel):
        min_px = float(algo_params.get('min_box_px', spec.p('min_box_px')))
        ok = (model.box_px(rs, h) >= min_px) & (rs <= spec.p('max_range'))
        half_v = math.radians(spec.vfov()) / 2.0
        ok &= (np.arctan2(spec.mount.z, rs) - math.radians(spec.mount.pitch)) <= half_v
    else:
        ok = (model.expected_points(rs, h) >= float(algo_params.get('min_cluster_points', 3))) & \
             (rs <= spec.p('max_range'))
    if not ok.any():
        return (0.0, 0.0)
    idx = np.nonzero(ok)[0]
    return float(rs[idx[0]]), float(rs[idx[-1]])


class RigPage(QtWidgets.QWidget):
    def __init__(self, state, parent=None):
        super(RigPage, self).__init__(parent)
        self.state = state
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.addWidget(label('Sensor rig', 'title'))
        outer.addWidget(label('Which sensors, how many, where. The previews use the same models the timeless '
                              'bench drives with; the 3D bench ray-casts / renders the real thing.', 'subtitle', True))

        # ---- left: sensor list
        left = card('Sensors')
        ll = QtWidgets.QVBoxLayout(left)
        self.list = QtWidgets.QListWidget()
        self.list.currentRowChanged.connect(self._select)
        ll.addWidget(self.list, 1)
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(list(PRESETS))
        ll.addWidget(self.preset)
        row = QtWidgets.QHBoxLayout()
        b_add = button('Add', primary=True)
        b_dup = button('Duplicate')
        b_rm = button('Remove')
        for b in (b_add, b_dup, b_rm):
            row.addWidget(b)
        ll.addLayout(row)
        b_add.clicked.connect(self._add)
        b_dup.clicked.connect(self._duplicate)
        b_rm.clicked.connect(self._remove)
        veh = QtWidgets.QFormLayout()
        self.vehicle = QtWidgets.QComboBox()
        self.vehicle.addItems(paths.list_vehicles())
        self.vehicle.currentTextChanged.connect(self._vehicle)
        self.hz = QtWidgets.QDoubleSpinBox()
        self.hz.setRange(5, 100)
        self.hz.setSuffix(' Hz')
        self.hz.valueChanged.connect(self._hz)
        veh.addRow('vehicle', self.vehicle)
        veh.addRow('pipeline rate', self.hz)
        ll.addLayout(veh)

        # ---- middle: editor
        self.editor_box = card('Selected sensor')
        el = QtWidgets.QVBoxLayout(self.editor_box)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.editor = QtWidgets.QWidget()
        self.editor_lay = QtWidgets.QVBoxLayout(self.editor)
        scroll.setWidget(self.editor)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        el.addWidget(scroll)

        # ---- right: previews
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(label('Coverage of', 'muted'))
        self.cov_mode = QtWidgets.QComboBox()
        self.cov_mode.addItems(['all sensors (best)', 'selected sensor'])
        self.cov_mode.currentIndexChanged.connect(self.refresh_preview)
        top.addWidget(self.cov_mode)
        top.addWidget(label('cone', 'muted'))
        self.cone_kind = QtWidgets.QComboBox()
        self.cone_kind.addItems(['small (0.325 m)', 'big orange (0.505 m)'])
        self.cone_kind.currentIndexChanged.connect(self.refresh_preview)
        top.addWidget(self.cone_kind)
        top.addStretch(1)
        rl.addLayout(top)
        self.top = pg.PlotWidget()
        self.top.setAspectLocked(True)
        self.top.showGrid(x=True, y=True, alpha=0.15)
        self.img = pg.ImageItem()
        self.top.addItem(self.img)
        cmap = pg.ColorMap([0.0, 0.5, 1.0], [(30, 33, 39), (120, 30, 40), (255, 205, 0)])
        self.img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        self.wedges = []
        rl.addWidget(self.top, 3)
        self.side = pg.PlotWidget()
        self.side.showGrid(x=True, y=True, alpha=0.15)
        self.side.setLabel('bottom', 'distance ahead', 'm')
        self.side.setLabel('left', 'height', 'm')
        rl.addWidget(self.side, 2)
        self.ranges = QtWidgets.QTextBrowser()
        self.ranges.setMaximumHeight(130)
        rl.addWidget(self.ranges)
        outer.addWidget(Splitter([(left, 270, 0.22), (self.editor_box, 390, 0.28), (right, 400, 0.50)]), 1)

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.refresh_preview)
        state.profile_loaded.connect(self.reload)
        self.reload()

    # ---- list --------------------------------------------------------------
    @property
    def rig(self):
        return self.state.profile.rig

    def reload(self):
        self._quiet = True
        self.vehicle.setCurrentText(self.state.profile.vehicle)
        self.hz.setValue(self.state.profile.pipeline_hz)
        self._quiet = False
        self._fill_list(0)

    def _fill_list(self, select=None):
        cur = self.list.currentRow() if select is None else select
        self.list.blockSignals(True)
        self.list.clear()
        for i, s in enumerate(self.rig.sensors):
            it = QtWidgets.QListWidgetItem('{}{}  ({})'.format('' if s.enabled else '[off] ', s.name, s.type))
            if s.is_perception:
                c = SENSOR_COLORS[i % len(SENSOR_COLORS)]
                it.setForeground(pg.mkColor(*c))
            self.list.addItem(it)
        self.list.blockSignals(False)
        if self.rig.sensors:
            self.list.setCurrentRow(max(0, min(cur, len(self.rig.sensors) - 1)))
            self._select(self.list.currentRow())
        else:
            self._select(-1)

    def _add(self):
        preset = self.preset.currentText()
        spec = from_preset(preset)
        spec.name = self.rig.unique_name(spec.name)
        self.rig.sensors.append(spec)
        self.state.rig_edited()
        self._fill_list(len(self.rig.sensors) - 1)

    def _duplicate(self):
        i = self.list.currentRow()
        if i < 0:
            return
        from ...sensors.rig import SensorSpec
        s = SensorSpec.from_dict(self.rig.sensors[i].to_dict())
        s.name = self.rig.unique_name(s.name)
        s.mount.y = -s.mount.y if abs(s.mount.y) > 0.01 else s.mount.y + 0.3
        self.rig.sensors.append(s)
        self.state.rig_edited()
        self._fill_list(len(self.rig.sensors) - 1)

    def _remove(self):
        i = self.list.currentRow()
        if i < 0:
            return
        del self.rig.sensors[i]
        self.state.rig_edited()
        self._fill_list(max(0, i - 1))

    def _vehicle(self, name):
        if not getattr(self, '_quiet', False) and name and name != self.state.profile.vehicle:
            self.state.profile.vehicle = name
            self.state.meta_edited()

    def _hz(self, v):
        if not getattr(self, '_quiet', False):
            self.state.profile.pipeline_hz = float(v)
            self.state.meta_edited()

    # ---- editor --------------------------------------------------------------
    def _select(self, i):
        while self.editor_lay.count():
            w = self.editor_lay.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        if i < 0 or i >= len(self.rig.sensors):
            self.editor_lay.addWidget(label('No sensor selected - add one from a preset.', 'muted'))
            self.refresh_preview()
            return
        s = self.rig.sensors[i]
        head = QtWidgets.QFormLayout()
        w = QtWidgets.QWidget()
        w.setLayout(head)
        name = QtWidgets.QLineEdit(s.name)
        name.editingFinished.connect(lambda: self._rename(s, name.text()))
        en = QtWidgets.QCheckBox('enabled')
        en.setChecked(s.enabled)
        en.toggled.connect(lambda v: self._set(s, 'enabled', v))
        head.addRow('name', name)
        head.addRow('', en)
        head.addRow('type', label(s.type + ('  -  ' + s.preset if s.preset else ''), 'muted'))
        if s.is_perception and s.type != 'timeless_classic':
            fid = QtWidgets.QComboBox()
            fid.addItems(['auto', 'raw', 'detection'])
            fid.setCurrentText(s.fidelity)
            fid.setToolTip('auto: real point clouds / images on the 3D bench, synthetic detections elsewhere.\n'
                           'detection: always the synthetic model (isolate planning from perception).')
            fid.currentTextChanged.connect(lambda v: self._set(s, 'fidelity', v))
            head.addRow('fidelity', fid)
        self.editor_lay.addWidget(w)

        mount_params = [P('x', s.mount.x, -2.0, 3.0, 0.05, 'm', 'forward of the car origin'),
                        P('y', s.mount.y, -1.0, 1.0, 0.05, 'm', 'left of the centreline'),
                        P('z', s.mount.z, 0.0, 2.5, 0.05, 'm', 'above the ground'),
                        P('roll', s.mount.roll, -30.0, 30.0, 0.5, 'deg'),
                        P('pitch', s.mount.pitch, -30.0, 30.0, 0.5, 'deg', 'positive looks down'),
                        P('yaw', s.mount.yaw, -180.0, 180.0, 1.0, 'deg', 'positive looks left')]
        mbox = card('Mount')
        ml = QtWidgets.QVBoxLayout(mbox)
        mf = ParamForm(mount_params)
        mf.changed.connect(lambda k, v: self._mount(s, k, v))
        ml.addWidget(mf)
        self.editor_lay.addWidget(mbox)

        defaults = TYPE_DEFAULTS.get(s.type, {})
        if defaults:
            ps = []
            for k, d in defaults.items():
                v = s.params.get(k, d)
                if isinstance(d, bool):
                    ps.append(P(k, bool(v)))
                elif isinstance(d, int) and not isinstance(d, bool):
                    ps.append(P(k, int(v), 0, 100000, 1, UNITS.get(k, '')))
                else:
                    ps.append(P(k, float(v), -1000.0, 100000.0, 0.01 if abs(float(d)) < 1 else 1.0, UNITS.get(k, '')))
            pbox = card('Sensor parameters')
            pl = QtWidgets.QVBoxLayout(pbox)
            pf = ParamForm(ps)
            pf.changed.connect(lambda k, v: self._param(s, k, v))
            pl.addWidget(pf)
            self.editor_lay.addWidget(pbox)
        if s.is_perception and s.type != 'timeless_classic':
            self.editor_lay.addWidget(label('Detection algorithm: set on the Perception page ({}).'.format(
                (s.algorithm or DEFAULT_ALGORITHM.get(s.type, {})).get('name', '-')), 'muted', True))
        self.editor_lay.addStretch(1)
        self.refresh_preview()

    def _rename(self, s, new):
        new = new.strip()
        if new and new != s.name and new not in [x.name for x in self.rig.sensors]:
            s.name = new
            self.state.rig_edited()
            self._fill_list()

    def _set(self, s, attr, v):
        setattr(s, attr, v)
        self.state.rig_edited()
        if attr == 'enabled':
            self._fill_list()
        self._timer.start(150)

    def _mount(self, s, k, v):
        setattr(s.mount, k, float(v))
        self.state.rig_edited()
        self._timer.start(150)

    def _param(self, s, k, v):
        s.params[k] = v
        self.state.rig_edited()
        self._timer.start(150)

    # ---- preview ------------------------------------------------------------
    def refresh_preview(self):
        sensors = [s for s in self.rig.sensors if s.enabled and s.is_perception]
        i = self.list.currentRow()
        sel = self.rig.sensors[i] if 0 <= i < len(self.rig.sensors) else None
        height = BIG_H if self.cone_kind.currentIndex() == 1 else CONE_H
        show = [sel] if (self.cov_mode.currentIndex() == 1 and sel is not None and sel.is_perception) else sensors
        extent, step = 40.0, 0.5
        prob = None
        xs = ys = None
        for s in show:
            algo = (s.algorithm or {}).get('params', {})
            xs, ys, pr = coverage_grid(s, extent=extent, step=step, height=height, algo_params=algo)
            prob = pr if prob is None else np.maximum(prob, pr)
        for w in self.wedges:
            self.top.removeItem(w)
        self.wedges = []
        if prob is not None:
            self.img.setImage(prob.T, levels=(0.0, 1.0))
            self.img.setRect(QtCore.QRectF(xs[0] - step / 2, ys[0] - step / 2, len(xs) * step, len(ys) * step))
        else:
            self.img.clear()
        # car outline
        veh = (2.6, 1.3)
        try:
            from ...vehicle import load_vehicle
            vp = load_vehicle(self.state.profile.vehicle)
            veh = (vp.length, vp.width)
        except Exception:
            pass
        hl, hw = veh[0] / 2, veh[1] / 2
        car = self.top.plot([hl, hl, -hl, -hl, hl], [hw, -hw, -hw, hw, hw], pen=pg.mkPen('w', width=2))
        self.wedges.append(car)
        # FOV wedges + mount markers
        for idx, s in enumerate(self.rig.sensors):
            if not (s.enabled and s.is_perception):
                continue
            c = SENSOR_COLORS[idx % len(SENSOR_COLORS)]
            m = s.mount
            hf = s.p('hfov') or s.p('fov') or 110.0
            rng = min(float(s.p('max_range') or s.p('max_dist') or 30.0), extent)
            if hf < 359:
                a0, a1 = math.radians(m.yaw - hf / 2), math.radians(m.yaw + hf / 2)
                arc = np.linspace(a0, a1, 40)
                px = np.concatenate([[m.x], m.x + rng * np.cos(arc), [m.x]])
                py = np.concatenate([[m.y], m.y + rng * np.sin(arc), [m.y]])
            else:
                arc = np.linspace(0, 2 * math.pi, 80)
                px, py = m.x + rng * np.cos(arc), m.y + rng * np.sin(arc)
            w = self.top.plot(px, py, pen=pg.mkPen(c, width=1.5, style=QtCore.Qt.DashLine))
            self.wedges.append(w)
            dot = pg.ScatterPlotItem([m.x], [m.y], size=10, brush=pg.mkBrush(*c), pen=None)
            self.top.addItem(dot)
            self.wedges.append(dot)
        self.top.setRange(xRange=(-extent / 2, extent), yRange=(-extent * 0.6, extent * 0.6))
        self._side(sel, height)
        self._ranges(sensors)

    def _side(self, sel, height):
        self.side.clear()
        for d in (5, 10, 15, 20, 25, 30):
            for hh, col in ((CONE_H, (255, 205, 0)), (BIG_H, (255, 120, 0))):
                if abs(hh - height) > 0.01 and hh != CONE_H:
                    continue
                self.side.plot([d - 0.11, d, d + 0.11, d - 0.11], [0, hh, 0, 0], pen=pg.mkPen(col, width=2))
        self.side.plot([0, 35], [0, 0], pen=pg.mkPen((120, 120, 120)))
        if sel is not None and sel.type == 'lidar':
            m = sel.mount
            ch = int(sel.p('channels'))
            lo, hi = sel.p('vfov_lower'), sel.p('vfov_upper')
            for e in (np.linspace(hi, lo, ch) if ch > 1 else [hi]):
                r = 35.0
                z_end = m.z + r * math.tan(math.radians(e - m.pitch))
                if z_end < 0:
                    r = m.z / max(1e-6, math.tan(math.radians(-(e - m.pitch))))
                    z_end = 0.0
                self.side.plot([m.x, m.x + r], [m.z, z_end], pen=pg.mkPen(214, 0, 28, 160))
            self.side.setTitle('{}: {} laser rings, {:.2f} deg apart - count the rings crossing each cone'.format(
                sel.name, ch, (hi - lo) / max(1, ch - 1)))
        elif sel is not None and sel.type in ('camera', 'stereo'):
            m = sel.mount
            half_v = sel.vfov() / 2.0
            for e in (-half_v, half_v):
                ang = math.radians(e - m.pitch)
                r = 35.0
                z_end = m.z + r * math.tan(ang)
                if z_end < 0:
                    r = m.z / max(1e-6, -math.tan(ang))
                    z_end = 0
                self.side.plot([m.x, m.x + r], [m.z, z_end], pen=pg.mkPen(255, 205, 0, 200))
            self.side.setTitle('{}: vertical FOV {:.0f} deg - cones closer than the lower ray are invisible'.format(
                sel.name, sel.vfov()))
        else:
            self.side.setTitle('select a LiDAR or camera to see its side view')
        self.side.setYRange(-0.1, 2.0)
        self.side.setXRange(0, 32)

    def _ranges(self, sensors):
        rows = []
        for s in sensors:
            algo = (s.algorithm or {}).get('params', {})
            r_small = forward_range(s, algo, CONE_H)
            r_big = forward_range(s, algo, BIG_H)
            extra = ''
            if s.type == 'lidar':
                pts = int(s.p('channels')) * int(round(float(s.p('hfov')) / float(s.p('h_res_deg'))))
                extra = '{:,} pts/scan @ {:g} Hz'.format(pts, s.p('rate_hz'))
            elif s.type in ('camera', 'stereo'):
                extra = 'f = {:.0f} px, {}x{} @ {:g} Hz{}'.format(s.focal_px(), int(s.p('width')), int(s.p('height')),
                                                                 s.p('rate_hz'), ', colour' if True else '')
            rows.append('<tr><td><b>{}</b></td><td>{}</td><td>{:.1f} - {:.1f} m</td><td>{:.1f} - {:.1f} m</td>'
                        '<td style="color:{}">{}</td></tr>'.format(
                            s.name, s.type, r_small[0], r_small[1], r_big[0], r_big[1], theme.MUTED, extra))
        self.ranges.setHtml('<table cellpadding="3"><tr style="color:{}"><td>sensor</td><td>type</td>'
                            '<td>small cone straight ahead</td><td>big cone</td><td></td></tr>{}</table>'.format(
                                theme.MUTED, ''.join(rows)) if rows else 'no perception sensors')
