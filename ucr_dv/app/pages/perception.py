"""
Perception page: the detection algorithm per sensor, the fusion, and an
evaluation that drives a ground-truth lap and scores every sensor + the
fused output against the true cones.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from ...core import registry
from ...core.types import COLOR_RGB
from ...perception.metrics import range_labels
from ...tracks import list_tracks
from .. import theme
from ..state import Worker
from ..widgets import AlgoPicker, EmptyHint, Splitter, button, card, color_legend, label

PALETTE = [(214, 0, 28), (255, 205, 0), (80, 170, 255), (90, 220, 140), (220, 120, 255), (255, 255, 255)]


class PerceptionPage(QtWidgets.QWidget):
    def __init__(self, state, parent=None):
        super(PerceptionPage, self).__init__(parent)
        self.state = state
        self.worker = None
        self.frames = []
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.addWidget(label('Perception', 'title'))
        outer.addWidget(label('Pick how each sensor turns its data into cones and how the sensors are fused, then '
                              'measure it: a ground-truth driver laps the track while every frame is scored.',
                              'subtitle', True))

        # ---- algorithms
        left = QtWidgets.QScrollArea()
        left.setWidgetResizable(True)
        self.algo_host = QtWidgets.QWidget()
        self.algo_lay = QtWidgets.QVBoxLayout(self.algo_host)
        left.setWidget(self.algo_host)
        left.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # ---- evaluation
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        ctl = card('Evaluate')
        cl = QtWidgets.QHBoxLayout(ctl)
        self.track = QtWidgets.QComboBox()
        self.track.addItems(list_tracks())
        self.track.setCurrentText('autocross')
        self.bench = QtWidgets.QComboBox()
        self.bench.addItem('timeless (detection models, instant)', 'timeless')
        self.bench.addItem('3D (raw point clouds / images)', 'lite')
        self.speed = QtWidgets.QDoubleSpinBox()
        self.speed.setRange(1, 20)
        self.speed.setValue(6)
        self.speed.setSuffix(' m/s')
        self.dist = QtWidgets.QDoubleSpinBox()
        self.dist.setRange(20, 2000)
        self.dist.setValue(200)
        self.dist.setSuffix(' m')
        self.run_btn = button('Evaluate', primary=True)
        self.run_btn.clicked.connect(self.evaluate)
        self.prog = QtWidgets.QProgressBar()
        for w in (label('track'), self.track, label('bench'), self.bench, label('GT driver'), self.speed,
                  label('distance'), self.dist, self.run_btn, self.prog):
            cl.addWidget(w)
        rl.addWidget(ctl)

        self.table = QtWidgets.QTableWidget()
        self.table.setMaximumHeight(170)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.hints = [EmptyHint(self.table, 'Press Evaluate: precision, recall, range error and colour accuracy '
                                            'per sensor and for the fused output.')]
        rl.addWidget(self.table)

        self.recall_plot = pg.PlotWidget(title='recall vs range')
        self.recall_plot.setYRange(0, 1.05)
        self.recall_plot.showGrid(y=True, alpha=0.2)
        self.recall_plot.addLegend(offset=(-5, 5))
        fbox = QtWidgets.QWidget()
        fl = QtWidgets.QVBoxLayout(fbox)
        fl.setContentsMargins(0, 0, 0, 0)
        self.frame_plot = pg.PlotWidget(title='frame: truth (rings) vs detections')
        self.frame_plot.setAspectLocked(True)
        self.frame_plot.showGrid(x=True, y=True, alpha=0.15)
        fl.addWidget(self.frame_plot, 1)
        self.hints += [EmptyHint(self.recall_plot, 'Recall by distance band, one bar per sensor.'),
                       EmptyHint(self.frame_plot, 'Scrub through evaluated frames:\ntrue cones as rings, detections as dots.')]
        self.prog.setMaximumWidth(160)
        self.prog.hide()
        cl.setStretchFactor(self.bench, 1)
        self.prog.setTextVisible(False)
        srow = QtWidgets.QHBoxLayout()
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self._show_frame)
        self.frame_label = label('', 'muted')
        self.frame_src = QtWidgets.QComboBox()
        self.frame_src.currentIndexChanged.connect(lambda _: self._show_frame(self.frame_slider.value()))
        self.frame_slider.setEnabled(False)
        self.frame_src.setEnabled(False)
        srow.addWidget(self.frame_slider, 1)
        srow.addWidget(self.frame_src)
        srow.addWidget(self.frame_label)
        fl.addLayout(srow)
        fl.addWidget(label(color_legend(), 'muted'))
        rl.addWidget(Splitter([(self.recall_plot, 260, 0.45), (fbox, 300, 0.55)]), 1)
        self.notes = label('', 'muted', True)
        rl.addWidget(self.notes)
        outer.addWidget(Splitter([(left, 340, 0.3), (right, 560, 0.7)]), 1)

        state.profile_loaded.connect(self.rebuild)
        state.profile_edited.connect(lambda what: self.rebuild() if what == 'rig' else None)
        self.rebuild()

    # ---- algorithm pickers --------------------------------------------------
    def rebuild(self):
        while self.algo_lay.count():
            w = self.algo_lay.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        prof = self.state.profile
        any_sensor = False
        for s in prof.rig.perception():
            if s.type == 'timeless_classic':
                self.algo_lay.addWidget(label('<b>{}</b>: timeless classic detector - a ground-truth + noise model, '
                                              'no algorithm to choose (edit its noise on the Rig page).'.format(s.name),
                                              None, True))
                continue
            kind = 'lidar' if s.type == 'lidar' else 'camera'
            picker = AlgoPicker(kind, '{} - {} ({})'.format(s.name, registry.STAGE_LABELS[kind], s.type))
            algo = s.algorithm or {'name': list(registry.entries(kind))[0], 'params': {}}
            picker.set(algo['name'], algo.get('params', {}))
            picker.algo_changed.connect(lambda n, sn=s.name: self.state.set_sensor_algo(sn, n))
            picker.param_changed.connect(lambda k, v, sn=s.name: self.state.set_sensor_algo_param(sn, k, v))
            self.algo_lay.addWidget(picker)
            any_sensor = True
        if not prof.rig.perception():
            self.algo_lay.addWidget(label('This rig has no perception sensors - add some on the Rig page.', 'warn'))
        fusion = AlgoPicker('fusion')
        st = prof.stages['fusion']
        fusion.set(st['name'], st.get('params', {}))
        fusion.algo_changed.connect(lambda n: self.state.set_stage('fusion', n))
        fusion.param_changed.connect(lambda k, v: self.state.set_stage_param('fusion', k, v))
        self.algo_lay.addWidget(fusion)
        self.algo_lay.addWidget(label('On the timeless bench every sensor uses its detection-level model; the '
                                      'algorithm parameters still matter (e.g. min_cluster_points sets how many LiDAR '
                                      'returns a cone needs, min_box_px how big a camera box must be).', 'muted', True))
        self.algo_lay.addStretch(1)
        del any_sensor

    # ---- evaluation --------------------------------------------------------------
    def evaluate(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.abort = True
            return
        from ...bench.perception_eval import evaluate
        prof = self.state.profile.copy()
        track, bench = self.track.currentText(), self.bench.currentData()
        speed, dist = self.speed.value(), self.dist.value()

        def job(w):
            return evaluate(prof, track, bench=bench, speed=speed, distance=dist,
                            progress=lambda f: w.progress.emit(int(f * 100), 100, ''),
                            stop_flag=lambda: w.abort)
        self.worker = Worker(job, self)
        self.worker.progress.connect(lambda i, n, t: self.prog.setValue(i))
        self.worker.done.connect(self._done)
        self.worker.error.connect(self._error)
        self.run_btn.setText('Stop')
        self.prog.setValue(0)
        self.prog.show()
        self.worker.start()

    def _error(self, tb):
        self.run_btn.setText('Evaluate')
        self.prog.hide()
        QtWidgets.QMessageBox.warning(self, 'Evaluation failed', tb[-2000:])

    def _done(self, out):
        self.run_btn.setText('Evaluate')
        self.prog.setValue(100)
        report, frames, _ = out
        for h in self.hints:
            h.hide()
        self.prog.hide()
        self.frame_slider.setEnabled(True)
        self.frame_src.setEnabled(True)
        self.frames = frames
        sens = report['sensors']
        names = [n for n in sens if n != 'fused'] + (['fused'] if 'fused' in sens else [])
        cols = ['fidelity', 'precision', 'recall', 'RMSE m', 'bias x/y m', 'colour acc.', 'coloured', 'ms mean/p95']
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(names))
        self.table.setVerticalHeaderLabels(names)
        for i, n in enumerate(names):
            s = sens[n]
            t = report['timings_ms'].get(n if n != 'fused' else 'fusion', {})
            vals = [report['fidelity'].get(n, '-'), '{:.3f}'.format(s['precision']), '{:.3f}'.format(s['recall']),
                    '{:.3f}'.format(s['rmse_m']), '{:+.2f} / {:+.2f}'.format(s['bias_x_m'], s['bias_y_m']),
                    '-' if s['color_accuracy'] is None else '{:.3f}'.format(s['color_accuracy']),
                    '{:.0%}'.format(s['colored_fraction']),
                    '{} / {}'.format(t.get('mean', '-'), t.get('p95', '-'))]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(str(v)))
        self.table.resizeColumnsToContents()
        # recall vs range bars
        self.recall_plot.clear()
        labels = range_labels()
        nb = len(names)
        width = 0.8 / max(1, nb)
        for k, n in enumerate(names):
            r = [0 if v is None else v for v in sens[n]['recall_by_range']]
            x = np.arange(len(labels)) + (k - (nb - 1) / 2) * width
            bg = pg.BarGraphItem(x=x, height=r, width=width * 0.95, brush=pg.mkBrush(*PALETTE[k % len(PALETTE)]))
            # legend entry comes from the empty curve below (bars only register one on newer pyqtgraph)
            self.recall_plot.addItem(bg)
            self.recall_plot.plot([], [], pen=pg.mkPen(*PALETTE[k % len(PALETTE)], width=6), name=n)
        self.recall_plot.getAxis('bottom').setTicks([list(enumerate(['{} m'.format(l) for l in labels]))])
        # frames
        self.frame_src.blockSignals(True)
        self.frame_src.clear()
        self.frame_src.addItems(['fused'] + [n for n in names if n != 'fused'])
        self.frame_src.blockSignals(False)
        self.frame_slider.setRange(0, max(0, len(frames) - 1))
        self._show_frame(0)
        self.notes.setText('{} on {} ({}), {:.0f} m, {} frames. Recall counts true cones within 30 m in front of '
                           'the rear axle; precision judges detections in the same region.'.format(
                               report['bench'], report['track'], ', '.join('{}={}'.format(k, v) for k, v in
                                                                         report['fidelity'].items()),
                               report['distance'], report['frames']))

    def _show_frame(self, i):
        self.frame_plot.clear()
        if not self.frames or i >= len(self.frames):
            return
        f = self.frames[i]
        src = self.frame_src.currentText() or 'fused'
        det = f['fused'] if src == 'fused' else f['by_sensor'].get(src, np.zeros((0, 4)))
        gt = f['gt']
        near = (np.hypot(gt[:, 0], gt[:, 1]) < 40) & (gt[:, 0] > -10)
        gt = gt[near]
        self.frame_plot.addItem(pg.ScatterPlotItem(
            pos=gt[:, :2], size=14, brush=None,
            pen=[pg.mkPen(*COLOR_RGB.get(int(c), (200, 200, 200)), width=1.5) for c in gt[:, 2]]))
        if len(det):
            self.frame_plot.addItem(pg.ScatterPlotItem(
                pos=det[:, :2], size=8, pen=None,
                brush=[pg.mkBrush(*COLOR_RGB.get(int(c), (230, 230, 230))) for c in det[:, 2]]))
        self.frame_plot.plot([-1.3, 1.3, 1.3, -1.3, -1.3], [-0.65, -0.65, 0.65, 0.65, -0.65], pen=pg.mkPen('w'))
        self.frame_plot.setRange(xRange=(-8, 35), yRange=(-18, 18))
        self.frame_label.setText('t = {:.1f} s, {} detections'.format(f['t'], len(det)))
