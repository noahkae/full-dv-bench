"""
Bench pages (timeless 2D and fsds_lite 3D share this one class).

    Run      live view, run controls (speed, pause, single step), result card
    Telemetry  speed vs target, steering, cross-track error, compute time
    Batch & gate   tracks x seeds headless, and the promotion gate
"""
from __future__ import annotations

import json

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ...tracks import list_tracks, load_track, track_reference
from .. import theme
from ..state import RunController, Worker
from ..widgets import EmptyHint, Splitter, Telemetry, TrackView, button, card, label, result_html

SPEEDS = [('as fast as possible', 0.0), ('real time x1', 1.0), ('x0.5 (slow motion)', 0.5), ('x2', 2.0),
          ('x5', 5.0)]


class BenchPage(QtWidgets.QWidget):
    run_saved = QtCore.pyqtSignal()

    def __init__(self, state, bench, parent=None):
        super(BenchPage, self).__init__(parent)
        self.state = state
        self.bench = bench
        self.ctl = RunController(state, self)
        self.ctl.started.connect(self._started)
        self.ctl.frame.connect(self._frame)
        self.ctl.finished.connect(self._finished)
        self.ctl.run_event.connect(self._event)
        self.ctl.failed.connect(self._failed)
        self.panel = None
        self.backend = None
        self.batch_worker = None
        self._last_tele = 0.0

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        if bench == 'timeless':
            outer.addWidget(label('Timeless bench (2D)', 'title'))
            outer.addWidget(label('Deterministic, faster than real time. Detection-level sensors from your rig, '
                                  'the same vehicle file as 3D, FS-rules scoring. Iterate here, then pass the gate.',
                                  'subtitle', True))
        else:
            outer.addWidget(label('3D bench (fsds_lite)', 'title'))
            outer.addWidget(label('Ray-cast LiDAR point clouds and rendered camera images through your real '
                                  'detectors, 4-wheel tyre dynamics. Set a sensor to "detection" to isolate '
                                  'dynamics from perception (promotion rung 2), "auto"/"raw" for the full thing.',
                                  'subtitle', True))

        bar = card()
        bl = QtWidgets.QHBoxLayout(bar)
        self.track = QtWidgets.QComboBox()
        self.track.addItems(list_tracks())
        self.track.setCurrentText('autocross')
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 9999)
        self.speed = QtWidgets.QComboBox()
        for name, v in SPEEDS:
            self.speed.addItem(name, v)
        if bench == 'lite':
            self.speed.setCurrentIndex(1)
        self.speed.currentIndexChanged.connect(lambda _: self.ctl.set_speed(self.speed.currentData()))
        self.b_run = button('Run', primary=True, tip='Start an episode with the open profile')
        self.b_pause = button('Pause')
        self.b_step = button('Step', tip='Advance one pipeline tick (pauses first)')
        self.b_stop = button('Stop')
        self.save = QtWidgets.QCheckBox('save run')
        self.save.setChecked(True)
        self.save.setToolTip('Write run.json + log.npz to runs/ so the run shows up under Results')
        self.follow = QtWidgets.QCheckBox('follow car')
        self.follow.setChecked(True)
        self.follow.toggled.connect(lambda v: setattr(self.view, 'follow', v))
        for w in (label('track'), self.track, label('seed'), self.seed, label('speed'), self.speed, self.b_run,
                  self.b_pause, self.b_step, self.b_stop, self.save, self.follow):
            bl.addWidget(w)
        bl.insertSpacing(bl.indexOf(self.b_run), 8)
        bl.insertSpacing(bl.indexOf(self.save), 8)
        bl.addStretch(1)
        self.hud = label('', 'hud')
        self.hud.setFont(theme.mono(10))
        bl.addWidget(self.hud)
        outer.addWidget(bar)
        self.b_run.clicked.connect(self.run)
        self.b_pause.clicked.connect(self._pause)
        self.b_step.clicked.connect(self._step)
        self.b_stop.clicked.connect(self.ctl.stop)

        if bench == 'lite':
            lite = card()
            ll = QtWidgets.QHBoxLayout(lite)
            self.phys = QtWidgets.QComboBox()
            for hz in (250, 500, 1000):
                self.phys.addItem('{} Hz physics'.format(hz), float(hz))
            self.phys.setCurrentIndex(1)
            self.cams = QtWidgets.QCheckBox('render cameras')
            self.cams.setChecked(True)
            self.fid_host = QtWidgets.QHBoxLayout()
            ll.addWidget(self.phys)
            ll.addWidget(self.cams)
            ll.addWidget(label('fidelity per sensor:', 'muted'))
            ll.addLayout(self.fid_host)
            ll.addStretch(1)
            self.gate_note = label('', 'warn')
            ll.addWidget(self.gate_note)
            outer.addWidget(lite)
            self.fid_boxes = {}

        self.view = TrackView()
        if bench == 'lite':
            self.left_tabs = QtWidgets.QTabWidget()
            self.sim_host = QtWidgets.QWidget()
            self.sim_lay = QtWidgets.QVBoxLayout(self.sim_host)
            self.sim_lay.setContentsMargins(0, 0, 0, 0)
            hint = label('Press Run to build the 3D world for the chosen track.', 'empty')
            hint.setAlignment(QtCore.Qt.AlignCenter)
            self.sim_lay.addWidget(hint)
            self.left_tabs.addTab(self.sim_host, 'fsds_lite view (3D / top-down)')
            self.left_tabs.addTab(self.view, 'Stack view (what the car believes)')
            from ..sensorview import SensorView
            self.sensor_view = SensorView()
            self.left_tabs.addTab(self.sensor_view, 'Sensor feeds')
            left_pane = self.left_tabs
        else:
            left_pane = self.view

        right = QtWidgets.QTabWidget()
        runw = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(runw)
        self.result = QtWidgets.QTextBrowser()
        rl.addWidget(self.result, 2)
        rl.addWidget(label('Events', 'muted'))
        self.events = QtWidgets.QListWidget()
        self.events.setFont(theme.mono(9))
        self.events.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.events.setWordWrap(True)
        rl.addWidget(self.events, 1)
        right.addTab(runw, 'Run')
        self.tele = Telemetry()
        right.addTab(self.tele, 'Telemetry')
        right.addTab(self._batch_tab(), 'Batch && gate')
        outer.addWidget(Splitter([(left_pane, 480, 0.64), (right, 360, 0.36)]), 1)
        self._buttons(False)
        self._idle_hint()
        self.track.currentTextChanged.connect(self._preview_track)
        self._preview_track(self.track.currentText())
        state.profile_loaded.connect(self._profile_changed)
        state.profile_edited.connect(lambda w: self._profile_changed() if w == 'rig' else None)
        self._profile_changed()

    # ---- batch / gate tab ------------------------------------------------------
    def _batch_tab(self):
        w = QtWidgets.QWidget()
        l = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        self.batch_tracks = QtWidgets.QListWidget()
        self.batch_tracks.setMaximumHeight(150)
        for t in list_tracks():
            it = QtWidgets.QListWidgetItem(t)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Checked if t in ('autocross', 'timeless_map1', 'timeless_map2',
                                                        'timeless_map3', 'acceleration') else QtCore.Qt.Unchecked)
            self.batch_tracks.addItem(it)
        top.addWidget(self.batch_tracks, 1)
        col = QtWidgets.QVBoxLayout()
        self.batch_seeds = QtWidgets.QSpinBox()
        self.batch_seeds.setRange(1, 50)
        self.batch_seeds.setValue(3)
        self.batch_seeds.setPrefix('seeds: ')
        self.b_batch = button('Run batch')
        self.b_gate = button('Run promotion gate', primary=True,
                             tip='every gate track on 3 seeds; must finish all, low penalties, no ground truth')
        self.batch_prog = QtWidgets.QProgressBar()
        col.addWidget(self.batch_seeds)
        col.addWidget(self.b_batch)
        col.addWidget(self.b_gate)
        col.addWidget(self.batch_prog)
        col.addStretch(1)
        top.addLayout(col)
        l.addLayout(top)
        self.batch_table = QtWidgets.QTableWidget(0, 7)
        self.batch_table.setHorizontalHeaderLabels(['track', 'seed', 'result', 'time', 'DOO', 'OC', 'max CTE'])
        self.batch_table.verticalHeader().setVisible(False)
        self.batch_table.setAlternatingRowColors(True)
        self.batch_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        self.batch_hint = EmptyHint(self.batch_table, 'Tick tracks, then Run batch - or run the promotion gate.')
        l.addWidget(self.batch_table, 1)
        self.batch_summary = label('', 'muted', True)
        l.addWidget(self.batch_summary)
        self.b_batch.clicked.connect(self._run_batch)
        self.b_gate.clicked.connect(self._run_gate)
        return w

    def _checked_tracks(self):
        return [self.batch_tracks.item(i).text() for i in range(self.batch_tracks.count())
                if self.batch_tracks.item(i).checkState() == QtCore.Qt.Checked]

    def _run_batch(self):
        if self.batch_worker is not None and self.batch_worker.isRunning():
            self.batch_worker.abort = True
            return
        from ...bench.session import run_episode
        prof = self.state.profile.copy()
        tracks, seeds = self._checked_tracks(), list(range(self.batch_seeds.value()))
        bench, save = self.bench, self.save.isChecked()
        kw = self._backend_kw()

        def job(w):
            rows = []
            n = len(tracks) * len(seeds)
            for tr in tracks:
                for s in seeds:
                    if w.abort:
                        return rows
                    res, _ = run_episode(prof, tr, bench=bench, seed=s, save=save, backend_kw=kw)
                    row = {'track': tr, 'seed': s, 'complete': res['complete'], 'summary': res['summary'],
                           'total': res['total'], 'doo': res['doo'], 'oc': res['off_course'],
                           'max_cte': res['max_cte']}
                    rows.append(row)
                    w.progress.emit(len(rows), n, json.dumps(row))
            return rows
        self.batch_table.setRowCount(0)
        self.batch_hint.hide()
        self.batch_worker = Worker(job, self)
        self.batch_worker.progress.connect(self._batch_row)
        self.batch_worker.done.connect(self._batch_done)
        self.batch_worker.error.connect(self._failed)
        self.b_batch.setText('Stop batch')
        self.batch_worker.start()

    def _batch_row(self, i, n, text):
        self.batch_prog.setMaximum(n)
        self.batch_prog.setValue(i)
        if not text:
            return
        r = json.loads(text)
        row = self.batch_table.rowCount()
        self.batch_table.insertRow(row)
        vals = [r['track'], r['seed'], 'OK' if r['complete'] else 'DNF', r['total'], r['doo'], r['oc'], r['max_cte']]
        for j, v in enumerate(vals):
            it = QtWidgets.QTableWidgetItem('' if v is None else str(v))
            if j == 2:
                it.setForeground(QtGui.QColor(theme.GOOD if r['complete'] else theme.BAD))
            self.batch_table.setItem(row, j, it)

    def _batch_done(self, rows):
        self.b_batch.setText('Run batch')
        if isinstance(rows, list) and rows:
            done = [r for r in rows if r['complete']]
            self.batch_summary.setText('{}/{} finished, mean DOO {:.2f}, mean OC {:.2f}, mean time {}'.format(
                len(done), len(rows), np.mean([r['doo'] for r in rows]), np.mean([r['oc'] for r in rows]),
                '{:.2f} s'.format(np.mean([r['total'] for r in done])) if done else '-'))
        self.run_saved.emit()

    def _run_gate(self):
        if self.batch_worker is not None and self.batch_worker.isRunning():
            return
        from ...bench.gate import run_gate
        prof = self.state.profile.copy()
        bench = self.bench

        def job(w):
            def prog(i, n, text):
                w.progress.emit(i, n, '')
            return run_gate(prof, bench=bench, progress=prog, stop_flag=lambda: w.abort)
        self.batch_worker = Worker(job, self)
        self.batch_worker.progress.connect(self._batch_row)
        self.batch_worker.done.connect(self._gate_done)
        self.batch_worker.error.connect(self._failed)
        self.batch_table.setRowCount(0)
        self.batch_hint.hide()
        self.batch_summary.setText('running the promotion gate...')
        self.batch_worker.start()

    def _gate_done(self, out):
        passed, report = out
        for r in report.get('rows', []):
            self._batch_row(0, 0, json.dumps({'track': r['track'], 'seed': r['seed'], 'complete': r['complete'],
                                              'total': r['total'], 'doo': r['doo'], 'oc': r['oc'],
                                              'max_cte': '', 'summary': r['summary']}))
        if report.get('aborted'):
            self.batch_summary.setText('gate aborted')
            return
        from ...bench.gate import record_gate
        record_gate(self.state.profile, report)
        self.state.meta_edited()
        color = theme.GOOD if passed else theme.BAD
        text = '<b style="color:{}">{} gate {}</b> - mean DOO {}, mean OC {}, mean time {} s. {}'.format(
            color, self.bench, 'PASSED' if passed else 'NOT passed', report['mean_doo'], report['mean_oc'],
            report['mean_total'], '; '.join(report['reasons']))
        self.batch_summary.setText(text + ('  (saved in the profile - press Save)' if passed else ''))
        self._profile_changed()

    # ---- run ---------------------------------------------------------------------
    def _backend_kw(self):
        if self.bench != 'lite':
            return None
        fid = {n: b.currentText() for n, b in self.fid_boxes.items()}
        return {'physics_hz': self.phys.currentData(), 'cameras': self.cams.isChecked(),
                'fidelity_override': {n: v for n, v in fid.items() if v != 'auto'}}

    def _profile_changed(self):
        if not self.ctl.running and self.ctl.runner is None:
            self._idle_hint()
        if self.bench != 'lite':
            return
        while self.fid_host.count():
            w = self.fid_host.takeAt(0).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self.fid_boxes = {}
        for s in self.state.profile.rig.perception():
            if s.type == 'timeless_classic':
                continue
            cb = QtWidgets.QComboBox()
            cb.addItems(['auto', 'detection'])
            cb.setToolTip('auto = raw data through the real detector; detection = synthetic model')
            self.fid_host.addWidget(label(s.name))
            self.fid_host.addWidget(cb)
            self.fid_boxes[s.name] = cb
        g = (getattr(self.state.profile, 'gates', None) or {}).get('timeless')
        if g and g.get('passed'):
            self.gate_note.setText('')
        else:
            self.gate_note.setText('timeless gate not passed yet - you can still run, but expect surprises')

    def _idle_hint(self):
        self.result.setHtml(
            '<p style="color:{m}">Press <b>Run</b> to drive <b>{p}</b> around the chosen track.</p>'
            '<p style="color:{f}">The result card (laps, cones hit, off-course, tracking error, compute time) '
            'appears here when the episode ends. Live plots are under Telemetry.</p>'.format(
                m=theme.MUTED, f=theme.FAINT, p=self.state.profile.name))

    def _preview_track(self, name):
        """Show the chosen track before a run, instead of an empty plot."""
        if self.ctl.running or not name:
            return
        try:
            t = load_track(name, scenery=False)
            self.view.set_track(t, track_reference(t))
        except Exception as e:
            self.state.message.emit('could not preview {}: {}'.format(name, e))

    def run(self):
        if self.ctl.running:
            return
        if self.state.profile.validate():
            r = QtWidgets.QMessageBox.question(self, 'Profile has problems',
                                               '\n'.join(self.state.profile.validate()) + '\n\nRun anyway?')
            if r != QtWidgets.QMessageBox.Yes:
                return
        self.events.clear()
        self.result.setHtml('<i>building {} ...</i>'.format(self.track.currentText()))
        QtWidgets.QApplication.processEvents()
        self._teardown()
        self.ctl.start(self.track.currentText(), self.bench, self.seed.value(), self.speed.currentData(),
                       backend_kw=self._backend_kw(), save=self.save.isChecked())

    def _pause(self):
        self.ctl.pause()
        self.b_pause.setText('Resume' if self.b_pause.text() == 'Pause' else 'Pause')

    def _step(self):
        if self.b_pause.text() == 'Pause':
            self.ctl.pause(True)
            self.b_pause.setText('Resume')
        self.ctl.step()

    def _buttons(self, running):
        self.b_run.setEnabled(not running)
        for b in (self.b_pause, self.b_step, self.b_stop):
            b.setEnabled(running)
        self.track.setEnabled(not running)
        self.b_pause.setText('Pause')

    def _started(self, runner):
        self.backend = runner.backend
        ref = track_reference(runner.track)
        self.view.set_track(runner.track, ref, runner.stack.vehicle)
        self.view.follow = self.follow.isChecked()
        self._buttons(True)
        self.result.setHtml('<i>running {} on {} (seed {})...</i>'.format(
            runner.stack.profile.name, runner.cfg.track, runner.cfg.seed))
        if self.bench == 'lite':
            self.sensor_view.reset()
            self._build_panel(runner.backend)
            st = runner.backend.status()
            fid = st.extras.get('fidelity', {})
            note = ', '.join('{}: {}'.format(k, v) for k, v in fid.items())
            if st.extras.get('camera_error'):
                note += ' | cameras unavailable: ' + st.extras['camera_error']
            self.events.addItem('sensors: ' + note)

    def _build_panel(self, backend):
        try:
            from fsds_lite.viewer import SimPanel
            self.panel = SimPanel(backend.sim, backend.suite, use_3d=True, embedded=True, fps_3d=30)
            while self.sim_lay.count():
                w = self.sim_lay.takeAt(0).widget()
                if w:
                    w.setParent(None)
            self.sim_lay.addWidget(self.panel)
        except Exception as e:
            self.events.addItem('3D view unavailable: {}'.format(e))

    def _teardown(self):
        """Release the previous 3D world: SimPanel timers/filters, camera GL context."""
        if self.panel is not None:
            p = self.panel
            try:
                QtWidgets.QApplication.instance().removeEventFilter(p)
                p.keys.timer.stop()
                p.hud_timer.stop()
                if p.view3d is not None:
                    p.view3d.timer.stop()
                p.view2d.timer.stop()
            except Exception:
                pass
            p.setParent(None)
            p.deleteLater()
            self.panel = None
        if self.backend is not None and self.bench != 'timeless':
            try:
                self.backend.close()
            except Exception:
                pass
        self.backend = None

    def _frame(self, f):
        b = self.backend
        hit = None
        if b is not None:
            hit = getattr(b, 'cones_hit', None)
            if hit is None and getattr(b, 'sim', None) is not None:
                hit = b.sim.cones_hit
        self.view.update_frame(f, hit_mask=hit)
        if self.bench == 'lite' and self.left_tabs.currentWidget() is self.sensor_view:
            self.sensor_view.update_frame(f)
        st = f.status
        lap = '{}/{}'.format(min(len(st.laps) + 1, st.laps_required), st.laps_required)
        self.hud.setText('t {:6.1f} s   {:5.1f} m/s   lap {}   DOO {}   OC {}'.format(
            f.t, f.car['speed'], lap, st.doo, st.off_course))
        if f.t - self._last_tele > 0.5 or f.t < self._last_tele:
            self._last_tele = f.t
            r = self.ctl.runner
            if r is not None:
                self.tele.set_log(r.log.arrays())

    def _finished(self, res):
        self._buttons(False)
        r = self.ctl.runner
        if r is not None:
            self.tele.set_log(r.log.arrays())
        extra = ''
        if res.get('path'):
            extra = '<p style="color:{}">saved to {}</p>'.format(theme.MUTED, res['path'])
        self.result.setHtml(result_html(res, extra))
        self.run_saved.emit()

    def _event(self, kind, text):
        self.events.addItem('{}: {}'.format(kind, text))
        self.events.scrollToBottom()

    def _failed(self, tb):
        self._buttons(False)
        self.b_batch.setText('Run batch')
        self.result.setHtml('<pre style="color:{}">{}</pre>'.format(theme.BAD, tb[-3000:].replace('<', '&lt;')))

    def shutdown(self):
        self.ctl.stop()
        if self.batch_worker is not None:
            self.batch_worker.abort = True
        self._teardown()
