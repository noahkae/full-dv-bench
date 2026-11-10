"""Results page: every saved run, filterable, with overlays and a side-by-side comparison."""
from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from ...bench.records import delete_run, list_runs, load_log
from ...tracks import load_track
from .. import theme
from ..widgets import EmptyHint, Splitter, button, label

COLS = ['time', 'bench', 'track', 'profile', 'seed', 'result', 'total s', 'DOO', 'OC', 'max CTE', 'v max', 'p95 ms']
PALETTE = [(214, 0, 28), (255, 205, 0), (80, 170, 255), (90, 220, 140), (220, 120, 255), (255, 150, 60),
           (255, 255, 255), (0, 200, 200)]


class ResultsPage(QtWidgets.QWidget):
    def __init__(self, state, parent=None):
        super(ResultsPage, self).__init__(parent)
        self.state = state
        self.records = []
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.addWidget(label('Results', 'title'))
        outer.addWidget(label('Select runs to overlay them. Comparing the same profile on timeless vs 3D shows '
                              'the sim-to-sim gap - the same thing you will check again on Assetto Corsa.',
                              'subtitle', True))
        bar = QtWidgets.QHBoxLayout()
        self.f_profile = QtWidgets.QComboBox()
        self.f_track = QtWidgets.QComboBox()
        self.f_bench = QtWidgets.QComboBox()
        for c in (self.f_profile, self.f_track, self.f_bench):
            c.currentIndexChanged.connect(self._fill)
        b_ref = button('Refresh')
        b_ref.clicked.connect(self.refresh)
        b_del = button('Delete selected')
        b_del.clicked.connect(self._delete)
        b_dir = button('Open folder')
        b_dir.clicked.connect(self._open_dir)
        for w in (label('profile'), self.f_profile, label('track'), self.f_track, label('bench'), self.f_bench):
            bar.addWidget(w)
        bar.addStretch(1)
        for w in (b_ref, b_dir, b_del):
            bar.addWidget(w)
        outer.addLayout(bar)
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table_hint = EmptyHint(self.table, 'No saved runs match. Run a bench with "save run" ticked.')
        self.table.itemSelectionChanged.connect(self._compare)
        self.traj = pg.PlotWidget(title='trajectories')
        self.traj.setAspectLocked(True)
        self.traj.showGrid(x=True, y=True, alpha=0.15)
        self.traj.addLegend(offset=(5, 5))
        self.vplot = pg.PlotWidget(title='speed vs distance')
        self.vplot.setLabel('bottom', 'distance', 'm')
        self.vplot.setLabel('left', 'speed', 'm/s')
        self.vplot.showGrid(x=True, y=True, alpha=0.15)
        self.cmp = QtWidgets.QTextBrowser()
        self.traj_hint = EmptyHint(self.traj, 'Select runs in the table to overlay their trajectories.\nCtrl / Shift-click selects several.')
        self.v_hint = EmptyHint(self.vplot, 'Speed traces of the selected runs.')
        self.count = label('', 'muted')
        bar.insertWidget(bar.count() - 3, self.count)
        bar.insertSpacing(bar.count() - 3, 10)
        lower = Splitter([(self.traj, 300, 0.36), (self.vplot, 300, 0.36), (self.cmp, 260, 0.28)])
        outer.addWidget(Splitter([(self.table, 150, 0.38), (lower, 260, 0.62)], QtCore.Qt.Vertical), 1)
        self.refresh()

    def refresh(self):
        self.records = list_runs()
        for combo, key in ((self.f_profile, 'profile_name'), (self.f_track, 'track'), (self.f_bench, 'bench')):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem('all')
            combo.addItems(sorted({r.get(key, '') for r in self.records}))
            i = combo.findText(cur)
            combo.setCurrentIndex(max(0, i))
            combo.blockSignals(False)
        self._fill()

    def _visible(self):
        out = []
        for r in self.records:
            ok = all(c.currentText() in ('all', '', r.get(k, ''))
                     for c, k in ((self.f_profile, 'profile_name'), (self.f_track, 'track'), (self.f_bench, 'bench')))
            if ok:
                out.append(r)
        return out

    def _fill(self, *_):
        rows = self._visible()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        self.table_hint.setVisible(not rows)
        self.count.setText('{} of {} runs'.format(len(rows), len(self.records)))
        for i, r in enumerate(rows):
            res = r.get('result', {})
            vals = [r.get('time', ''), r.get('bench', ''), r.get('track', ''), r.get('profile_name', ''),
                    r.get('seed', ''), 'OK' if res.get('complete') else (res.get('reason') or 'DNF'),
                    res.get('total'), res.get('doo'), res.get('off_course'), res.get('max_cte'),
                    res.get('max_speed'), res.get('compute_ms_p95')]
            for j, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem()
                if isinstance(v, (int, float)) and v is not None:
                    it.setData(QtCore.Qt.DisplayRole, v)
                else:
                    it.setText('' if v is None else str(v))
                if j == 5:
                    it.setForeground(QtGui.QColor(theme.GOOD if res.get('complete') else theme.BAD))
                it.setData(QtCore.Qt.UserRole, r['dir'])
                self.table.setItem(i, j, it)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self._compare()

    def _selected(self):
        dirs = []
        for idx in self.table.selectionModel().selectedRows():
            it = self.table.item(idx.row(), 0)
            if it is not None:
                dirs.append(it.data(QtCore.Qt.UserRole))
        by_dir = {r['dir']: r for r in self.records}
        return [by_dir[d] for d in dirs if d in by_dir]

    def _compare(self):
        sel = self._selected()[:8]
        self.traj.clear()
        self.vplot.clear()
        self.traj_hint.setVisible(not sel)
        self.v_hint.setVisible(not sel)
        if not sel:
            self.cmp.setHtml('<p style="color:{}">Side-by-side numbers for up to 8 selected runs appear here.</p>'
                             .format(theme.FAINT))
            return
        tracks = {r['track'] for r in sel}
        if len(tracks) == 1:
            try:
                t = load_track(sel[0]['track'], scenery=False)
                cols = [{'yellow': (255, 210, 0), 'blue': (40, 110, 255)}.get(c, (255, 130, 0)) for c in t.colors]
                self.traj.addItem(pg.ScatterPlotItem(pos=t.positions, size=5, pen=None,
                                                     brush=[pg.mkBrush(*c) for c in cols]))
            except Exception:
                pass
        rows = []
        for k, r in enumerate(sel):
            c = PALETTE[k % len(PALETTE)]
            name = '{} {} s{}'.format(r['profile_name'], r['bench'], r['seed'])
            try:
                a = load_log(r['dir'])
            except Exception:
                continue
            if len(a.get('t', [])):
                self.traj.plot(a['x'], a['y'], pen=pg.mkPen(c, width=2), name=name)
                s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(a['x']), np.diff(a['y'])))])
                self.vplot.plot(s, a['speed'], pen=pg.mkPen(c, width=1.5))
            res = r.get('result', {})
            rows.append((c, name, res))
        keys = [('event', 'event'), ('result', 'summary'), ('laps', 'laps'), ('DOO', 'doo'), ('OC', 'off_course'),
                ('max CTE', 'max_cte'), ('RMS CTE', 'rms_cte'), ('max speed', 'max_speed'),
                ('mean speed', 'mean_speed'), ('peak lat g', 'max_lat_g'), ('steer rate RMS', 'steer_rate_rms'),
                ('compute p50', 'compute_ms_p50'), ('compute p95', 'compute_ms_p95'), ('uses GT', 'uses_gt'),
                ('speed-up', 'speedup')]
        head = ''.join('<th style="color:rgb{}">{}</th>'.format(c, n) for c, n, _ in rows)
        body = ''
        for lab, k in keys:
            body += '<tr><td style="color:{}">{}</td>{}</tr>'.format(
                theme.MUTED, lab, ''.join('<td>{}</td>'.format(res.get(k, '')) for _, _, res in rows))
        self.cmp.setHtml('<table cellpadding="3"><tr><th></th>{}</tr>{}</table>'.format(head, body))

    def _delete(self):
        sel = self._selected()
        if not sel:
            return
        r = QtWidgets.QMessageBox.question(self, 'Delete runs', 'Delete {} run(s) from disk?'.format(len(sel)))
        if r == QtWidgets.QMessageBox.Yes:
            for rec in sel:
                delete_run(rec['dir'])
            self.refresh()

    def _open_dir(self):
        from ... import paths
        sel = self._selected()
        d = sel[0]['dir'] if sel else paths.RUNS
        os.makedirs(d, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(d))
