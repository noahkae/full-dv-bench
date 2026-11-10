"""Home: the workflow at a glance, the profile library, validation and recent runs."""
from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from ...bench.records import list_runs
from ...core import registry
from ...core.profile import list_profiles, load_profile
from .. import theme
from ..widgets import EmptyHint, button, card, label

STEPS = [
    ('rig', 'Sensor rig', 'Choose sensors: types, counts, mounting. See coverage instantly.'),
    ('perception', 'Perception', 'Detection algorithm per sensor + fusion. Measure precision / recall.'),
    ('planning', 'Planning & control', 'Odometry, mapping, planner, speed profile, steering, pedals.'),
    ('timeless', 'Timeless bench', 'Fast 2D runs, batches and the promotion gate.'),
    ('lite', '3D bench', 'fsds_lite: ray-cast LiDAR, rendered cameras, full dynamics.'),
    ('results', 'Results', 'Every run, compared side by side.'),
    ('deploy', 'Deploy / AC', 'ROS 2 parameters for the car; Assetto Corsa (planned).'),
]


class StepCard(QtWidgets.QFrame):
    """One clickable workflow step: number, title, what it is for, and where the open profile stands."""
    clicked = QtCore.pyqtSignal()

    def __init__(self, number, title, text, parent=None):
        super(StepCard, self).__init__(parent)
        self.setObjectName('step')
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setToolTip(text)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(8)
        badge = label(str(number), 'badge')
        badge.setFixedSize(22, 22)
        badge.setAlignment(QtCore.Qt.AlignCenter)
        head.addWidget(badge)
        t = label(title, 'h2')
        t.setTextFormat(QtCore.Qt.PlainText)
        t.setStyleSheet('font-size: 10pt;')
        head.addWidget(t, 1)
        lay.addLayout(head)
        d = label(text, 'faint', wrap=True)
        d.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        lay.addWidget(d, 1)
        self.status = label('', 'faint')
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        lay.addWidget(self.status)

    def set_status(self, text, color=None):
        self.status.setText(text)
        self.status.setToolTip(text)
        self.status.setStyleSheet('color: {}; font-weight: bold;'.format(color) if color else '')

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton and self.rect().contains(ev.pos()):
            self.clicked.emit()
        super(StepCard, self).mouseReleaseEvent(ev)


class HomePage(QtWidgets.QWidget):
    navigate = QtCore.pyqtSignal(str)

    def __init__(self, state, parent=None):
        super(HomePage, self).__init__(parent)
        self.state = state
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.addWidget(label('UCalgary Racing - Driverless Stack', 'title'))
        outer.addWidget(label(u'One stack profile \u2192 timeless 2D \u2192 fsds_lite 3D \u2192 (Assetto Corsa). '
                              'Pick a profile, work through the steps, promote when the gate passes.', 'subtitle'))

        # workflow strip
        strip = QtWidgets.QHBoxLayout()
        strip.setSpacing(8)
        self.steps = {}
        for i, (key, title, tip) in enumerate(STEPS):
            c = StepCard(i + 1, title, tip)
            c.setMinimumHeight(104)
            c.clicked.connect(lambda k=key: self.navigate.emit(k))
            self.steps[key] = c
            strip.addWidget(c, 1)
        outer.addLayout(strip)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(10)
        outer.addLayout(body, 1)

        # profiles
        left = card('Stack profiles')
        ll = QtWidgets.QVBoxLayout(left)
        self.list = QtWidgets.QListWidget()
        self.list.itemDoubleClicked.connect(lambda it: self._open(it.data(QtCore.Qt.UserRole)))
        self.list.currentItemChanged.connect(self._preview)
        ll.addWidget(self.list, 1)
        self.desc = label('', 'muted', wrap=True)
        self.desc.setMinimumHeight(52)
        self.desc.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        ll.addWidget(self.desc)
        row = QtWidgets.QHBoxLayout()
        self.b_open = button('Open', primary=True, tip='Open the selected profile (or double-click it)')
        self.b_new = button('Duplicate as...')
        self.b_del = button('Delete')
        for b in (self.b_open, self.b_new, self.b_del):
            row.addWidget(b)
        ll.addLayout(row)
        self.b_open.clicked.connect(lambda: self._open(self._selected()))
        self.b_new.clicked.connect(self._duplicate)
        self.b_del.clicked.connect(self._delete)
        body.addWidget(left, 2)

        # current profile summary
        mid = card('Open profile')
        ml = QtWidgets.QVBoxLayout(mid)
        self.summary = QtWidgets.QTextBrowser()
        self.summary.setOpenExternalLinks(False)
        ml.addWidget(self.summary)
        body.addWidget(mid, 3)

        # recent runs
        right = card('Recent runs')
        rl = QtWidgets.QVBoxLayout(right)
        self.runs = QtWidgets.QTreeWidget()
        self.runs.setHeaderLabels(['time', 'bench', 'track', 'profile', 'result'])
        self.runs.setRootIsDecorated(False)
        self.runs.setUniformRowHeights(True)
        self.runs.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.runs.setFocusPolicy(QtCore.Qt.NoFocus)
        self.runs.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.runs.header().setStretchLastSection(True)
        self.runs.itemDoubleClicked.connect(lambda *_: self.navigate.emit('results'))
        self.runs_hint = EmptyHint(self.runs, 'No runs yet.\nRun the timeless bench and they show up here.')
        rl.addWidget(self.runs)
        b = button('All results')
        b.clicked.connect(lambda: self.navigate.emit('results'))
        rl.addWidget(b)
        body.addWidget(right, 3)

        state.profile_loaded.connect(self.refresh)
        state.profile_edited.connect(lambda *_: self.refresh_summary())
        self.refresh()

    def _selected(self):
        it = self.list.currentItem()
        return it.data(QtCore.Qt.UserRole) if it else None

    def _open(self, name):
        if not name:
            return
        if self.state.dirty and not self._confirm_discard():
            return
        self.state.open(name)

    def _confirm_discard(self):
        r = QtWidgets.QMessageBox.question(self, 'Unsaved changes',
                                           'The open profile has unsaved changes. Discard them?')
        return r == QtWidgets.QMessageBox.Yes

    def _duplicate(self):
        base = self._selected() or self.state.profile.name
        name, ok = QtWidgets.QInputDialog.getText(self, 'Duplicate profile', 'New profile name:',
                                                  text=base + '_copy')
        if ok and name:
            name = ''.join(ch if ch.isalnum() or ch in '_-' else '_' for ch in name)
            self.state.new(name, load_profile(base) if base in list_profiles() else self.state.profile)
            self.refresh()

    def _delete(self):
        name = self._selected()
        if not name:
            return
        if name == self.state.profile.name:
            QtWidgets.QMessageBox.information(self, 'Delete', 'Open another profile before deleting this one.')
            return
        r = QtWidgets.QMessageBox.question(self, 'Delete profile', 'Delete profile "{}"?'.format(name))
        if r == QtWidgets.QMessageBox.Yes:
            self.state.delete(name)
            self.refresh()

    def _preview(self, it, _=None):
        if it is None:
            return
        try:
            p = load_profile(it.data(QtCore.Qt.UserRole))
            self.desc.setText(p.description or '(no description)')
        except Exception as e:
            self.desc.setText(str(e))

    def refresh(self):
        self.list.clear()
        for n in list_profiles():
            is_open = n == self.state.profile.name
            it = QtWidgets.QListWidgetItem(n + (u'   \u2022 open' if is_open else ''))
            it.setData(QtCore.Qt.UserRole, n)
            if is_open:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            self.list.addItem(it)
            if is_open:
                self.list.setCurrentItem(it)
        self.runs.clear()
        self._n_runs = 0
        recs = list_runs()
        self._n_runs = len(recs)
        for rec in recs[:25]:
            res = rec.get('result', {})
            it = QtWidgets.QTreeWidgetItem([rec.get('time', '')[5:16], rec.get('bench', ''), rec.get('track', ''),
                                            rec.get('profile_name', ''), res.get('summary', '')])
            it.setForeground(4, QtGui.QColor(theme.GOOD if res.get('complete') else theme.BAD))
            for c in (0, 1):
                it.setForeground(c, QtGui.QColor(theme.MUTED))
            it.setToolTip(4, res.get('summary', ''))
            self.runs.addTopLevelItem(it)
        for c in range(4):
            self.runs.resizeColumnToContents(c)
        self.runs_hint.setVisible(not recs)
        self.refresh_summary()

    def _refresh_steps(self):
        p = self.state.profile
        sensors = [s for s in p.rig.sensors if getattr(s, 'enabled', True)]
        n_perc = len(p.rig.perception())
        self.steps['rig'].set_status(u'{} sensor{} \u00b7 {} perception'.format(
            len(sensors), '' if len(sensors) == 1 else 's', n_perc), theme.MUTED)
        self.steps['perception'].set_status(registry.get('fusion', p.stages['fusion']['name']).name, theme.MUTED)
        self.steps['planning'].set_status(p.stages['lateral']['name'], theme.MUTED)
        gates = getattr(p, 'gates', None) or {}
        for key in ('timeless', 'lite'):
            g = gates.get(key)
            if g and g.get('passed'):
                self.steps[key].set_status('gate passed', theme.GOOD)
            elif g:
                self.steps[key].set_status('gate not passed', theme.BAD)
            else:
                self.steps[key].set_status('gate not run')
        n = getattr(self, '_n_runs', 0)
        self.steps['results'].set_status('{} saved run{}'.format(n, '' if n == 1 else 's'), theme.MUTED if n else None)
        self.steps['deploy'].set_status('ROS 2 export ready' if not p.validate() else 'fix profile first',
                                        theme.MUTED if not p.validate() else theme.WARN)

    def refresh_summary(self):
        self._refresh_steps()
        p = self.state.profile
        rows = []
        for s in p.rig.sensors:
            algo = s.algorithm['name'] if s.algorithm else ''
            rows.append('<li><b>{}</b> - {} {}<span style="color:{}"> {}</span></li>'.format(
                s.name, s.type, ('(' + s.preset + ')') if s.preset else '', theme.MUTED, algo))
        stages = []
        for kind in ('fusion', 'odometry', 'mapping', 'planner', 'speed', 'lateral', 'longitudinal', 'mission'):
            e = registry.get(kind, p.stages[kind]['name'])
            stages.append('<tr><td style="color:{};padding-right:10px">{}</td><td>{}</td></tr>'.format(
                theme.MUTED, registry.STAGE_LABELS[kind], e.label))
        problems = p.validate()
        prob = ''.join('<li style="color:{}">{}</li>'.format(theme.WARN, x) for x in problems) or \
            '<span style="color:{}">profile is valid</span>'.format(theme.GOOD)
        gates = getattr(p, 'gates', None) or {}
        gate = ''
        for bench, g in gates.items():
            gate += '<div>{} gate: <b style="color:{}">{}</b> ({})</div>'.format(
                bench, theme.GOOD if g.get('passed') else theme.BAD, 'PASSED' if g.get('passed') else 'not passed',
                g.get('time', ''))
        self.summary.setHtml(
            '<h2>{}</h2><p style="color:{}">{}</p><p>vehicle: <b>{}</b> &nbsp; pipeline: <b>{:g} Hz</b></p>'
            '<h3>Rig</h3><ul>{}</ul><h3>Pipeline</h3><table>{}</table><h3>Checks</h3>{}{}'.format(
                p.name, theme.MUTED, p.description, p.vehicle, p.pipeline_hz, ''.join(rows), ''.join(stages),
                prob if problems else prob, gate))
