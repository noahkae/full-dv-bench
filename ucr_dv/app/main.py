"""
The UCR driverless workbench: one window for the whole stack.

    python -m ucr_dv                   # or ./ucr-dv
    python -m ucr_dv --profile lidar_camera --page timeless
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

from .. import paths  # noqa: F401  (fsds_lite on sys.path)

PAGES = [('home', 'Home'), ('rig', 'Sensor rig'), ('perception', 'Perception'),
         ('planning', 'Planning & control'), ('timeless', 'Timeless bench'), ('lite', '3D bench'),
         ('results', 'Results'), ('research', 'Research'), ('deploy', 'Deploy / AC')]

# sidebar grouping: the workflow reads top to bottom (build -> test -> ship)
NAV = [(None, ['home']),
       ('BUILD', ['rig', 'perception', 'planning']),
       ('TEST', ['timeless', 'lite', 'results']),
       ('SHIP', ['deploy']),
       ('REFERENCE', ['research'])]


def build_window(state):
    from PyQt5 import QtCore, QtWidgets
    from .pages.bench import BenchPage
    from .pages.deploy import DeployPage
    from .pages.home import HomePage
    from .pages.perception import PerceptionPage
    from .pages.planning import PlanningPage
    from .pages.research import ResearchPage
    from .pages.results import ResultsPage
    from .pages.rig import RigPage
    from . import theme
    from .widgets import button, chip, label, set_chip

    class PageHost(QtWidgets.QWidget):
        """Gives a page exactly the viewport (or its minimum size, whichever is larger).

        A page with word-wrapped labels has a height-for-width layout, and
        QScrollArea would then stretch it to its *preferred* height - taller
        than the window - instead of letting it fill the viewport."""

        def __init__(self, page):
            super(PageHost, self).__init__()
            self.page = page
            page.setParent(self)

        def minimumSizeHint(self):
            return self.page.minimumSizeHint()

        def sizeHint(self):
            return self.page.minimumSizeHint()

        def resizeEvent(self, ev):
            self.page.setGeometry(self.rect())

    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super(MainWindow, self).__init__()
            self.state = state
            self.setWindowTitle('UCR Driverless Workbench')
            self.resize(1600, 980)
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            lay = QtWidgets.QHBoxLayout(central)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            side = QtWidgets.QWidget()
            side.setObjectName('sidebar')
            side.setAttribute(QtCore.Qt.WA_StyledBackground, True)
            side.setFixedWidth(216)
            sl = QtWidgets.QVBoxLayout(side)
            sl.setContentsMargins(0, 0, 0, 0)
            sl.setSpacing(0)
            brand = label('UCR <span style="color:{}">DV</span> stack'.format(theme.ACCENT_2), 'brand')
            brand.setTextFormat(QtCore.Qt.RichText)
            sl.addWidget(brand)
            sl.addWidget(label('Driverless workbench', 'brandsub'))
            titles = dict(PAGES)
            keys = [k for k, _ in PAGES]
            self.nav = QtWidgets.QButtonGroup(self)
            self.nav.setExclusive(True)
            self.nav_buttons = {}
            for section, members in NAV:
                if section:
                    sl.addWidget(label(section, 'section'))
                for key in members:
                    b = QtWidgets.QPushButton(titles[key].replace('&', '&&'))
                    b.setObjectName('nav')
                    b.setCheckable(True)
                    b.setCursor(QtCore.Qt.PointingHandCursor)
                    b.setFocusPolicy(QtCore.Qt.NoFocus)
                    b.setToolTip('{}  (Ctrl+{})'.format(titles[key], keys.index(key) + 1))
                    b.clicked.connect(lambda _=False, k=key: self.go(k))
                    self.nav.addButton(b)
                    self.nav_buttons[key] = b
                    sl.addWidget(b)
            sl.addStretch(1)
            # profile bar
            pb = QtWidgets.QWidget()
            pb.setObjectName('profilebar')
            pb.setAttribute(QtCore.Qt.WA_StyledBackground, True)
            pl = QtWidgets.QVBoxLayout(pb)
            pl.setContentsMargins(14, 12, 14, 14)
            pl.setSpacing(6)
            self.prof_label = label('', 'h2')
            self.prof_label.setWordWrap(True)
            head = QtWidgets.QHBoxLayout()
            head.addWidget(label('OPEN PROFILE', 'faint'))
            head.addStretch(1)
            self.prof_state = chip()
            head.addWidget(self.prof_state)
            pl.addLayout(head)
            pl.addWidget(self.prof_label)
            self.gate_chip = chip()
            gr = QtWidgets.QHBoxLayout()
            gr.addWidget(self.gate_chip)
            gr.addStretch(1)
            pl.addLayout(gr)
            row = QtWidgets.QHBoxLayout()
            self.b_save = button('Save', primary=True, tip='Save the open profile  (Ctrl+S)')
            self.b_save_as = button('Save as', tip='Save a copy under a new name and open it')
            row.addWidget(self.b_save)
            row.addWidget(self.b_save_as)
            pl.addLayout(row)
            sl.addWidget(pb)
            lay.addWidget(side)

            self.stack = QtWidgets.QStackedWidget()
            lay.addWidget(self.stack, 1)
            self.pages = {
                'home': HomePage(state), 'rig': RigPage(state), 'perception': PerceptionPage(state),
                'planning': PlanningPage(state), 'timeless': BenchPage(state, 'timeless'),
                'lite': BenchPage(state, 'lite'), 'results': ResultsPage(state),
                'research': ResearchPage(state), 'deploy': DeployPage(state)}
            for key, _ in PAGES:
                # pages keep their minimum sizes; a window narrower than that
                # scrolls instead of squeezing a pane to nothing
                wrap = QtWidgets.QScrollArea()
                wrap.setWidgetResizable(True)
                wrap.setFrameShape(QtWidgets.QFrame.NoFrame)
                wrap.setWidget(PageHost(self.pages[key]))
                self.stack.addWidget(wrap)
            self.pages['home'].navigate.connect(self.go)
            for k in ('timeless', 'lite'):
                self.pages[k].run_saved.connect(self.pages['results'].refresh)
                self.pages[k].run_saved.connect(self.pages['home'].refresh)
            self.go('home')
            self.b_save.clicked.connect(state.save)
            self.b_save_as.clicked.connect(self._save_as)
            state.dirty_changed.connect(lambda _: self._title())
            state.profile_loaded.connect(self._title)
            state.profile_edited.connect(lambda *_: self._title())
            state.message.connect(lambda m: self.statusBar().showMessage(m, 6000))
            QtWidgets.QShortcut(QtGui_seq('Ctrl+S'), self, activated=state.save)
            for i, (key, _) in enumerate(PAGES):
                QtWidgets.QShortcut(QtGui_seq('Ctrl+{}'.format(i + 1)), self,
                                    activated=lambda k=key: self.go(k))
            self._title()
            self.statusBar().showMessage('ready - {} profile(s); runs are saved in {}'.format(
                len(os.listdir(paths.PROFILES)), os.path.relpath(paths.RUNS, paths.ROOT)))

        def go(self, key):
            keys = [k for k, _ in PAGES]
            if key in keys:
                self.nav_buttons[key].setChecked(True)
                self.stack.setCurrentIndex(keys.index(key))

        def _title(self):
            p = self.state.profile
            dirty = self.state.dirty
            self.prof_label.setText(p.name)
            set_chip(self.prof_state, 'unsaved' if dirty else 'saved', theme.WARN if dirty else None)
            self.b_save.setEnabled(dirty)
            gates = getattr(p, 'gates', None) or {}
            g = gates.get('timeless')
            if g and g.get('passed'):
                set_chip(self.gate_chip, 'timeless gate passed', theme.GOOD)
            elif g:
                set_chip(self.gate_chip, 'timeless gate failed', theme.BAD)
            else:
                set_chip(self.gate_chip, 'gate not run')
            self.setWindowTitle('UCR Driverless Workbench - {}{}'.format(p.name, ' *' if dirty else ''))

        def _save_as(self):
            name, ok = QtWidgets.QInputDialog.getText(self, 'Save profile as', 'Name:',
                                                      text=self.state.profile.name + '_v2')
            if ok and name:
                name = ''.join(ch if ch.isalnum() or ch in '_-' else '_' for ch in name)
                self.state.save_as(name)
                self.pages['home'].refresh()

        def closeEvent(self, ev):
            if self.state.dirty:
                r = QtWidgets.QMessageBox.question(
                    self, 'Unsaved changes', 'Save changes to "{}" before quitting?'.format(self.state.profile.name),
                    QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel)
                if r == QtWidgets.QMessageBox.Cancel:
                    ev.ignore()
                    return
                if r == QtWidgets.QMessageBox.Save:
                    self.state.save()
            for k in ('timeless', 'lite'):
                self.pages[k].shutdown()
            super(MainWindow, self).closeEvent(ev)

    def QtGui_seq(s):
        from PyQt5 import QtGui
        return QtGui.QKeySequence(s)

    return MainWindow()


def main(argv=None):
    ap = argparse.ArgumentParser(description='UCR driverless workbench')
    ap.add_argument('--profile', default=None)
    ap.add_argument('--page', default='home', choices=[k for k, _ in PAGES])
    ap.add_argument('--ui-scale', type=float, default=None)
    ap.add_argument('--screenshot', default=None, help='render the window to a PNG and exit (testing)')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format='[%(asctime)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
    sys.setswitchinterval(0.001)
    from fsds_lite.__main__ import configure_qt_display, isolate_shader_cache
    isolate_shader_cache()
    os.environ.setdefault('QT_LOGGING_RULES', 'qt.qpa.wayland=false')
    configure_qt_display(args.ui_scale)
    try:
        from fsds_lite.render.viewer3d import configure_surface_format
        configure_surface_format()
    except ImportError:
        pass
    from PyQt5 import QtCore, QtWidgets
    from . import theme
    from .state import AppState
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('ucr-dv')
    theme.apply(app)
    state = AppState(args.profile)
    win = build_window(state)
    win.go(args.page)
    win.show()
    signal.signal(signal.SIGINT, lambda *_: win.close())
    tick = QtCore.QTimer()
    tick.timeout.connect(lambda: None)
    tick.start(200)
    if args.screenshot:
        def shoot():
            win.grab().save(args.screenshot)
            win.close()
        QtCore.QTimer.singleShot(1500, shoot)
    code = app.exec_()
    os._exit(code)          # skip sip teardown order issues (see fsds_lite / workbench)
