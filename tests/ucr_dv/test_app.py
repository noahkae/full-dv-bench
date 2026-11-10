import os

import pytest

pytest.importorskip('PyQt5')
pytest.importorskip('pyqtgraph')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def test_main_window_builds_every_page(qapp_once=None):
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ucr_dv.app import theme
    from ucr_dv.app.main import PAGES, build_window
    from ucr_dv.app.state import AppState
    theme.apply(app)
    st = AppState('baseline')
    win = build_window(st)
    for key, _ in PAGES:
        win.go(key)
        app.processEvents()
    st.set_stage_param('planner', 'max_length', 25.0)
    assert st.dirty
    win.pages['planning'].preview()
    for k in ('timeless', 'lite'):
        win.pages[k].shutdown()
    win.deleteLater()
