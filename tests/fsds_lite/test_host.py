"""SimHost wiring and the embeddable SimPanel (no ROS, no GL needed)."""
import os
import socket

import pytest

from fsds_lite.__main__ import parse_args
from fsds_lite.host import SimHost


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def make_host(*extra):
    return SimHost(parse_args(['--no-ros', '--no-cameras', '--rpc-port', str(free_port()),
                               '--track', 'skidpad'] + list(extra)), own_rclpy=False)


def test_host_starts_and_stops():
    host = make_host().start()
    try:
        assert host.mission == 'skidpad'
        assert host.rpc is not None and host.physics.is_alive()
        host.sim.enable_api_control(True)
        assert host.sim.api_control
    finally:
        host.stop()
    assert not host.physics.is_alive() or host.sim._stop.is_set()


def test_host_reports_port_in_use():
    first = make_host().start()
    try:
        args = parse_args(['--no-ros', '--no-cameras', '--rpc-port', str(first.rpc.address[1])])
        with pytest.raises(OSError):
            SimHost(args).start()
    finally:
        first.stop()


def test_embedded_panel_leaves_reset_to_the_workbench():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    QtWidgets = pytest.importorskip('PyQt5.QtWidgets')
    from PyQt5 import QtCore
    from fsds_lite.viewer import SimPanel
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    host = make_host().start()
    try:
        host.sim.set_paused(True)
        panel = SimPanel(host.sim, host.sensors, use_3d=False, embedded=True)
        assert panel.keys.handle_reset is False
        assert not panel.keys.press(QtCore.Qt.Key_R)       # workbench owns R
        assert panel.keys.press(QtCore.Qt.Key_W)
        panel.reset_inputs()
        assert not panel.keys.keys and panel.keys.cmd.throttle == 0
        panel._update_hud()
        assert 'skidpad' in panel.hud.text()
    finally:
        host.stop()
