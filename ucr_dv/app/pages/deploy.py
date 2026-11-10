"""Deploy page: export to the ROS 2 stack / FSDS, and the Assetto Corsa roadmap."""
from __future__ import annotations

import json
import os

from PyQt5 import QtCore, QtGui, QtWidgets

from ... import paths
from ...deploy import ROS_PKG, export_ros, flat_params, fsds_settings
from .. import theme
from ..widgets import button, card, label


class DeployPage(QtWidgets.QWidget):
    def __init__(self, state, parent=None):
        super(DeployPage, self).__init__(parent)
        self.state = state
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.addWidget(label('Deploy', 'title'))
        outer.addWidget(label('The same Stack class runs inside a ROS 2 node on the car (Galactic, Python 3.8). '
                              'Export a profile and launch it; every parameter stays tunable with ros2 param set.',
                              'subtitle', True))
        tabs = QtWidgets.QTabWidget()
        outer.addWidget(tabs, 1)

        ros = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(ros)
        row = QtWidgets.QHBoxLayout()
        b_exp = button('Export profile to ros/ucr_dv_ros/config', primary=True)
        b_exp.clicked.connect(self._export)
        b_fsds = button('Export FSDS settings.json for this rig')
        b_fsds.clicked.connect(self._fsds)
        b_dir = button('Open ROS package folder')
        b_dir.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(ROS_PKG)))
        for b in (b_exp, b_fsds, b_dir):
            row.addWidget(b)
        row.addStretch(1)
        rl.addLayout(row)
        self.out = QtWidgets.QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setFont(theme.mono(9.5))
        self.out.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        rl.addWidget(self.out, 1)
        tabs.addTab(ros, 'ROS 2 / car')

        ac = QtWidgets.QTextBrowser()
        doc = os.path.join(paths.ROOT, 'docs', 'ASSETTO_CORSA.md')
        if os.path.isfile(doc):
            with open(doc) as f:
                text = f.read()
            if hasattr(ac, 'setMarkdown'):
                ac.setMarkdown(text)
            else:
                ac.setPlainText(text)
        tabs.addTab(ac, 'Assetto Corsa (planned)')
        state.profile_loaded.connect(self._preview)
        self._preview()

    def _preview(self):
        p = self.state.profile
        flat = flat_params(p)
        self.out.setPlainText(
            'Profile "{}" -> {} ROS parameters.\n\n'
            'On the car / WSL (Ubuntu 20.04, ROS 2 Galactic):\n'
            '  1. copy or symlink ucr-dv-stack into the colcon workspace src/ (ros/ucr_dv_ros is an ament_python package)\n'
            '  2. colcon build --packages-select ucr_dv_ros && source install/setup.bash\n'
            '  3. ros2 launch ucr_dv_ros stack.launch.py profile:={} \n'
            '  4. live tuning:  ros2 param set /ucr_dv_stack planner.max_gap 5.5\n\n'
            'Topics in (fsds_ros2_bridge / fsds_lite contract): /lidar/<sensor> PointCloud2, /fsds/<cam>/image_color,\n'
            '/gss, /imu, /gps, (/testing_only/odom only if odometry = ground_truth)\n'
            'Topics out: /perception/cones (PoseArray), /slam/map, /planning/path, /control_command (fs_msgs)\n\n'
            'Parameters:\n{}'.format(p.name, len(flat), p.name,
                                     '\n'.join('  {} = {}'.format(k, v) for k, v in sorted(flat.items()))))

    def _export(self):
        if self.state.dirty:
            self.state.save()
        j, y = export_ros(self.state.profile)
        self.out.appendPlainText('\nwrote {}\nwrote {}'.format(j, y))

    def _fsds(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'FSDS settings.json',
                                                        os.path.join(paths.ROOT, 'settings_{}.json'.format(
                                                            self.state.profile.name)), 'JSON (*.json)')
        if path:
            with open(path, 'w') as f:
                json.dump(fsds_settings(self.state.profile), f, indent=2)
            self.out.appendPlainText('\nwrote ' + path)
