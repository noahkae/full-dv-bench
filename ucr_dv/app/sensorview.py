"""Live sensor feeds for the 3D bench: camera images with detection boxes, LiDAR bird's-eye view."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from ..core.types import COLOR_RGB, CameraImage, PointCloud


class SensorView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(SensorView, self).__init__(parent)
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.panels = {}
        self.boxes = {}
        self.note = QtWidgets.QLabel('Sensor feeds appear here while a 3D run is going (raw-fidelity sensors only).')
        self.note.setObjectName('muted')
        self.grid.addWidget(self.note, 0, 0)

    def reset(self):
        for w in list(self.panels.values()):
            w[0].setParent(None)
            w[0].deleteLater()
        self.panels = {}
        self.boxes = {}
        self.note.show()

    def _panel(self, name, kind):
        if name in self.panels:
            return self.panels[name]
        self.note.hide()
        pw = pg.PlotWidget(title=name)
        pw.setAspectLocked(True)
        if kind == 'image':
            pw.invertY(True)
            pw.hideAxis('left')
            pw.hideAxis('bottom')
            item = pg.ImageItem(axisOrder='row-major')
            pw.addItem(item)
        else:
            pw.showGrid(x=True, y=True, alpha=0.15)
            item = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush(214, 60, 60, 150))
            pw.addItem(item)
            pw.setRange(xRange=(-5, 30), yRange=(-15, 15))
        overlay = pg.PlotDataItem(connect='finite', pen=pg.mkPen((90, 255, 120), width=2))
        pw.addItem(overlay)
        dets = pg.ScatterPlotItem(size=10, symbol='o')
        pw.addItem(dets)
        n = len(self.panels)
        self.grid.addWidget(pw, n // 2, n % 2)
        self.panels[name] = (pw, item, overlay, dets)
        return self.panels[name]

    def update_frame(self, f):
        dbg = f.out.debug
        for name, fr in f.frames.items():
            if isinstance(fr, CameraImage) and not fr.is_depth:
                pw, item, overlay, _ = self._panel(name, 'image')
                img = fr.image[..., ::-1] if fr.image.ndim == 3 else fr.image   # BGR -> RGB
                item.setImage(img, autoLevels=False, levels=(0, 255))
                boxes = (dbg.get('perception:' + name) or {}).get('boxes', [])
                xs, ys = [], []
                for (x0, y0, x1, y1, c) in boxes:
                    xs += [x0, x1, x1, x0, x0, np.nan]
                    ys += [y0, y0, y1, y1, y0, np.nan]
                overlay.setData(xs, ys)
                pw.setTitle('{} - {} cones'.format(name, len(boxes)))
            elif isinstance(fr, PointCloud):
                pw, item, overlay, dets = self._panel(name, 'cloud')
                pts = fr.points
                pts = pts[(pts[:, 2] > 0.04) & (np.abs(pts[:, 0]) < 40) & (np.abs(pts[:, 1]) < 40)]
                item.setData(pos=pts[:, :2])
                c = f.out.cones_by_sensor.get(name)
                if c is not None and len(c):
                    dets.setData(pos=c[:, :2], brush=None,
                                 pen=[pg.mkPen(*COLOR_RGB.get(int(k), (230, 230, 230)), width=2) for k in c[:, 2]])
                st = dbg.get('perception:' + name) or {}
                pw.setTitle('{} - {} non-ground points, {} clusters, {} cones'.format(
                    name, st.get('points', len(pts)), st.get('clusters', '-'), 0 if c is None else len(c)))
