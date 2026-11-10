"""Dark theme with UCalgary Racing red accents.

Colours come from a QPalette (so dialogs, viewports and item views agree) and
the stylesheet only adds shape: borders, radii, padding, custom arrows.
"""
from __future__ import annotations

import os
import tempfile

ACCENT = '#d6001c'          # UCalgary red
ACCENT_HI = '#ee1f37'
ACCENT_DIM = '#4a1a22'      # selection fill
ACCENT_2 = '#ffcd00'        # UCalgary gold
BG = '#111318'
PANEL = '#191c22'
PANEL_2 = '#232730'
PANEL_3 = '#2c313b'
BORDER = '#2e333d'
BORDER_HI = '#4a515e'
TEXT = '#e7e9ec'
MUTED = '#909aa6'
FAINT = '#5d6570'
GOOD = '#3fbf6f'
WARN = '#e8a33d'
BAD = '#e5534b'
LINK = '#7fb4ff'

STYLE = """
* {{ font-size: 10pt; }}
QMainWindow, QDialog {{ background: {BG}; }}
QLabel {{ background: transparent; color: {TEXT}; }}
QLabel#title {{ font-size: 17pt; font-weight: bold; }}
QLabel#subtitle {{ color: {MUTED}; padding-bottom: 6px; }}
QLabel#h2 {{ font-size: 12pt; font-weight: bold; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#faint {{ color: {FAINT}; font-size: 9pt; }}
QLabel#good {{ color: {GOOD}; font-weight: bold; }}
QLabel#bad {{ color: {BAD}; font-weight: bold; }}
QLabel#warn {{ color: {WARN}; font-weight: bold; }}
QLabel#empty {{ color: {FAINT}; font-size: 10pt; }}
QLabel#hud {{ color: {TEXT}; font-weight: bold; }}
QLabel#section {{ color: {FAINT}; font-size: 8pt; font-weight: bold; padding: 12px 16px 4px 16px; }}
QLabel#brand {{ font-size: 15pt; font-weight: bold; padding: 16px 16px 0px 16px; }}
QLabel#brandsub {{ color: {FAINT}; font-size: 8pt; padding: 0px 16px 10px 16px; }}
QLabel#chip {{ border: 1px solid {BORDER}; border-radius: 7px; padding: 2px 8px; font-size: 8pt;
              color: {MUTED}; background: {PANEL_2}; }}
QLabel#badge {{ background: {PANEL_3}; color: {TEXT}; border-radius: 11px; font-weight: bold; font-size: 9pt; }}

/* ---- sidebar ---- */
QWidget#sidebar {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
QWidget#profilebar {{ background: {PANEL}; border-top: 1px solid {BORDER}; border-right: 1px solid {BORDER}; }}
QPushButton#nav {{ text-align: left; padding: 8px 14px; border: none; border-left: 3px solid transparent;
                  border-radius: 0px; background: transparent; color: {MUTED}; }}
QPushButton#nav:hover {{ background: {PANEL_2}; color: {TEXT}; }}
QPushButton#nav:checked {{ background: {PANEL_2}; color: white; border-left: 3px solid {ACCENT};
                          font-weight: bold; }}

/* ---- containers ---- */
QFrame#card, QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#step {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#step:hover {{ background: {PANEL_2}; border-color: {BORDER_HI}; }}
QGroupBox {{ margin-top: 22px; padding: 2px 4px 4px 4px; font-weight: bold; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 4px; top: 2px;
                   padding: 0 4px; color: {MUTED}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QSplitter::handle {{ background: transparent; border-radius: 2px; }}
QSplitter::handle:hover {{ background: {BORDER_HI}; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QStatusBar QLabel {{ color: {MUTED}; }}
QToolTip {{ background: {PANEL_3}; color: {TEXT}; border: 1px solid {BORDER_HI}; padding: 4px 6px; }}
QGraphicsView {{ border: 1px solid {BORDER}; background: {PANEL}; }}

/* ---- item views ---- */
QListWidget, QTreeWidget, QTableWidget, QPlainTextEdit, QTextBrowser {{
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; outline: 0; }}
QTextBrowser, QPlainTextEdit {{ padding: 6px; }}
QListWidget::item {{ padding: 5px 8px; border-radius: 4px; }}
QTreeWidget::item {{ padding: 4px 4px; }}
QListWidget::item:hover, QTreeWidget::item:hover {{ background: {PANEL_2}; }}
QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{
    background: {ACCENT_DIM}; color: white; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{ background: {PANEL_2}; border: none; border-bottom: 1px solid {BORDER};
                       padding: 5px 8px; color: {MUTED}; font-weight: bold; }}
QTableCornerButton::section {{ background: {PANEL_2}; border: none; }}
QTableWidget {{ gridline-color: {BORDER}; alternate-background-color: #1d2027; }}

/* ---- buttons ---- */
QPushButton {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 14px;
              color: {TEXT}; }}
QPushButton:hover {{ background: {PANEL_3}; border-color: {BORDER_HI}; }}
QPushButton:pressed {{ background: {PANEL}; }}
QPushButton:focus {{ border-color: {BORDER_HI}; }}
QPushButton:disabled {{ color: {FAINT}; background: {PANEL}; border-color: {BORDER}; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: bold; }}
QPushButton#primary:hover {{ background: {ACCENT_HI}; border-color: {ACCENT_HI}; }}
QPushButton#primary:pressed {{ background: #b00017; }}
QPushButton#primary:disabled {{ background: #3a1a1f; border-color: #3a1a1f; color: #96787b; }}

/* ---- inputs ---- */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 5px; padding: 3px 8px; min-height: 22px;
    color: {TEXT}; selection-background-color: {ACCENT}; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{ border-color: {BORDER_HI}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {{ color: {FAINT}; }}
QComboBox {{ padding-right: 22px; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 22px; border: none; }}
QComboBox::down-arrow {{ image: url({ICONS}/down.png); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{ background: {PANEL_2}; border: 1px solid {BORDER_HI}; outline: 0;
                              selection-background-color: {ACCENT}; selection-color: white; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 20px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border; subcontrol-position: top right; width: 18px; border: none;
    border-left: 1px solid {BORDER}; border-top-right-radius: 5px; background: transparent; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; subcontrol-position: bottom right; width: 18px; border: none;
    border-left: 1px solid {BORDER}; border-bottom-right-radius: 5px; background: transparent; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {PANEL_3}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({ICONS}/up.png); width: 8px; height: 8px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({ICONS}/down.png); width: 8px; height: 8px; }}
QCheckBox {{ background: transparent; color: {TEXT}; spacing: 7px; }}
QCheckBox::indicator, QListWidget::indicator {{ width: 15px; height: 15px; border: 1px solid {BORDER_HI};
                                              border-radius: 4px; background: {PANEL_2}; }}
QCheckBox::indicator:hover {{ border-color: {MUTED}; }}
QCheckBox::indicator:checked, QListWidget::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT}; image: url({ICONS}/check.png); }}
QSlider::groove:horizontal {{ height: 4px; background: {PANEL_3}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
                             background: {TEXT}; border: 1px solid {BG}; }}
QSlider::handle:horizontal:disabled {{ background: {FAINT}; }}

/* ---- tabs ---- */
QTabWidget::pane {{ border: none; border-top: 1px solid {BORDER}; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 7px 14px; border: none;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected {{ color: white; border-bottom: 2px solid {ACCENT}; }}

QProgressBar {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 5px; text-align: center;
               color: {TEXT}; min-height: 18px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}

/* ---- scrollbars ---- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {PANEL_3}; border-radius: 3px; }}
QScrollBar::handle:vertical {{ min-height: 28px; }}
QScrollBar::handle:horizontal {{ min-width: 28px; }}
QScrollBar::handle:hover {{ background: {BORDER_HI}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QAbstractScrollArea::corner {{ background: transparent; }}
"""

_icon_dir = None


def _icons():
    """Draw the few glyphs the stylesheet needs (no image files to ship)."""
    global _icon_dir
    if _icon_dir is not None:
        return _icon_dir
    from PyQt5 import QtCore, QtGui
    d = tempfile.mkdtemp(prefix='ucr-dv-ui-')
    glyphs = {'down': (MUTED, [(2.5, 4.5), (6, 8), (9.5, 4.5)]),
              'up': (MUTED, [(2.5, 7.5), (6, 4), (9.5, 7.5)]),
              'check': ('#ffffff', [(2.5, 6.2), (5, 8.7), (9.5, 3.5)])}
    for name, (color, pts) in glyphs.items():
        for scale, suffix in ((1, ''), (2, '@2x')):
            img = QtGui.QImage(12 * scale, 12 * scale, QtGui.QImage.Format_ARGB32_Premultiplied)
            img.fill(QtCore.Qt.transparent)
            p = QtGui.QPainter(img)
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.scale(scale, scale)
            pen = QtGui.QPen(QtGui.QColor(color))
            pen.setWidthF(1.7)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            pen.setJoinStyle(QtCore.Qt.RoundJoin)
            p.setPen(pen)
            p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in pts]))
            p.end()
            img.save(os.path.join(d, name + suffix + '.png'))
    _icon_dir = d.replace(os.sep, '/')
    return _icon_dir


def palette():
    from PyQt5 import QtGui
    Q, P = QtGui.QColor, QtGui.QPalette
    pal = P()
    for role, color in ((P.Window, BG), (P.WindowText, TEXT), (P.Base, PANEL), (P.AlternateBase, PANEL_2),
                        (P.Text, TEXT), (P.Button, PANEL_2), (P.ButtonText, TEXT), (P.BrightText, '#ffffff'),
                        (P.ToolTipBase, PANEL_3), (P.ToolTipText, TEXT), (P.Highlight, ACCENT_DIM),
                        (P.HighlightedText, '#ffffff'), (P.Link, LINK), (P.LinkVisited, LINK),
                        (P.Light, PANEL_3), (P.Midlight, PANEL_2), (P.Mid, BORDER), (P.Dark, BG),
                        (P.Shadow, '#000000')):
        pal.setColor(role, Q(color))
    if hasattr(P, 'PlaceholderText'):
        pal.setColor(P.PlaceholderText, Q(FAINT))
    for role in (P.WindowText, P.Text, P.ButtonText):
        pal.setColor(P.Disabled, role, Q(FAINT))
    return pal


def mono(point_size=None):
    """The system fixed-width font, for logs, parameter dumps and live read-outs."""
    from PyQt5 import QtGui
    f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
    if point_size:
        f.setPointSizeF(point_size)
    return f


def apply(app):
    app.setStyle('Fusion')
    app.setPalette(palette())
    env = dict(globals())
    env['ICONS'] = _icons()
    app.setStyleSheet(STYLE.format(**env))
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, background=PANEL, foreground=MUTED)
