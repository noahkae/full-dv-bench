"""Research page: the fsd-research library, and which algorithms in the stack each paper backs."""
from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from ...core import registry
from ...research.library import index_html, library
from .. import theme
from ..widgets import Splitter, button, label


class ResearchPage(QtWidgets.QWidget):
    def __init__(self, state, parent=None):
        super(ResearchPage, self).__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 14)
        outer.addWidget(label('Research', 'title'))
        outer.addWidget(label('The fsd-research library. Pick a paper to see what it covers and open it; pick a stage '
                              'to see the algorithms here and the papers behind them.', 'subtitle', True))
        bar = QtWidgets.QHBoxLayout()
        self.stage = QtWidgets.QComboBox()
        self.stage.addItem('all stages', '')
        for k in registry.STAGES:
            self.stage.addItem(registry.STAGE_LABELS[k], k)
        self.stage.currentIndexChanged.connect(self.refresh)
        bar.addWidget(label('stage'))
        bar.addWidget(self.stage)
        bar.addStretch(1)
        b_idx = button('Open research summary (index.html)')
        b_idx.clicked.connect(self._open_index)
        b_idx.setEnabled(index_html() is not None)
        bar.addWidget(b_idx)
        outer.addLayout(bar)
        self.papers = QtWidgets.QListWidget()
        self.papers.currentRowChanged.connect(self._show)
        self.papers.itemDoubleClicked.connect(lambda it: self._open_pdf())
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.detail = QtWidgets.QTextBrowser()
        rl.addWidget(self.detail, 1)
        b_pdf = button('Open PDF', primary=True, tip='Open the selected paper (or double-click it in the list)')
        b_pdf.clicked.connect(self._open_pdf)
        self.b_pdf = b_pdf
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(b_pdf)
        prow.addStretch(1)
        rl.addLayout(prow)
        self.algos = QtWidgets.QTextBrowser()
        rl.addWidget(label('Algorithms in this stack', 'h2'))
        rl.addWidget(self.algos, 3)
        outer.addWidget(Splitter([(self.papers, 300, 0.36), (right, 420, 0.64)]), 1)
        self.refresh()

    def refresh(self, *_):
        stage = self.stage.currentData()
        self.items = [p for p in library() if not stage or stage in p['stages']]
        self.papers.clear()
        for p in self.items:
            it = QtWidgets.QListWidgetItem(u'{}\n{} \u00b7 {}'.format(p['title'], p['team'], p['year']))
            if not p['available']:
                it.setForeground(QtGui.QColor(theme.MUTED))
            self.papers.addItem(it)
        if self.items:
            self.papers.setCurrentRow(0)
        kinds = [stage] if stage else list(registry.STAGES)
        html = ''
        for k in kinds:
            ents = registry.entries(k)
            html += '<h3>{}</h3><ul>'.format(registry.STAGE_LABELS[k])
            for e in ents.values():
                refs = ', '.join(e.refs) if e.refs else ''
                html += '<li><b>{}</b> <code>{}</code><br><span style="color:{}">{}</span>{}</li>'.format(
                    e.label, e.name, theme.MUTED, e.help.split('\n')[0],
                    ('<br><i>based on: {}</i>'.format(refs)) if refs else '')
            html += '</ul>'
        self.algos.setHtml(html)

    def _show(self, i):
        if i < 0 or i >= len(self.items):
            self.detail.setHtml('')
            self.b_pdf.setEnabled(False)
            return
        p = self.items[i]
        self.b_pdf.setEnabled(bool(p['available']))
        stages = ', '.join(registry.STAGE_LABELS.get(s, s) for s in p['stages'])
        avail = '' if p['available'] else '<p style="color:{}">file not found at {}</p>'.format(theme.WARN, p['path'])
        self.detail.setHtml('<h2>{}</h2><p>{} - {}</p><p>{}</p><p style="color:{}">stages: {}</p>{}'.format(
            p['title'], p['team'], p['year'], p['key'], theme.MUTED, stages, avail))

    def _open_pdf(self):
        i = self.papers.currentRow()
        if 0 <= i < len(self.items) and self.items[i]['available']:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.items[i]['path']))

    def _open_index(self):
        path = index_html()
        if path:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
