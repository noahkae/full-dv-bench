"""
App state shared by every page: the open profile and the active run.

Pages edit the profile through AppState so the dirty flag, live tuning of a
running stack and the "profile changed" notifications all happen in one place.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import traceback

from PyQt5 import QtCore

from .. import paths
from ..core.profile import Profile, list_profiles, load_profile


class AppState(QtCore.QObject):
    profile_loaded = QtCore.pyqtSignal()          # a different profile is open
    profile_edited = QtCore.pyqtSignal(str)       # what changed: 'rig', 'stage:<kind>', 'meta'
    dirty_changed = QtCore.pyqtSignal(bool)
    message = QtCore.pyqtSignal(str)

    def __init__(self, profile_name=None):
        super(AppState, self).__init__()
        os.makedirs(paths.PROFILES, exist_ok=True)
        names = list_profiles()
        name = profile_name if profile_name in names else ('baseline' if 'baseline' in names else None)
        self.profile = load_profile(name) if name else Profile('baseline')
        self.dirty = False
        self.runner = None                          # the EpisodeRunner currently driving, if any

    # ---- profile files ---------------------------------------------------
    def open(self, name):
        self.profile = load_profile(name)
        self._set_dirty(False)
        self.profile_loaded.emit()

    def new(self, name, base=None):
        p = (base or Profile()).copy(name)
        p.path = None
        self.profile = p
        p.save()
        self._set_dirty(False)
        self.profile_loaded.emit()

    def save(self):
        path = self.profile.save()
        self._set_dirty(False)
        self.message.emit('saved {}'.format(os.path.relpath(path, paths.ROOT)))

    def save_as(self, name):
        self.profile.name = name
        self.profile.path = None
        self.save()
        self.profile_loaded.emit()

    def delete(self, name):
        path = os.path.join(paths.PROFILES, name + '.json')
        if os.path.isfile(path):
            os.remove(path)

    def _set_dirty(self, d):
        if d != self.dirty:
            self.dirty = d
            self.dirty_changed.emit(d)

    # ---- edits ------------------------------------------------------------
    def set_stage(self, kind, name):
        self.profile.set_stage(kind, name, {})
        self._set_dirty(True)
        self.profile_edited.emit('stage:' + kind)
        if self.runner is not None and self.runner.running:
            self.message.emit('algorithm changes apply to the next run (parameters apply live)')

    def set_stage_param(self, kind, key, value):
        self.profile.stages[kind].setdefault('params', {})[key] = value
        self._set_dirty(True)
        self._live(kind, {key: value})

    def set_sensor_algo(self, sensor, name):
        spec = self.profile.rig.get(sensor)
        spec.algorithm = {'name': name, 'params': {}}
        self._set_dirty(True)
        self.profile_edited.emit('rig')

    def set_sensor_algo_param(self, sensor, key, value):
        spec = self.profile.rig.get(sensor)
        spec.algorithm.setdefault('params', {})[key] = value
        self._set_dirty(True)
        self._live('sensor:' + sensor, {key: value})

    def rig_edited(self):
        self._set_dirty(True)
        self.profile_edited.emit('rig')

    def meta_edited(self):
        self._set_dirty(True)
        self.profile_edited.emit('meta')

    def _live(self, stage, params):
        r = self.runner
        if r is not None and r.running:
            r.post('params', stage=stage, params=params)


class RunController(QtCore.QObject):
    """Owns one EpisodeRunner on a worker thread; the GUI polls it at 30 Hz."""
    started = QtCore.pyqtSignal(object)           # the runner (has .backend, .stack, .track)
    frame = QtCore.pyqtSignal(object)             # runner.latest
    finished = QtCore.pyqtSignal(object)          # result dict
    run_event = QtCore.pyqtSignal(str, str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, state, parent=None):
        super(RunController, self).__init__(parent)
        self.state = state
        self.runner = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll)
        self._events = queue.Queue()
        self._last_tick = -1
        self.save = True
        self.keep_backend = False

    @property
    def running(self):
        return self.runner is not None and self.runner.running

    def start(self, track, bench, seed=0, speed=0.0, backend_kw=None, save=True):
        from ..bench.session import prepare
        if self.running:
            return
        self.save = save
        try:
            runner, tr, stack, b = prepare(self.state.profile.copy(), track, bench, seed,
                                           backend_kw=backend_kw)
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        runner.cfg.realtime_factor = speed
        runner.on_event = lambda k, t: self._events.put((k, t))
        self.runner = runner
        self.state.runner = runner
        self._last_tick = -1
        self.started.emit(runner)
        runner.start_thread()
        self.timer.start(33)

    def stop(self):
        if self.runner is not None:
            self.runner.stop()

    def pause(self, value=None):
        if self.runner is not None:
            self.runner.post('pause', value=value)

    def step(self):
        if self.runner is not None:
            self.runner.post('step')

    def set_speed(self, value):
        if self.runner is not None:
            self.runner.post('speed', value=value)

    def _poll(self):
        r = self.runner
        if r is None:
            return
        while True:
            try:
                k, t = self._events.get_nowait()
            except queue.Empty:
                break
            self.run_event.emit(k, t)
        f = r.latest
        if f is not None and f.tick != self._last_tick:
            self._last_tick = f.tick
            self.frame.emit(f)
        if not r.running and r.result is not None:
            self.timer.stop()
            res = dict(r.result)
            if self.save:
                try:
                    from ..bench.records import save_run
                    res['path'] = save_run(r, r.stack.profile, r.cfg.track, r.cfg.bench, r.cfg.seed)
                except Exception as e:
                    res['save_error'] = str(e)
            if r.backend is not None and r.cfg.bench != 'timeless' and not self.keep_backend:
                pass                                    # the bench page closes it when the view is rebuilt
            self.state.runner = None
            self.finished.emit(res)


class Worker(QtCore.QThread):
    """Run a function off the GUI thread; progress(i, n, text) -> signal."""
    progress = QtCore.pyqtSignal(int, int, str)
    done = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super(Worker, self).__init__(parent)
        self.fn = fn
        self.abort = False

    def run(self):
        try:
            out = self.fn(self)
            self.done.emit(out)
        except Exception:
            self.error.emit(traceback.format_exc())
