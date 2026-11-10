"""
Plugin registry: every swappable algorithm registers itself here with a
parameter schema. The GUI builds its forms from the schema, profiles store
{name, params}, and the ROS wrappers expose the same params - so adding an
algorithm is one decorated class and nothing else.

    @register('planner', 'delaunay', label='Delaunay midline',
              params=[P('max_width', 6.0, 2.0, 10.0, unit='m', help='...')],
              refs=['KIT 2022', 'AMZ 2019'])
    class DelaunayPlanner(Component):
        def plan(self, cones, ctx): ...

Stages (the `kind` argument):
    lidar      raw point cloud -> cones           (perception/lidar.py)
    camera     raw image -> cones                 (perception/camera.py)
    fusion     cones per sensor -> one cone list  (perception/fusion.py)
    odometry   ego measurements -> pose           (estimation/odometry.py)
    mapping    cones + pose -> cone map           (estimation/mapping.py)
    planner    cones -> path (car frame)          (planning/planners.py)
    speed      path -> speed profile              (planning/speed.py)
    lateral    path -> steering                   (control/lateral.py)
    longitudinal  speed target -> throttle/brake  (control/longitudinal.py)
    mission    lap counting / finish / stop logic (planning/mission.py)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

STAGES = ('lidar', 'camera', 'fusion', 'odometry', 'mapping', 'planner', 'speed',
          'lateral', 'longitudinal', 'mission')

STAGE_LABELS = {
    'lidar': 'LiDAR cone detection', 'camera': 'Camera cone detection', 'fusion': 'Sensor fusion',
    'odometry': 'Odometry / state estimation', 'mapping': 'Cone mapping (SLAM)',
    'planner': 'Path planner', 'speed': 'Speed profile', 'lateral': 'Lateral control (steering)',
    'longitudinal': 'Longitudinal control (pedals)', 'mission': 'Mission logic',
}


@dataclass
class P:
    """One tunable parameter. Type follows the default (bool/int/float/str)."""
    name: str
    default: Any
    lo: Optional[float] = None
    hi: Optional[float] = None
    step: Optional[float] = None
    unit: str = ''
    help: str = ''
    choices: Optional[List[str]] = None

    @property
    def type(self):
        return type(self.default)

    def coerce(self, value):
        t = self.type
        if t is bool:
            if isinstance(value, str):
                return value.strip().lower() in ('1', 'true', 'yes', 'on')
            return bool(value)
        if t is int:
            v = int(round(float(value)))
        elif t is float:
            v = float(value)
        else:
            v = str(value)
            if self.choices and v not in self.choices:
                raise ValueError('{}: {!r} not in {}'.format(self.name, v, self.choices))
            return v
        if self.lo is not None:
            v = max(v, t(self.lo))
        if self.hi is not None:
            v = min(v, t(self.hi))
        return v


@dataclass
class Entry:
    kind: str
    name: str
    cls: type
    label: str
    params: List[P]
    help: str = ''
    refs: List[str] = field(default_factory=list)
    needs: List[str] = field(default_factory=list)   # e.g. ['raw_lidar'], ['colors']

    def defaults(self) -> Dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def param(self, name) -> Optional[P]:
        for p in self.params:
            if p.name == name:
                return p
        return None

    def resolve(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Defaults overlaid with `params`, coerced; unknown keys are dropped."""
        out = self.defaults()
        for k, v in (params or {}).items():
            p = self.param(k)
            if p is not None:
                out[k] = p.coerce(v)
        return out

    def create(self, params=None, **context):
        return self.cls(self.resolve(params), **context)


_REGISTRY: Dict[str, Dict[str, Entry]] = {k: {} for k in STAGES}


def register(kind, name, label=None, params=(), help='', refs=(), needs=()):
    if kind not in _REGISTRY:
        raise KeyError('unknown stage {!r}'.format(kind))

    def deco(cls):
        doc = (cls.__doc__ or '').strip()
        _REGISTRY[kind][name] = Entry(kind, name, cls, label or name, list(params),
                                      help or doc, list(refs), list(needs))
        cls.registry_name = name
        return cls
    return deco


def _load_all():
    """Import every algorithm module so their decorators run."""
    from ..perception import lidar, camera, fusion  # noqa: F401
    from ..estimation import odometry, mapping  # noqa: F401
    from ..planning import planners, speed, mission  # noqa: F401
    from ..control import lateral, longitudinal  # noqa: F401


_loaded = False


def entries(kind) -> Dict[str, Entry]:
    global _loaded
    if not _loaded:
        _loaded = True
        _load_all()
    return _REGISTRY[kind]


def get(kind, name) -> Entry:
    e = entries(kind)
    if name not in e:
        raise KeyError('no {} algorithm named {!r} (have: {})'.format(kind, name, ', '.join(sorted(e))))
    return e[name]


class Component(object):
    """Base class: holds params, supports live re-tuning and reset."""
    registry_name = ''

    def __init__(self, params, **context):
        self.params = dict(params)
        self.context = context

    def set_params(self, **params):
        """Live tuning from the GUI (called between ticks on the worker thread)."""
        entry = None
        for kind in STAGES:
            if self.registry_name in _REGISTRY[kind] and _REGISTRY[kind][self.registry_name].cls is type(self):
                entry = _REGISTRY[kind][self.registry_name]
        for k, v in params.items():
            p = entry.param(k) if entry else None
            self.params[k] = p.coerce(v) if p else v
        self.on_params_changed()

    def on_params_changed(self):
        pass

    def reset(self):
        pass

    def __getitem__(self, key):
        return self.params[key]


def clone(d):
    return copy.deepcopy(d)
