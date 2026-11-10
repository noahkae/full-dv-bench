"""
Stack profile: one JSON file = one complete driverless stack.

    {
      "name": "baseline",
      "description": "...",
      "vehicle": "ucr_ev",                       # data/vehicle/<name>.json
      "rig": { "name": ..., "sensors": [ SensorSpec... ] },   # incl. each sensor's algorithm
      "stages": {
        "fusion":       {"name": "nearest_assoc", "params": {...}},
        "odometry":     {...}, "mapping": {...}, "planner": {...},
        "speed":        {...}, "lateral": {...}, "longitudinal": {...}, "mission": {...}
      },
      "pipeline_hz": 20
    }

Profiles live in data/profiles/. The same profile runs on every bench.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict

from .. import paths
from ..sensors.rig import Rig, default_rig
from . import registry

DEFAULT_STAGES = {
    'fusion': {'name': 'nearest_assoc', 'params': {}},
    'odometry': {'name': 'dead_reckoning', 'params': {}},
    'mapping': {'name': 'cone_map', 'params': {}},
    'planner': {'name': 'track_template', 'params': {}},
    'speed': {'name': 'grip_limited', 'params': {}},
    'lateral': {'name': 'pure_pursuit', 'params': {}},
    'longitudinal': {'name': 'pid_speed', 'params': {}},
    'mission': {'name': 'auto', 'params': {}},
}
PIPELINE_STAGES = ('fusion', 'odometry', 'mapping', 'planner', 'speed', 'lateral', 'longitudinal', 'mission')


class Profile(object):
    def __init__(self, name='baseline', description='', vehicle='ucr_ev', rig=None, stages=None,
                 pipeline_hz=20.0, path=None, gates=None):
        self.name = name
        self.gates = dict(gates or {})          # promotion-gate results per bench
        self.description = description
        self.vehicle = vehicle
        self.rig = rig or default_rig()
        self.stages = copy.deepcopy(DEFAULT_STAGES)
        for k, v in (stages or {}).items():
            self.stages[k] = {'name': v.get('name', DEFAULT_STAGES.get(k, {}).get('name')),
                              'params': dict(v.get('params', {}))}
        self.pipeline_hz = float(pipeline_hz)
        self.path = path

    # ---- (de)serialise ----------------------------------------------------
    def to_dict(self) -> Dict:
        return {'name': self.name, 'description': self.description, 'vehicle': self.vehicle,
                'rig': self.rig.to_dict(), 'stages': copy.deepcopy(self.stages),
                'pipeline_hz': self.pipeline_hz, 'gates': copy.deepcopy(self.gates)}

    @classmethod
    def from_dict(cls, d, path=None):
        return cls(d.get('name', 'unnamed'), d.get('description', ''), d.get('vehicle', 'ucr_ev'),
                   Rig.from_dict(d['rig']) if 'rig' in d else None, d.get('stages'),
                   d.get('pipeline_hz', 20.0), path, d.get('gates'))

    def copy(self, name=None):
        p = Profile.from_dict(copy.deepcopy(self.to_dict()))
        p.path = self.path
        if name:
            p.name = name
            p.gates = {}
        return p

    def save(self, path=None):
        path = path or self.path or os.path.join(paths.PROFILES, self.name + '.json')
        d = self.to_dict()
        d['saved'] = time.strftime('%Y-%m-%d %H:%M:%S')
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)
        self.path = path
        return path

    # ---- helpers ---------------------------------------------------------
    def stage(self, kind):
        return self.stages[kind]

    def set_stage(self, kind, name, params=None):
        self.stages[kind] = {'name': name, 'params': dict(params or {})}

    def resolved(self, kind):
        s = self.stages[kind]
        return registry.get(kind, s['name']).resolve(s.get('params'))

    @property
    def uses_ground_truth(self):
        return self.stages['odometry']['name'] == 'ground_truth'

    def validate(self):
        """List of human-readable problems (empty = OK)."""
        problems = []
        for kind in PIPELINE_STAGES:
            try:
                registry.get(kind, self.stages[kind]['name'])
            except KeyError as e:
                problems.append(str(e))
        perc = self.rig.perception()
        if not perc:
            problems.append('the rig has no perception sensor (add a LiDAR or a camera)')
        names = [s.name for s in self.rig.sensors]
        if len(set(names)) != len(names):
            problems.append('sensor names must be unique')
        needs_colors = 'colors' in registry.get('planner', self.stages['planner']['name']).needs
        sees_colors = any(s.type in ('camera', 'stereo', 'timeless_classic') for s in perc)
        if needs_colors and not sees_colors:
            problems.append('planner {!r} needs cone colours but no sensor sees colour '
                            '(add a camera, or use delaunay/gates)'.format(self.stages['planner']['name']))
        if self.stages['odometry']['name'] == 'ekf' and self.rig.ego('gnss') is None:
            problems.append('EKF odometry without a GNSS sensor is just dead reckoning')
        return problems


def list_profiles():
    if not os.path.isdir(paths.PROFILES):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(paths.PROFILES) if f.endswith('.json'))


def load_profile(name_or_path) -> Profile:
    path = name_or_path if os.path.isfile(name_or_path) else os.path.join(paths.PROFILES, name_or_path + '.json')
    with open(path) as f:
        return Profile.from_dict(json.load(f), path)
