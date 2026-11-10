"""Where things live. Everything is relative to the ucr-dv-stack folder."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
TRACKS = os.path.join(DATA, 'tracks')
VEHICLES = os.path.join(DATA, 'vehicle')
PROFILES = os.path.join(DATA, 'profiles')
RIGS = os.path.join(DATA, 'rigs')
RUNS = os.environ.get('UCR_DV_RUNS', os.path.join(ROOT, 'runs'))
# The research library is a sibling folder of the master Capstone folder.
RESEARCH = os.environ.get('UCR_DV_RESEARCH',
                          os.path.join(os.path.dirname(ROOT), 'fsd-research'))

# The vendored fsds_lite package sits next to ucr_dv; make sure it imports
# even when ucr_dv is used from another working directory.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def vehicle_file(name: str) -> str:
    """A vehicle JSON by bare name ('vehicle', 'ucr_ev') or path."""
    if os.path.isfile(name):
        return name
    path = os.path.join(VEHICLES, name if name.endswith('.json') else name + '.json')
    if not os.path.isfile(path):
        raise FileNotFoundError('vehicle file not found: {}'.format(name))
    return path


def list_vehicles():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(VEHICLES)
                  if f.endswith('.json') and f != 'settings.json')
