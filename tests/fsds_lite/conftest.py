import os
import sys

import pytest

SIM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM_DIR)

from fsds_lite.sensors import load_settings  # noqa: E402
from fsds_lite.simulator import Simulator  # noqa: E402
from fsds_lite.track import load_track  # noqa: E402

SETTINGS = os.path.join(SIM_DIR, 'data', 'vehicle', 'settings.json')


@pytest.fixture
def sensors():
    s = load_settings(SETTINGS)
    s.imu.noise = False             # exact values in tests; noise has its own test
    return s


@pytest.fixture
def sim(sensors):
    return Simulator(load_track('acceleration'), sensors, seed=0)
