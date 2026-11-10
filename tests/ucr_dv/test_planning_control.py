import math

import numpy as np
import pytest

from ucr_dv.core.registry import get
from ucr_dv.core.types import BLUE, UNKNOWN, YELLOW, Pose2D
from ucr_dv.planning.planners import bowyer_watson, delaunay_edges
from ucr_dv.vehicle import VehicleSpec, load_vehicle

VEH = VehicleSpec(load_vehicle('ucr_ev'))


def _corridor(curv=0.0, colored=True):
    s = np.arange(2.0, 22.0, 3.0)
    out = []
    for x in s:
        y0 = 0.5 * curv * x * x
        out.append((x, y0 + 1.6, BLUE if colored else UNKNOWN, 1.0))
        out.append((x, y0 - 1.6, YELLOW if colored else UNKNOWN, 1.0))
    return np.array(out)


@pytest.mark.parametrize('planner', ['delaunay', 'boundaries', 'gates', 'greedy_pairing'])
def test_planners_follow_a_straight(planner):
    path = get('planner', planner).create({}).plan(_corridor(), {'pose': Pose2D()})
    assert len(path) > 5 and path[-1, 0] > 10
    assert np.max(np.abs(path[:, 1])) < 0.4


@pytest.mark.parametrize('planner', ['delaunay', 'boundaries'])
def test_planners_work_without_colours(planner):
    path = get('planner', planner).create({}).plan(_corridor(0.01, colored=False), {'pose': Pose2D()})
    assert path[-1, 0] > 10 and path[-1, 1] > 0.3        # bends left with the corridor


def test_bowyer_watson_matches_scipy():
    pytest.importorskip('scipy')
    pts = np.random.default_rng(3).uniform(0, 20, size=(25, 2))
    a = {tuple(e) for e in delaunay_edges(pts).tolist()}
    tris = bowyer_watson(pts)
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [0, 2]]])
    e.sort(axis=1)
    b = {tuple(x) for x in np.unique(e, axis=0).tolist()}
    assert len(a ^ b) <= 2                           # co-circular ties may differ


@pytest.mark.parametrize('lat', ['pure_pursuit', 'stanley', 'p_steer', 'mpc'])
def test_lateral_controllers_steer_towards_a_left_bend(lat):
    xs = np.linspace(0, 15, 31)
    path = np.column_stack([xs, 0.05 * xs ** 2])
    c = get('lateral', lat).create({}, vehicle=VEH)
    assert c.steer(path, 6.0, {}) > 0.01
    assert c.steer(path * np.array([1, -1]), 6.0, {}) < -0.01


def test_speed_profile_slows_for_corners_and_respects_caps():
    sp = get('speed', 'grip_limited').create({'v_max': 20.0, 'end_speed': 20.0})
    straight = np.column_stack([np.linspace(0, 30, 61), np.zeros(61)])
    a = np.linspace(0, math.pi, 61)
    corner = np.column_stack([6 * np.sin(a), 6 - 6 * np.cos(a)])
    _, v_s = sp.profile(straight, 10.0, {})
    sp.reset()
    _, v_c = sp.profile(corner, 10.0, {})
    assert v_c < v_s
    sp.reset()
    _, v0 = sp.profile(straight, 10.0, {'speed_cap': 0.0})
    assert v0 == 0.0


def test_pid_pedals():
    pid = get('longitudinal', 'pid_speed').create({}, vehicle=VEH)
    thr, brk, _ = pid.command(10.0, 5.0, 0.05)
    assert thr > 0 and brk == 0
    pid.reset()
    thr, brk, _ = pid.command(2.0, 10.0, 0.05)
    assert thr == 0 and brk > 0
