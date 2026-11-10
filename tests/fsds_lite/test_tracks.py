"""Builtin events against the FS Rules 2026 / FS Driverless Specification 2026."""

import numpy as np
import pytest

from fsds_lite.track import load_track, _curvature, TimingLine


def cones(track, tag):
    return track.positions[[c == tag for c in track.colors]]


def test_acceleration_layout_D5():
    t = load_track('acceleration', scenery=False)
    assert t.mission == 'acceleration' and t.laps == 1
    start, finish = t.start_line, t.finish_line
    assert abs((finish.a[0] - start.a[0]) - 75.0) < 1e-9                  # D 5.1.1: 75 m
    assert abs(abs(start.a[1] - start.b[1]) - 3.0) < 1e-9                 # >= 3 m wide
    assert abs(start.a[0] - (0.30 + 0.9)) < 1e-9                          # D 5.2.3: nose 0.30 m back
    for tag in ('blue', 'yellow'):
        xs = np.sort(cones(t, tag)[:, 0])
        timed = xs[(xs > start.a[0]) & (xs < finish.a[0])]
        assert np.diff(timed).max() <= 5.0 + 1e-9                         # ~5 m intervals
    assert np.all(cones(t, 'blue')[:, 1] > 0) and np.all(cones(t, 'yellow')[:, 1] < 0)
    assert len(cones(t, 'big_orange')) == 8                               # before+after both lines
    stop = cones(t, 'orange')
    assert stop[:, 0].max() - finish.a[0] >= 75.0 - 1e-9                  # D 5.2.4: 75 m to stop


def test_skidpad_layout_D4():
    t = load_track('skidpad', scenery=False)
    assert t.mission == 'skidpad' and t.laps == 4
    right, left = (np.array(c) for c in t.skidpad_centres)
    assert abs(np.linalg.norm(left - right) - 18.25) < 1e-9              # D 4.1.2
    for centre, inner_tag, outer_tag in ((right, 'yellow', 'blue'), (left, 'blue', 'yellow')):
        r_inner = np.linalg.norm(cones(t, inner_tag) - centre, axis=1)
        r_outer = np.linalg.norm(cones(t, outer_tag) - centre, axis=1)
        assert np.sum(np.isclose(r_inner, 15.25 / 2)) == 17                # D 4.1.3: 17 inside
        assert np.sum(np.isclose(r_outer, 21.25 / 2)) == 13                # 13 outside
    assert abs(right[0] - (15.0 + 0.9)) < 1e-9                            # D 4.2.4: staged 15 m back
    assert len(cones(t, 'big_orange')) == 4
    lane = cones(t, 'orange')
    assert lane[:, 0].max() - right[0] <= 26.0 + 1e-9                     # D 4.2.5: stop within 25 m


@pytest.mark.parametrize('name', ['autocross', 'trackdrive'])
def test_circuit_follows_D6_and_D8(name):
    t = load_track(name, scenery=False)
    c = t.centreline
    seg = np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)
    k = _curvature(c)
    assert 200.0 < seg.sum() < 1500.0                                     # D 6.1.2 / D 8.1.2
    assert 2.0 / np.abs(k).max() >= 9.0                                   # D 6.1.1: >= 9 m turning diameter
    run, longest = 0.0, 0.0
    for i in range(2 * len(c)):                                           # straights <= 80 m
        if abs(k[i % len(c)]) < 1.0 / 150.0:
            run += seg[i % len(c)]
            longest = max(longest, run)
        else:
            run = 0.0
    assert longest <= 80.0
    # sections that are far apart along the lap never come within a track width + margin
    n = len(c)
    idx = np.arange(n)
    d = np.linalg.norm(c[:, None] - c[None], axis=2)
    sep = np.minimum(np.abs(idx[:, None] - idx[None]), n - np.abs(idx[:, None] - idx[None]))
    assert d[sep > 80].min() > 3.0 + 4.0
    assert (t.laps, t.mission) == ((10, 'trackdrive') if name == 'trackdrive' else (1, 'autocross'))


def test_circuit_cones_blue_left_yellow_right_within_5m():
    t = load_track('autocross', scenery=False)
    c = t.centreline
    tang = np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)
    for tag, sign in (('blue', 1), ('yellow', -1)):
        pts = cones(t, tag)
        nearest = np.argmin(np.linalg.norm(pts[:, None] - c[None], axis=2), axis=1)
        rel = pts - c[nearest]
        side = np.sign(tang[nearest, 0] * rel[:, 1] - tang[nearest, 1] * rel[:, 0])
        assert np.all(side == sign)                                       # DS 1.1
        d = np.linalg.norm(pts[:, None] - pts[None], axis=2) + np.eye(len(pts)) * 1e9
        assert d.min(axis=1).max() <= 5.0                                  # DS 1.1: <= 5 m


def test_car_starts_6m_before_line_on_the_circuit():
    t = load_track('autocross', scenery=False)
    mid = (np.array(t.start_line.a) + np.array(t.start_line.b)) / 2
    assert abs(mid[0] - (6.0 + 0.8)) < 0.3 and abs(mid[1]) < 0.3         # D 6.2.3 (front axle)


@pytest.mark.parametrize('name', ['acceleration', 'skidpad', 'autocross', 'droneport'])
def test_scenery_keeps_clear_of_the_track(name):
    t = load_track(name)
    assert len(t.obstacles) > 10
    keep = t.positions if t.centreline is None else np.vstack([t.positions, t.centreline])
    labels = {o.label for o in t.obstacles}
    assert {'barrier', 'tyres', 'spare_cones'} <= labels
    for o in t.obstacles:
        gap = np.min(np.hypot(keep[:, 0] - o.x, keep[:, 1] - o.y)) - o.bound_radius
        assert gap >= 1.0, (o.label, gap)


def test_csv_track_gets_line_from_big_orange():
    t = load_track('droneport', scenery=False)
    assert isinstance(t.start_line, TimingLine) and t.start_line.forward[0] > 0


def test_timing_line_direction():
    line = TimingLine((5.0, -1.5), (5.0, 1.5), (1.0, 0.0))
    assert line.crossed((4.9, 0.0), (5.1, 0.0))
    assert not line.crossed((5.1, 0.0), (4.9, 0.0))       # backwards
    assert not line.crossed((4.9, 3.0), (5.1, 3.0))       # beside the line
