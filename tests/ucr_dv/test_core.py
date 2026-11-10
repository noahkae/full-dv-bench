import numpy as np
import pytest

from ucr_dv.core import registry
from ucr_dv.core.profile import Profile, list_profiles, load_profile
from ucr_dv.core.types import Pose2D, to_car, to_world
from ucr_dv.tracks import list_tracks, load_track, track_reference


def test_every_stage_has_algorithms():
    for kind in registry.STAGES:
        assert registry.entries(kind), kind


def test_param_coercion_and_clamping():
    e = registry.get('lateral', 'pure_pursuit')
    r = e.resolve({'k_v': '0.5', 'l_min': -100, 'unknown': 3})
    assert r['k_v'] == 0.5 and r['l_min'] == e.param('l_min').lo and 'unknown' not in r


@pytest.mark.parametrize('name', list_profiles())
def test_shipped_profiles_load_and_validate(name):
    p = load_profile(name)
    assert p.validate() == []
    q = Profile.from_dict(p.to_dict())
    assert q.to_dict()['stages'] == p.to_dict()['stages']


def test_frames_roundtrip():
    pose = Pose2D(3.0, -2.0, 0.7)
    pts = np.random.default_rng(0).normal(size=(20, 2)) * 10
    assert np.allclose(to_world(to_car(pts, pose), pose), pts)


@pytest.mark.parametrize('name', list_tracks())
def test_tracks_start_at_origin_facing_x(name):
    t = load_track(name, scenery=False)
    ref = track_reference(t)
    assert np.hypot(*ref.xy[0]) < 3.0
    assert (ref.xy[5] - ref.xy[0])[0] > 0          # reference runs forward (+x) from the start
    assert t.start_line is not None
