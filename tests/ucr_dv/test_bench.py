import math

import numpy as np
import pytest

from ucr_dv.bench.session import prepare, run_episode
from ucr_dv.core.profile import load_profile
from ucr_dv.core.types import ControlCommand


@pytest.mark.parametrize('track', ['acceleration', 'autocross', 'timeless_map1', 'skidpad'])
def test_baseline_completes_on_timeless(track):
    res, _ = run_episode('baseline', track, bench='timeless', seed=0)
    assert res['complete'], res['summary']
    assert res['doo'] <= 5 and res['off_course'] == 0


def test_timeless_is_deterministic():
    a, ra = run_episode('baseline', 'timeless_map1', seed=3)
    b, rb = run_episode('baseline', 'timeless_map1', seed=3)
    assert a['run_time'] == b['run_time']
    assert np.allclose(ra.log.arrays()['x'], rb.log.arrays()['x'])


@pytest.mark.parametrize('bench', ['timeless', 'lite'])
def test_positive_steering_turns_left_on_every_bench(bench):
    runner, track, stack, b = prepare(load_profile('baseline'), 'acceleration', bench, 0,
                                      backend_kw={'cameras': False} if bench == 'lite' else None)
    b.reset(0)
    yaw0 = b.car_state()['yaw']
    for _ in range(40):
        b.apply(ControlCommand(steering=0.2, throttle=0.4))
        b.advance(0.05)
    st = b.car_state()
    assert st['yaw'] - yaw0 > 0.2 and st['y'] > 0.5
    b.close()


def test_baseline_completes_acceleration_on_3d():
    res, _ = run_episode('baseline', 'acceleration', bench='lite', seed=0, backend_kw={'cameras': False})
    assert res['complete'], res['summary']
    assert res['doo'] == 0


def test_perception_eval_reports_every_sensor():
    from ucr_dv.bench.perception_eval import evaluate
    rep, frames, _ = evaluate(load_profile('lidar_camera'), 'autocross', bench='timeless', distance=40)
    assert {'lidar', 'cam', 'fused'} <= set(rep['sensors'])
    assert rep['sensors']['fused']['recall'] > 0.3
