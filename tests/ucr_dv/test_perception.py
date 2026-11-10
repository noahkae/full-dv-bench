import math

import numpy as np
import pytest

from ucr_dv.core.registry import get
from ucr_dv.core.types import BIG_ORANGE, PointCloud
from ucr_dv.perception.fusion import _merge
from ucr_dv.perception.metrics import PerceptionStats
from ucr_dv.sensors.rig import from_preset, to_lite_lidar


def _scan(cone_xy, heights=None):
    from fsds_lite.sensors import Lidar
    spec = from_preset('Hesai Pandar40P (KIT 2021)')
    lid = Lidar(to_lite_lidar(spec), np.random.default_rng(0))
    xy = np.asarray(cone_xy, float)
    h = np.full(len(xy), 0.325) if heights is None else np.asarray(heights)
    pts = lid.scan(0.0, 0.0, 0.0, xy, np.where(h > 0.4, 0.1425, 0.114), h)
    m = spec.mount
    return PointCloud('lidar', 0.0, pts.astype(float) + np.array([m.x, m.y, m.z]), (m.x, m.y, m.z))


@pytest.mark.parametrize('algo', ['lidar_grid', 'lidar_dbscan'])
def test_lidar_detectors_find_raycast_cones(algo):
    truth = np.array([[5.0, 1.6], [5.0, -1.6], [9.0, 1.6], [9.0, -1.6], [13.0, 1.7]])
    cloud = _scan(truth)
    det = get('lidar', algo).create({'min_cluster_points': 3})
    cones, _ = det.detect(cloud)
    stats = PerceptionStats()
    stats.add(cones, np.column_stack([truth, np.zeros(len(truth)), np.ones(len(truth))]))
    assert stats.recall == 1.0 and stats.precision == 1.0 and stats.rmse < 0.12


def test_lidar_marks_tall_cones_big_orange():
    cloud = _scan(np.array([[6.0, 1.5], [6.0, -1.5]]), heights=[0.505, 0.325])
    cones, _ = get('lidar', 'lidar_grid').create({}).detect(cloud)
    big = cones[cones[:, 2] == BIG_ORANGE]
    assert len(big) == 1 and big[0, 1] > 0


def test_fusion_merges_across_sensors_not_within():
    a = np.array([[5.0, 1.5, 4, 1.0], [5.3, 1.6, 3, 1.0]])         # one sensor, two objects
    b = np.array([[5.1, 1.55, 1, 1.0]])                              # other sensor
    out = _merge(np.vstack([a, b]), None, 0.6, False, np.array([0, 0, 1]))
    assert len(out) == 2


def test_camera_blob_detector_on_rendered_image():
    pytest.importorskip('moderngl')
    from ucr_dv.backends.lite import LiteBackend
    from ucr_dv.core.profile import Profile
    from ucr_dv.sensors.rig import Rig
    from ucr_dv.tracks import load_track
    from ucr_dv.vehicle import VehicleSpec, load_vehicle
    p = Profile('t', rig=Rig('cam', [from_preset('Mono wide 1080p', 'cam')]))
    b = LiteBackend(physics_hz=250)
    try:
        b.load(load_track('acceleration', scenery=False), p, VehicleSpec(load_vehicle('ucr_ev')), 0)
    except Exception as e:
        pytest.skip('no GPU context: {}'.format(e))
    if b.renderer is None:
        b.close()
        pytest.skip('camera renderer unavailable: {}'.format(b.camera_error))
    frame = None
    for _ in range(6):
        obs = b.advance(0.05)
        if 'cam' in obs.frames:
            frame = obs.frames['cam']
    b.close()
    assert frame is not None
    cones, dbg = get('camera', 'camera_color_blob').create({}).detect(frame)
    assert len(cones) >= 4
    # acceleration lane: cones at |y| = 1.5 m
    near = cones[np.hypot(cones[:, 0], cones[:, 1]) < 15]
    assert np.all(np.abs(np.abs(near[:, 1]) - 1.5) < 0.6)
