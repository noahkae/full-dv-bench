import math

import numpy as np

from fsds_lite.sensors import Lidar, LidarConfig
from fsds_lite.track import load_track


def make_lidar(**kw):
    cfg = dict(name='L', x=0.0, y=0.0, z=0.55, n_lasers=4, points_per_scan=4096,
               vfov_upper=5, vfov_lower=-5, hfov_start=-90, hfov_end=90)
    cfg.update(kw)
    return Lidar(LidarConfig(**cfg))


def no_cones():
    return np.zeros((0, 2)), np.zeros(0), np.zeros(0)


def test_ground_ring_distance():
    lidar = make_lidar()
    pts = lidar.scan(0, 0, 0, *no_cones())
    # only the two downward lasers hit; lowest (-5 deg) hits at 0.55 / tan(5 deg)
    rng = np.hypot(pts[:, 0], pts[:, 1])
    assert np.isclose(rng.min(), 0.55 / math.tan(math.radians(5)), rtol=1e-3)
    assert np.allclose(pts[:, 2][rng < 7.0], -0.55, atol=1e-4)
    assert np.all(pts[:, 0] >= -1e-6)       # 180 deg FOV: nothing behind


def test_cone_ahead_is_hit_in_sensor_frame():
    lidar = make_lidar(vfov_upper=0, vfov_lower=-15, n_lasers=16)
    xy = np.array([[5.0, 0.0]])
    pts = lidar.scan(0, 0, 0, xy, np.array([0.114]), np.array([0.325]))
    above_ground = pts[pts[:, 2] > -0.55 + 1e-3]
    assert len(above_ground) > 0
    assert np.all(np.abs(above_ground[:, 1]) < 0.12)
    assert np.all((above_ground[:, 0] > 4.88) & (above_ground[:, 0] < 5.0))
    assert np.all(above_ground[:, 2] <= -0.55 + 0.325 + 1e-6)


def test_hfov_sign_convention():
    """HorizontalFOVStart/End are clockwise-positive: a [0, 90] FOV sees the RIGHT side."""
    lidar = make_lidar(hfov_start=0, hfov_end=90, vfov_upper=-5, vfov_lower=-5, n_lasers=1)
    pts = lidar.scan(0, 0, 0, *no_cones())
    assert len(pts) and np.all(pts[:, 1] <= 1e-6)


def test_world_pose_is_applied():
    lidar = make_lidar(vfov_upper=0, vfov_lower=-15, n_lasers=16)
    cone = np.array([[10.0, 20.0]])
    # car at (10, 15) facing +y: cone is 5 m dead ahead in the sensor frame
    pts = lidar.scan(10.0, 15.0, math.pi / 2, cone, np.array([0.114]), np.array([0.325]))
    above = pts[pts[:, 2] > -0.55 + 1e-3]
    assert len(above) and np.all(np.abs(above[:, 1]) < 0.12)
    assert np.all(np.abs(above[:, 0] - 4.94) < 0.07)


def test_bundled_droneport_track():
    t = load_track('droneport')
    assert t.colors.count('blue') == 67 and t.colors.count('yellow') == 63
    assert t.colors.count('big_orange') == 2


def test_column_culling_matches_brute_force():
    t = load_track('droneport')
    for kw in ({}, dict(hfov_start=-180, hfov_end=180, n_lasers=8, vfov_lower=-15), dict(yaw=30)):
        fast, slow = make_lidar(**kw), make_lidar(**kw)
        slow.level_mount = False
        for x, y, yaw in ((0, 0, 0), (-3, -6, 2.0), (-20, 5, -2.8), (5, 1, 0.7)):
            a = fast.scan(x, y, yaw, t.positions, t.radii, t.heights)
            b = slow.scan(x, y, yaw, t.positions, t.radii, t.heights)
            assert a.shape == b.shape and np.allclose(a, b, atol=1e-5), (kw, x, y, yaw)


def test_column_culling_matches_brute_force_with_body_tilt():
    t = load_track('droneport')
    fast, slow = make_lidar(vfov_lower=-15, n_lasers=8), make_lidar(vfov_lower=-15, n_lasers=8)
    slow.level_mount = False
    for x, y, yaw, roll, pitch in ((0, 0, 0, 0.03, 0.02), (-3, -6, 2.0, -0.04, 0.03), (5, 1, 0.7, 0.0, -0.05)):
        a = fast.scan(x, y, yaw, t.positions, t.radii, t.heights, roll, pitch)
        b = slow.scan(x, y, yaw, t.positions, t.radii, t.heights, roll, pitch)
        assert a.shape == b.shape and np.allclose(a, b, atol=1e-5), (x, y, yaw, roll, pitch)


def test_nose_down_pitch_pulls_ground_ring_closer():
    level = make_lidar().scan(0, 0, 0, *no_cones())
    pitched = make_lidar().scan(0, 0, 0, *no_cones(), 0.0, math.radians(1.0))
    assert np.hypot(pitched[:, 0], pitched[:, 1]).min() < np.hypot(level[:, 0], level[:, 1]).min() - 0.5
