import numpy as np

from ucr_dv.core.types import BIG_ORANGE, UNKNOWN
from ucr_dv.sensors.models import make_model
from ucr_dv.sensors.rig import from_preset
from ucr_dv.app.pages.rig import forward_range


def _line_of_cones(n=40, spacing=1.0, color=0):
    xs = np.arange(1, n + 1) * spacing
    return np.column_stack([xs, np.full(n, 1.5), np.full(n, color), np.ones(n)]), np.full(n, 0.325)


def test_more_lidar_channels_see_further():
    r4 = forward_range(from_preset('FSDS default 4-ch LiDAR'), {})[1]
    r16 = forward_range(from_preset('Velodyne VLP-16 (IIT Bombay)'), {})[1]
    r40 = forward_range(from_preset('Hesai Pandar40P (KIT 2021)'), {})[1]
    assert r4 < r40 and r16 < r40          # (a 4-ch unit with finer azimuth steps can out-range a VLP-16)


def test_lidar_reports_no_colour_but_knows_big_cones():
    spec = from_preset('Hesai Pandar40P (KIT 2021)')
    m = make_model(spec)
    cones = np.array([[6.0, 1.5, 0, 1], [6.0, -1.5, BIG_ORANGE, 1]], dtype=float)
    det = m.detect(cones, np.array([0.325, 0.505]), np.random.default_rng(0), 0.0)
    cols = sorted(det.cones[:, 2].tolist())
    assert UNKNOWN in cols and BIG_ORANGE in cols


def test_camera_colours_and_range_noise_grow_with_distance():
    spec = from_preset('Mono wide 1080p')
    m = make_model(spec)
    cones, h = _line_of_cones(25, 1.0, color=1)
    rng = np.random.default_rng(1)
    err_near, err_far = [], []
    for _ in range(30):
        d = m.detect(cones, h, rng, 0.0).cones
        d = d[d[:, 3] > 0.3]
        for c in d:
            k = int(np.argmin(np.hypot(cones[:, 0] - c[0], cones[:, 1] - c[1])))
            (err_near if cones[k, 0] < 8 else err_far).append(abs(c[0] - cones[k, 0]))
    assert np.mean(err_far) > np.mean(err_near)


def test_camera_cannot_see_under_its_nose():
    spec = from_preset('Mono wide 1080p')
    spec.mount.z, spec.mount.pitch = 1.2, 0.0
    m = make_model(spec)
    cones = np.array([[1.0, 0.0, 0, 1], [10.0, 0.0, 0, 1]], dtype=float)
    vis = m.visible(cones, np.array([0.325, 0.325]))
    assert not vis[0] and vis[1]


def test_occlusion_hides_cone_behind_cone_for_low_lidar():
    spec = from_preset('Velodyne VLP-16 (IIT Bombay)')
    m = make_model(spec)
    cones = np.array([[5.0, 0.0, 0, 1], [9.0, 0.0, 0, 1]], dtype=float)
    vis = m.visible(cones, np.array([0.325, 0.325]))
    assert vis[0] and not vis[1]
