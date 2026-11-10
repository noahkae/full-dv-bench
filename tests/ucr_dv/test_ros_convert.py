import importlib.util
import os
from types import SimpleNamespace as NS

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location('convert', os.path.join(ROOT, 'ros', 'ucr_dv_ros', 'ucr_dv_ros',
                                                                      'convert.py'))
convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert)


def test_pointcloud2_roundtrip_with_padding():
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    rec = np.zeros(2, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('pad', '<f4')])
    rec['x'], rec['y'], rec['z'] = pts.T
    msg = NS(width=2, height=1, is_bigendian=False, point_step=16, data=rec.tobytes(),
             fields=[NS(name='x', offset=0, datatype=7), NS(name='y', offset=4, datatype=7),
                     NS(name='z', offset=8, datatype=7)])
    assert np.allclose(convert.cloud_to_xyz(msg), pts)


def test_image_bgr_and_depth():
    img = np.arange(4 * 3 * 3, dtype=np.uint8).reshape(4, 3, 3)
    msg = NS(encoding='bgr8', height=4, width=3, step=9, data=img.tobytes())
    assert np.array_equal(convert.image_to_bgr(msg), img)
    d = np.random.default_rng(0).random((2, 5)).astype(np.float32)
    msg = NS(encoding='32FC1', height=2, width=5, step=20, data=d.tobytes())
    assert np.allclose(convert.image_to_bgr(msg), d)


def test_sensor_to_car_yaw():
    out = convert.sensor_to_car(np.array([[1.0, 0.0, 0.0]]), (1.0, 0.0, 0.5, 0.0, 0.0, np.pi / 2))
    assert np.allclose(out, [[1.0, 1.0, 0.5]])
