"""Headless rendering. Skipped where moderngl or a GL driver is unavailable."""
import math
import time

import numpy as np
import pytest

from fsds_lite.simulator import BodyPose, CameraSample
from fsds_lite.sensors import CameraConfig
from fsds_lite.track import Track
from fsds_lite.vehicle import VehicleParams

moderngl = pytest.importorskip('moderngl')


@pytest.fixture(scope='module')
def ctx():
    try:
        c = moderngl.create_standalone_context(require=330)
    except Exception as e:
        pytest.skip('no OpenGL 3.3: {}'.format(e))
    yield c
    c.release()


def render_one(ctx, track, cam, pose=BodyPose(0, 0, 0)):
    from fsds_lite.render.renderer import Renderer, camera_projection, camera_view
    r = Renderer(ctx, track, VehicleParams())
    fbo = ctx.framebuffer(ctx.renderbuffer((cam.width, cam.height)),
                          ctx.depth_renderbuffer((cam.width, cam.height)))
    view, eye, _ = camera_view(pose, cam)
    ctx.viewport = (0, 0, cam.width, cam.height)
    car = dict(x=pose.x, y=pose.y, yaw=pose.yaw, roll=0.0, pitch=0.0, delta=0.0, wheel_spin=0.0)
    r.render(view, camera_projection(cam), eye, car, fbo, draw_car=False)
    img = np.frombuffer(fbo.read(components=3, alignment=1), np.uint8).reshape(cam.height, cam.width, 3)
    return img[::-1]                                   # RGB, top row first


def test_cone_projects_where_the_pinhole_model_says(ctx):
    # a blue cone 3 m ahead and 0.5 m to the left of a level camera at 0.5 m
    cam = CameraConfig('c', x=0.0, z=0.5, width=320, height=240, fov_deg=90)
    track = Track('t', [(3.0, 0.5)], ['blue'])
    img = render_one(ctx, track, cam).astype(int)
    blue = (img[:, :, 2] > 150) & (img[:, :, 0] < 80)
    ys, xs = np.nonzero(blue)
    assert len(xs) > 20
    fx, fy, cx, cy = cam.intrinsics()
    u_expected = cx - fx * 0.5 / 3.0                  # +y (left) maps to smaller u
    assert abs(np.median(xs) - u_expected) < 4
    assert ys.max() > cy                              # cone base is below the horizon


def test_camera_renderer_thread_emits_bgr_samples(sim, sensors):
    from fsds_lite.render.camera_sensor import CameraRenderer
    got = []
    sim.add_listener(lambda s: got.append(s) if isinstance(s, CameraSample) else None)
    cams = [CameraConfig('cam1', z=0.8, width=160, height=120)]
    cr = CameraRenderer(sim, cams, samples=0)
    try:
        cr.start()
    except RuntimeError as e:
        pytest.skip(str(e))
    try:
        sim.step(200)                                 # 20 Hz -> a few requests
        deadline = time.time() + 5
        while not got and time.time() < deadline:
            time.sleep(0.02)
    finally:
        cr.stop()
    s = got[-1]
    assert s.image.shape == (120, 160, 3) and s.image.dtype == np.uint8
    # GPU-side flip + BGR: the top row must be sky, and in BGR order blue is index 0.
    # (Grey asphalt has B - R of only a few counts, so this also catches an upside-down image.)
    top = s.image[2, 80].astype(int)
    assert top[0] - top[2] > 40


def test_depth_camera_measures_distance_to_a_wall(sim):
    from fsds_lite.render.camera_sensor import CameraRenderer
    from fsds_lite.track import Obstacle
    t = Track('wall', [], [])
    t.set_obstacles([Obstacle('box', 12.0, 0.0, 3.0, half_x=0.2, half_y=30.0, label='barrier')])
    cams = [CameraConfig('d', z=0.8, width=96, height=72, image_type=1, depth_noise=False),
            CameraConfig('p', z=0.8, width=96, height=72, image_type=2, depth_noise=False)]
    sim.sensors.cameras = cams                     # the sim schedules requests per camera
    sim2 = type(sim)(t, sim.sensors, seed=0)
    got = []
    sim2.add_listener(lambda s: got.append(s) if isinstance(s, CameraSample) else None)
    cr = CameraRenderer(sim2, cams, samples=0)
    try:
        cr.start()
    except RuntimeError as e:
        pytest.skip(str(e))
    try:
        sim2.step(150)
        deadline = time.time() + 5
        while len({s.name for s in got}) < 2 and time.time() < deadline:
            time.sleep(0.02)
    finally:
        cr.stop()
    planar = [s for s in got if s.name == 'd'][-1].image
    persp = [s for s in got if s.name == 'p'][-1].image
    # the wall face is 11.8 m ahead of the car centre; the camera sits on the centre
    assert planar.dtype == np.float32 and abs(planar[36, 48] - 11.8) < 0.05
    assert abs(planar[36, 5] - 11.8) < 0.05                       # planar: same across the row
    fx, fy, cx, cy = cams[1].intrinsics()
    ray = math.sqrt(1 + ((5 + 0.5 - cx) / fx) ** 2 + ((36 + 0.5 - cy) / fy) ** 2)
    assert abs(persp[36, 5] - 11.8 * ray) < 0.1                   # perspective: along the ray
    assert np.isinf(planar[0, 48]) or planar[0, 48] > 11.0        # sky (or wall top) above


def test_noisify_depth_rounds_up_by_range():
    from fsds_lite.render.camera_sensor import noisify_depth
    flat = np.full((20, 20), 3.4, dtype=np.float32)
    assert np.allclose(noisify_depth(flat), 4.0)                  # 1-4 m steps -> 4 m
    far = np.full((20, 20), 23.0, dtype=np.float32)
    assert np.allclose(noisify_depth(far), 30.0)                  # 10 m steps beyond 10 m
    sky = flat.copy()
    sky[:5] = np.inf
    assert np.isinf(noisify_depth(sky)[0]).all()
