"""Lap timing, finished signal, trackside objects, wheel states, IMU noise."""
import math

import numpy as np
import pytest

from fsds_lite.sensors import Lidar, LidarConfig, load_settings
from fsds_lite.simulator import Controls, ImuSample, Referee, Simulator, WheelStatesSample
from fsds_lite.track import load_track, Obstacle, Track, TimingLine
from conftest import SETTINGS


def make_sim(name, scenery=False, **kw):
    s = load_settings(SETTINGS)
    s.imu.noise = kw.pop('imu_noise', False)
    return Simulator(load_track(name, scenery=scenery), s, seed=0)


def test_acceleration_run_is_timed_start_to_finish():
    sim = make_sim('acceleration')
    sim.enable_api_control(True)
    sim.set_api_controls(Controls(throttle=1.0))
    while not sim.referee.mission_complete and sim.sim_time < 15.0:
        sim.step(10)
    sim.set_api_controls(Controls(brake=1.0))       # D 5.2.4: stop in the stop area
    sim.step(4000)
    ref = sim.referee.summary()
    assert ref['complete'] and len(ref['laps']) == 1
    assert 4.0 < ref['laps'][0] < 8.0 and ref['doo'] == 0


def test_skidpad_run_time_is_mean_of_timed_laps():
    line = TimingLine((0.0, -9.0), (0.0, 9.0), (1.0, 0.0))
    t = Track('sp', [], [], line, line, mission='skidpad', laps=4)
    ref = Referee(t)
    times = [1.0, 6.0, 11.5, 16.5, 21.5]             # five forward crossings -> four laps
    for k, tc in enumerate(times):
        ref.update(tc - 0.01, (-0.1, 0.0))
        ref.update(tc, (0.1, 0.0))
        if k < len(times) - 1:
            ref.update(tc + 2.0, (-5.0, -5.0))         # off around the circle
            ref._prev = (-0.1, 0.0)
    assert ref.laps == pytest.approx([5.0, 5.5, 5.0, 5.0])
    assert ref.run_time() == pytest.approx((5.5 + 5.0) / 2)


def test_trackdrive_counts_laps_and_finished_signal_hands_back_control():
    sim = make_sim('trackdrive')
    ref = sim.referee
    line = sim.track.start_line
    a, b = np.array(line.a), np.array(line.b)
    mid, fwd = (a + b) / 2, np.array(line.forward)
    for lap in range(3):
        ref.update(lap * 30.0, tuple(mid - fwd * 0.2))
        ref.update(lap * 30.0 + 0.01, tuple(mid + fwd * 0.2))
    assert len(ref.laps) == 2 and not ref.mission_complete
    sim.enable_api_control(True)
    sim.finish()
    assert ref.finished and not sim.api_control and sim.manual.brake == 1.0


def test_driving_into_a_barrier_stops_the_car_and_counts_a_crash():
    t = load_track('acceleration', scenery=False)
    t.set_obstacles([Obstacle('box', 25.0, 0.0, 0.9, yaw=math.pi / 2, half_x=2.0, half_y=0.3,
                              label='barrier')])
    s = load_settings(SETTINGS)
    sim = Simulator(t, s, seed=0)
    sim.enable_api_control(True)
    sim.set_api_controls(Controls(throttle=0.6))
    sim.step(6000)
    st = sim.vehicle.state
    assert sim.referee.crashes == 1 and st.vx == 0.0 and st.x < 25.0 - 0.3 - 0.9 + 0.05


def test_spare_cones_get_knocked_but_are_not_penalties():
    t = load_track('acceleration', scenery=False)
    t.set_obstacles([Obstacle('cone', 15.0, 0.0, 0.5, radius=0.114, label='spare_cones', color='orange')])
    sim = Simulator(t, load_settings(SETTINGS), seed=0)
    sim.enable_api_control(True)
    sim.set_api_controls(Controls(throttle=0.4))
    sim.step(5000)
    assert sim.spare_hit.all() and sim.referee.doo == 0 and sim.referee.crashes == 0


def lidar(**kw):
    cfg = dict(name='L', z=0.55, n_lasers=8, points_per_scan=4096, vfov_upper=5, vfov_lower=-15)
    cfg.update(kw)
    return Lidar(LidarConfig(**cfg))


def test_lidar_hits_boxes_and_cylinders_where_they_are():
    none = (np.zeros((0, 2)), np.zeros(0), np.zeros(0))
    box = np.array([[10.0, 0.0, 0.0, 0.3, 2.0, 0.9]])            # wall face at x = 9.7
    pts = lidar().scan(0, 0, 0, *none, boxes=box)
    wall = pts[pts[:, 2] > -0.5]
    assert len(wall) and np.allclose(wall[:, 0], 9.7, atol=1e-3)
    assert np.all(np.abs(wall[:, 1]) <= 2.0 + 1e-6) and np.all(wall[:, 2] <= 0.9 - 0.55 + 1e-6)
    cyl = np.array([[8.0, 3.0, 0.35, 0.8]])
    pts = lidar().scan(0, 0, 0, *none, cylinders=cyl)
    tyre = pts[pts[:, 2] > -0.5]
    assert len(tyre)
    assert np.all(np.abs(np.hypot(tyre[:, 0] - 8.0, tyre[:, 1] - 3.0) - 0.35) < 1e-3)


def test_lidar_object_culling_matches_brute_force():
    t = load_track('autocross')
    for pose in ((0, 0, 0), (20, 15, 1.0), (-5, 30, -2.0)):
        fast, slow = lidar(hfov_start=-180, hfov_end=180), lidar(hfov_start=-180, hfov_end=180)
        slow.level_mount = False
        args = (t.positions, t.radii, t.heights)
        a = fast.scan(*pose, *args, 0.02, -0.01, boxes=t.boxes, cylinders=t.cylinders)
        b = slow.scan(*pose, *args, 0.02, -0.01, boxes=t.boxes, cylinders=t.cylinders)
        assert a.shape == b.shape and np.allclose(a, b, atol=1e-5)


def test_wheel_states_track_road_speed():
    sim = make_sim('acceleration')
    got = []
    sim.add_listener(lambda s: got.append(s) if isinstance(s, WheelStatesSample) else None)
    sim.enable_api_control(True)
    sim.set_api_controls(Controls(throttle=0.3))
    sim.step(4000)
    ws = got[-1]
    road_rpm = sim.vehicle.state.vx / sim.vehicle.p.wheel_radius * 60 / (2 * math.pi)
    assert abs(ws.rpm[0] - road_rpm) / road_rpm < 0.02          # free-rolling front
    assert 99 <= len(got) // 4 * 1 or len(got) >= 390             # 100 Hz


def test_imu_noise_matches_airsim_model():
    sim = make_sim('acceleration', imu_noise=True)
    got = []
    sim.add_listener(lambda s: got.append(s) if isinstance(s, ImuSample) else None)
    sim.step(10000)                                               # 40 s parked, 250 Hz
    gyro = np.array([s.angular_velocity for s in got])
    cfg = sim.sensors.imu
    expected = cfg.gyro_arw / math.sqrt(1.0 / cfg.rate_hz)
    assert abs(gyro[:, 2].std() / expected - 1.0) < 0.15
    acc = np.array([s.linear_acceleration for s in got])
    assert abs(acc[:, 2].mean() - 9.81) < 0.01
