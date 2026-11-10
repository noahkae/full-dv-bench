from fsds_lite.simulator import Controls, ImuSample, LidarSample


def test_api_controls_ignored_until_enabled(sim):
    assert not sim.set_api_controls(Controls(throttle=1.0))
    sim.step(1000)
    assert sim.vehicle.state.vx == 0.0
    sim.enable_api_control(True)
    assert sim.set_api_controls(Controls(throttle=1.0))
    sim.step(1000)
    assert sim.vehicle.state.vx > 1.0


def test_reset_drops_api_control_and_returns_home(sim):
    sim.enable_api_control(True)
    sim.set_api_controls(Controls(throttle=1.0))
    sim.step(2000)
    sim.reset()
    s = sim.vehicle.state
    assert not sim.api_control and s.x == 0.0 and s.vx == 0.0


def test_sensor_rates(sim):
    counts = {}
    sim.add_listener(lambda s: counts.__setitem__(type(s).__name__ + getattr(s, 'name', ''),
                                                  counts.get(type(s).__name__ + getattr(s, 'name', ''), 0) + 1))
    sim.step(1000)                         # 1 s
    assert 9 <= counts['LidarSampleLidar1'] <= 11      # 10 Hz in settings.json
    assert 4 <= counts['LidarSampleLidar2'] <= 6       # 5 Hz
    assert 240 <= counts['ImuSampleImu'] <= 251
    assert 9 <= counts['GpsSampleGps'] <= 11


def test_lidar_sees_acceleration_track_cones(sim):
    got = []
    sim.add_listener(lambda s: got.append(s) if isinstance(s, LidarSample) and s.name == 'Lidar1' else None)
    sim.step(150)
    pts = got[-1].points
    cones = pts[pts[:, 2] > -0.55 + 0.03]          # Lidar1 sits 0.55 m up
    assert len(cones) > 20
    assert (cones[:, 1] > 0).any() and (cones[:, 1] < 0).any()   # both sides of the track


def test_driving_off_track_knocks_cones(sim):
    sim.teleport(2.0, 1.5, 0.0)            # lined up on the left cone row
    sim.enable_api_control(True)
    sim.set_api_controls(Controls(throttle=0.5))
    sim.step(3000)
    assert sim.cones_hit.sum() >= 1
    sim.reset()
    assert sim.cones_hit.sum() == 0


def test_imu_sees_gravity(sim):
    got = []
    sim.add_listener(lambda s: got.append(s) if isinstance(s, ImuSample) else None)
    sim.step(20)
    assert abs(got[-1].linear_acceleration[2] - 9.81) < 1e-6
