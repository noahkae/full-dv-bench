import math

import pytest

from fsds_lite.vehicle import Vehicle, VehicleParams


def drive(v, seconds, throttle=0.0, brake=0.0, steering=0.0, dt=0.001):
    for _ in range(int(seconds / dt)):
        v.step(dt, throttle, brake, steering)
    return v.state


def test_standstill_is_stable():
    v = Vehicle()
    s = drive(v, 2.0, steering=1.0)
    assert s.vx == 0.0 and abs(s.x) < 1e-9 and abs(s.yaw) < 1e-9


def test_full_throttle_tops_out_at_the_rev_limiter_in_top_gear():
    v = Vehicle()
    s = drive(v, 40.0, throttle=1.0)
    assert s.gear == len(v.p.gear_ratios)
    assert abs(s.vx - v.p.top_speed()) < 0.3
    assert s.rpm <= v.p.max_rpm
    assert abs(s.y) < 1e-6 and abs(s.yaw) < 1e-9


def test_braking_stops_and_does_not_reverse():
    v = Vehicle()
    drive(v, 5.0, throttle=1.0)
    s = drive(v, 10.0, brake=1.0)
    assert s.vx == 0.0
    x_stopped = s.x
    s = drive(v, 2.0, brake=1.0)
    assert s.x == x_stopped


def test_positive_steering_turns_right():
    """FSDS convention: +steering = right = clockwise = negative yaw in ENU."""
    v = Vehicle()
    s = drive(v, 3.0, throttle=0.3, steering=0.5)
    assert s.yaw < -0.1 and s.y < 0.0


def test_low_speed_turn_radius_matches_kinematic_bicycle():
    p = VehicleParams()
    v = Vehicle(p)
    drive(v, 3.0, throttle=0.03, steering=-0.4)        # steady slow left turn
    s = v.state
    delta = math.radians(p.max_steer_deg) * 0.4
    beta = math.atan(p.lr * math.tan(delta) / p.wheelbase)
    expected_r = p.wheelbase / (math.tan(delta) * math.cos(beta))
    assert s.vx < p.kinematic_below
    assert abs(s.speed / s.r - expected_r) / expected_r < 0.02


def test_high_speed_cornering_is_finite_and_bounded():
    v = Vehicle()
    drive(v, 6.0, throttle=1.0)
    s = drive(v, 3.0, throttle=1.0, steering=1.0)
    for val in (s.x, s.y, s.vx, s.vy, s.r):
        assert math.isfinite(val)
    assert abs(s.ay) < 1.6 * v.p.mu * 9.81       # can't exceed the friction limit by much


def test_gearbox_upshifts_in_order_with_a_torque_cut():
    v = Vehicle()
    shifts, prev = [], 1
    for i in range(15000):
        v.step(0.001, 1.0, 0.0, 0.0)
        if v.state.gear != prev:
            shifts.append((prev, v.state.gear, v.state.ax))
            prev = v.state.gear
    assert [(a, b) for a, b, _ in shifts] == [(1, 2), (2, 3), (3, 4)]
    v2 = Vehicle()
    drive(v2, 0.5, throttle=1.0)
    assert v2.state.ax > 3.0                       # pulling hard in 1st
    # during a shift there is no drive: the car only decelerates (drag + rolling)
    v3 = Vehicle()
    while v3.state.gear == 1:
        v3.step(0.001, 1.0, 0.0, 0.0)
    for _ in range(30):            # a few ms in, the spinning tyres have stopped pushing
        v3.step(0.001, 1.0, 0.0, 0.0)
    assert v3.state.shift_timer > 0.0 and v3.state.ax < 0.0


def test_downshifts_to_first_when_stopping():
    v = Vehicle()
    drive(v, 10.0, throttle=1.0)
    s = drive(v, 8.0, brake=1.0)
    assert s.vx == 0.0 and s.gear == 1 and s.rpm == v.p.idle_rpm


def test_engine_braking_slows_the_car_on_lift_off():
    with_eb, without = Vehicle(), Vehicle(VehicleParams(engine_brake_torque=0.0))
    for v in (with_eb, without):
        drive(v, 5.0, throttle=1.0)
        drive(v, 2.0)
    assert with_eb.state.vx < without.state.vx


def test_torque_curve_interpolation_and_validation():
    import pytest
    p = VehicleParams()
    assert p.torque_at(0) == p.torque_curve[0][1]
    assert p.torque_at(1e6) == p.torque_curve[-1][1]
    (r0, t0), (r1, t1) = p.torque_curve[1], p.torque_curve[2]
    assert abs(p.torque_at((r0 + r1) / 2) - (t0 + t1) / 2) < 1e-9
    with pytest.raises(ValueError):
        VehicleParams(gear_ratios=(3.0, 4.0)).validate()
    with pytest.raises(ValueError):
        VehicleParams(upshift_rpm=7000.0, downshift_rpm=6500.0).validate()


def test_full_throttle_launch_spins_the_rear_wheels_and_tc_limits_it():
    raw, tc = Vehicle(), Vehicle(VehicleParams(traction_control=True))
    peak = {}
    for name, v in (('raw', raw), ('tc', tc)):
        p = 0.0
        for _ in range(1500):
            v.step(0.001, 1.0, 0.0, 0.0)
            p = max(p, v.state.slip[2])
        peak[name] = p
    assert peak['raw'] > 1.0                    # proper wheelspin off the line
    assert peak['tc'] < 0.4
    assert tc.state.vx > raw.state.vx           # and TC gets away quicker


def test_hard_braking_locks_fronts_only_on_low_grip():
    for mu, should_lock in ((1.5, False), (0.6, True)):
        v = Vehicle(VehicleParams(mu=mu))
        drive(v, 3.0, throttle=0.6)
        locked = False
        while v.state.vx > 1.0:
            v.step(0.001, 0.0, 1.0, 0.0)
            locked |= v.state.omega[0] == 0.0
        assert locked == should_lock


def test_differentials_split_wheel_speeds_as_designed():
    ratio = {}
    for d in ('open', 'limited_slip', 'locked'):
        v = Vehicle(VehicleParams(diff_type=d))
        drive(v, 1.5, throttle=0.5)
        best = 0.0
        for _ in range(1500):
            v.step(0.001, 1.0, 0.0, -1.0)            # power on in a tight left
            best = max(best, v.state.omega[2] / max(v.state.omega[3], 1e-3))
        ratio[d] = best
    assert ratio['open'] > ratio['limited_slip'] > 1.0
    assert ratio['limited_slip'] < 1.3 * 1.15        # held near the bias
    assert ratio['locked'] == pytest.approx(1.0)


def test_vehicle_json_loads_strings_and_bools(tmp_path):
    import json
    p = tmp_path / 'v.json'
    p.write_text(json.dumps({'diff_type': 'open', 'traction_control': True, 'mu': 1.2}))
    params = VehicleParams.from_json(str(p))
    assert params.diff_type == 'open' and params.traction_control is True and params.mu == 1.2
    p.write_text(json.dumps({'diff_type': 'welded'}))
    with pytest.raises(ValueError):
        VehicleParams.from_json(str(p))
