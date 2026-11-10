"""Body attitude and load transfer."""
import math

from fsds_lite.simulator import Controls, ImuSample
from fsds_lite.vehicle import Vehicle, VehicleParams


def drive(v, seconds, throttle=0.0, brake=0.0, steering=0.0, dt=0.001):
    for _ in range(int(seconds / dt)):
        v.step(dt, throttle, brake, steering)
    return v.state


def test_braking_pitches_nose_down_and_loads_front():
    v = Vehicle()
    drive(v, 6.0, throttle=1.0)
    s = drive(v, 0.4, brake=1.0)
    assert s.pitch > math.radians(0.3)              # +pitch = nose down
    fl, fr, rl, rr = s.fz
    assert fl + fr > rl + rr


def test_acceleration_squats():
    v = Vehicle()
    s = drive(v, 0.5, throttle=1.0)
    assert s.pitch < 0.0


def test_left_turn_rolls_right_side_down_and_loads_outside_wheels():
    v = Vehicle()
    drive(v, 4.0, throttle=0.5)
    s = drive(v, 2.0, throttle=0.5, steering=-0.3)  # -ve steering = left
    assert s.ay > 2.0
    assert s.roll > 0.0                             # +roll = right side down
    fl, fr, rl, rr = s.fz
    assert fr > fl and rr > rl


def test_steady_roll_matches_spring_balance():
    p = VehicleParams()
    v = Vehicle(p)
    drive(v, 4.0, throttle=0.5)
    s = drive(v, 4.0, throttle=0.5, steering=-0.2)
    expected = p.mass * s.ay * p.cg_height / p.roll_stiffness
    assert abs(s.roll - expected) < 0.1 * abs(expected) + 1e-4


def test_load_sensitivity_costs_cornering_grip():
    def max_lat(sens):
        v = Vehicle(VehicleParams(load_sensitivity=sens))
        drive(v, 5.0, throttle=1.0)
        best = 0.0
        for _ in range(3000):
            v.step(0.001, 0.3, 0.0, -1.0)
            best = max(best, v.state.ay)
        return best
    assert max_lat(0.3) < max_lat(0.0)


def test_attitude_settles_at_rest():
    v = Vehicle()
    drive(v, 3.0, throttle=1.0)
    s = drive(v, 8.0, brake=1.0)
    assert s.vx == 0.0 and abs(s.pitch) < 1e-3 and abs(s.roll) < 1e-3


def test_imu_feels_tilted_gravity_under_braking(sim):
    got = []
    sim.add_listener(lambda s: got.append(s) if isinstance(s, ImuSample) else None)
    sim.enable_api_control(True)
    sim.set_api_controls(Controls(throttle=1.0))
    sim.step(4000)
    sim.set_api_controls(Controls(brake=1.0))
    sim.step(400)
    ax, ay, az = got[-1].linear_acceleration
    assert ax < -5.0 and az < 9.81 and abs(math.sqrt(ax * ax + az * az) - math.hypot(sim.vehicle.state.ax, 9.81)) < 0.5
