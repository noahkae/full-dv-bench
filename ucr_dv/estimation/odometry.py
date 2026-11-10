"""
Odometry: ego measurements -> pose in the world (spawn) frame.

    ground_truth    the simulator's pose. Marks the run as using GT (not
                    allowed past the promotion gate) - for isolating planning
                    and control from estimation problems.
    dead_reckoning  integrate speed (GSS if fitted, else wheel speed) and the
                    gyro's yaw rate. Drifts, but needs nothing else.
    ekf             dead reckoning + GNSS position updates (state x, y, yaw, v).
"""
from __future__ import annotations

import math

import numpy as np

from ..core.registry import Component, P, register
from ..core.types import Pose2D, wrap_angle


def _speed_and_rate(ego, last_v=0.0):
    if ego.gss_velocity is not None:
        vx, vy = ego.gss_velocity
    elif ego.wheel_speed is not None:
        vx, vy = ego.wheel_speed, 0.0
    else:
        vx, vy = last_v, 0.0
    r = ego.yaw_rate if ego.yaw_rate is not None else 0.0
    return vx, vy, r


class _Odom(Component):
    uses_gt = False

    def __init__(self, params, **context):
        super(_Odom, self).__init__(params, **context)
        self.reset()

    def reset(self):
        self.pose = Pose2D()
        self.speed = 0.0
        self.distance = 0.0
        self._t = None

    def correct(self, pose):
        """Accept a pose fix from the map (SLAM loop)."""
        if not self.uses_gt:
            self.pose = pose


@register('odometry', 'ground_truth', label='Ground truth (simulator)',
          params=[], help='Perfect pose from the simulator. Runs using it cannot pass the promotion gate.')
class GroundTruthOdometry(_Odom):
    uses_gt = True

    def update(self, obs, dt):
        tr = obs.truth
        if tr is not None:
            self.distance += abs(tr.speed) * dt
            self.pose = Pose2D(tr.pose.x, tr.pose.y, tr.pose.yaw)
            self.speed = tr.speed
        return self.pose


@register('odometry', 'dead_reckoning', label='Dead reckoning (speed + gyro)',
          params=[P('speed_filter', 0.3, 0.0, 0.95, 0.05, '', 'low-pass on the measured speed (0 = none)')])
class DeadReckoning(_Odom):
    """Integrates body velocity and yaw rate (midpoint rule)."""

    def update(self, obs, dt):
        vx, vy, r = _speed_and_rate(obs.ego, self.speed)
        a = self.params['speed_filter']
        self.speed = a * self.speed + (1 - a) * vx
        p = self.pose
        yaw_mid = p.yaw + 0.5 * r * dt
        c, s = math.cos(yaw_mid), math.sin(yaw_mid)
        self.pose = Pose2D(p.x + (self.speed * c - vy * s) * dt, p.y + (self.speed * s + vy * c) * dt,
                           wrap_angle(p.yaw + r * dt))
        self.distance += abs(self.speed) * dt
        return self.pose


@register('odometry', 'ekf', label='EKF (dead reckoning + GNSS)',
          params=[P('q_pos', 0.05, 0.001, 1.0, 0.005, 'm/sqrt(s)', 'process noise, position'),
                  P('q_yaw', 0.02, 0.001, 0.5, 0.005, 'rad/sqrt(s)', 'process noise, heading'),
                  P('q_v', 0.5, 0.01, 5.0, 0.05, 'm/s/sqrt(s)', 'process noise, speed'),
                  P('r_gnss', 0.05, 0.005, 3.0, 0.005, 'm', 'GNSS position std'),
                  P('r_speed', 0.1, 0.01, 2.0, 0.01, 'm/s', 'speed measurement std')],
          refs=['IIT Bombay 2024 (EKF)', 'AMZ 2019 (velocity estimation)'])
class EkfOdometry(_Odom):
    """State [x, y, yaw, v]; predict with gyro; update with speed and GNSS."""

    def reset(self):
        super(EkfOdometry, self).reset()
        self.x = np.zeros(4)
        self.P = np.diag([0.01, 0.01, 0.001, 0.1])

    def correct(self, pose):
        self.x[0], self.x[1], self.x[2] = pose.x, pose.y, pose.yaw
        self.pose = pose

    def update(self, obs, dt):
        p = self.params
        ego = obs.ego
        x, P = self.x, self.P
        r = ego.yaw_rate if ego.yaw_rate is not None else 0.0
        vy = ego.gss_velocity[1] if ego.gss_velocity is not None else 0.0
        yaw, v = x[2], x[3]
        c, s = math.cos(yaw), math.sin(yaw)
        x = np.array([x[0] + (v * c - vy * s) * dt, x[1] + (v * s + vy * c) * dt, wrap_angle(yaw + r * dt), v])
        F = np.eye(4)
        F[0, 2], F[0, 3] = -(v * s + vy * c) * dt, c * dt
        F[1, 2], F[1, 3] = (v * c - vy * s) * dt, s * dt
        Q = np.diag([p['q_pos'] ** 2 * dt, p['q_pos'] ** 2 * dt, p['q_yaw'] ** 2 * dt, p['q_v'] ** 2 * dt])
        P = F @ P @ F.T + Q
        # speed
        meas_v = ego.gss_velocity[0] if ego.gss_velocity is not None else ego.wheel_speed
        if meas_v is not None:
            H = np.array([[0.0, 0.0, 0.0, 1.0]])
            S = H @ P @ H.T + p['r_speed'] ** 2
            K = P @ H.T / S
            x = x + (K * (meas_v - x[3])).ravel()
            P = (np.eye(4) - K @ H) @ P
        # GNSS
        if ego.gnss_xy is not None:
            H = np.zeros((2, 4))
            H[0, 0] = H[1, 1] = 1.0
            S = H @ P @ H.T + np.eye(2) * p['r_gnss'] ** 2
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (np.asarray(ego.gnss_xy) - x[:2])
            P = (np.eye(4) - K @ H) @ P
        x[2] = wrap_angle(x[2])
        self.x, self.P = x, P
        self.speed = float(x[3])
        self.distance += abs(self.speed) * dt
        self.pose = Pose2D(float(x[0]), float(x[1]), float(x[2]))
        return self.pose
