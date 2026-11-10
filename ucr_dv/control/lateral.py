"""
Lateral control: path (car frame) + speed -> road-wheel steering angle (rad, +left).

    pure_pursuit   adaptive lookahead L = clip(k v + L0, Lmin, Lmax),
                   delta = atan(2 L_wb sin(alpha) / L)       (research: 2.5-5 m bounds)
    stanley        heading error + atan(k e / (ks + v)) at the front axle
    p_steer        ucr-dv-timeless' controller: gain * bearing to a lookahead point
    mpc            kinematic-bicycle path-tracking MPC: states (e_y, e_psi),
                   input delta, linearised, horizon N, solved in closed form
                   (batch least squares with input-rate cost), then clipped.
                   KIT 2022 / ETH 2020 use (N)MPC; this is the teaching version.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.registry import Component, P, register
from ..planning.pathutil import arc_length, cross_track, curvature, point_at


def _clip(v, lim):
    return max(-lim, min(lim, v))


class _Lateral(Component):
    def __init__(self, params, **context):
        super(_Lateral, self).__init__(params, **context)
        self.vehicle = context.get('vehicle')
        self.target = None

    @property
    def wheelbase(self):
        return self.vehicle.wheelbase if self.vehicle else 1.53

    @property
    def max_steer(self):
        return self.vehicle.max_steer if self.vehicle else math.radians(25)


@register('lateral', 'pure_pursuit', label='Pure pursuit (adaptive lookahead)',
          params=[P('k_v', 0.3, 0.0, 2.0, 0.02, 's', 'lookahead growth with speed'),
                  P('l0', 2.0, 0.5, 10.0, 0.1, 'm', 'lookahead at standstill'),
                  P('l_min', 2.5, 0.5, 10.0, 0.1, 'm', 'shortest lookahead'),
                  P('l_max', 6.0, 1.0, 20.0, 0.1, 'm', 'longest lookahead'),
                  P('gain', 1.0, 0.2, 3.0, 0.05, '', 'steering multiplier')],
          refs=['Heriot-Watt 2025 (adaptive PP)', 'IIT Bombay 2024 (PP beat Stanley in FSDS)'])
class PurePursuit(_Lateral):
    """Steer onto a point one lookahead distance down the path."""

    def steer(self, path, v, ctx):
        p = self.params
        if len(path) < 2:
            return 0.0
        L = min(max(p['k_v'] * v + p['l0'], p['l_min']), p['l_max'])
        # measure the lookahead from the rear axle, where the geometry is exact
        rear = -self.vehicle.lr if self.vehicle else 0.0
        tx, ty = point_at(path, L)
        tx -= rear
        ld = math.hypot(tx, ty)
        if ld < 0.3:
            return 0.0
        alpha = math.atan2(ty, tx)
        self.target = (tx + rear, ty)
        delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld) * p['gain']
        return _clip(delta, self.max_steer)


@register('lateral', 'stanley', label='Stanley',
          params=[P('k', 1.5, 0.0, 10.0, 0.05, '', 'cross-track gain'),
                  P('k_soft', 1.0, 0.1, 10.0, 0.1, 'm/s', 'low-speed softening'),
                  P('k_heading', 1.0, 0.0, 3.0, 0.05, '', 'heading-error gain'),
                  P('preview', 0.5, 0.0, 5.0, 0.1, 'm', 'tangent measured this far ahead of the front axle')],
          refs=['Stanford Stanley (DARPA 2005)', 'IIT Bombay 2024'])
class Stanley(_Lateral):
    """Front-axle cross-track + heading error."""

    def steer(self, path, v, ctx):
        p = self.params
        if len(path) < 2:
            return 0.0
        front = (self.vehicle.lf if self.vehicle else 0.8, 0.0)
        e, heading = cross_track(path, front)
        # heading of the path a little ahead of the front axle
        s = arc_length(path)
        k = int(np.argmin(np.hypot(path[:, 0] - front[0], path[:, 1] - front[1])))
        a = point_at(path, s[k] + p['preview'])
        b = point_at(path, s[k] + p['preview'] + 1.0)
        heading = math.atan2(b[1] - a[1], b[0] - a[0])
        self.target = tuple(a)
        # e > 0 means the axle is LEFT of the path -> steer right (negative)
        delta = p['k_heading'] * heading + math.atan2(-p['k'] * e, p['k_soft'] + v)
        return _clip(delta, self.max_steer)


@register('lateral', 'p_steer', label='Proportional bearing (timeless)',
          params=[P('lookahead', 4.5, 0.5, 15.0, 0.1, 'm', 'distance along the path'),
                  P('gain', 0.55, 0.05, 3.0, 0.01, '', 'road-wheel rad per rad of bearing')],
          refs=['ucr-dv-timeless stanley_steering (lateral gain 0)'])
class PSteer(_Lateral):
    """gain x bearing to a lookahead point (the original teaching controller)."""

    def steer(self, path, v, ctx):
        if len(path) < 2:
            return 0.0
        tx, ty = point_at(path, self.params['lookahead'])
        self.target = (tx, ty)
        if math.hypot(tx, ty) < 0.3:
            return 0.0
        return _clip(self.params['gain'] * math.atan2(ty, tx), self.max_steer)


@register('lateral', 'mpc', label='Kinematic MPC (linear, closed form)',
          params=[P('horizon', 15, 5, 40, 1, 'steps', 'prediction steps'),
                  P('dt', 0.08, 0.02, 0.3, 0.01, 's', 'prediction step'),
                  P('q_ey', 2.0, 0.0, 50.0, 0.1, '', 'weight on lateral error'),
                  P('q_epsi', 1.0, 0.0, 50.0, 0.1, '', 'weight on heading error'),
                  P('r_delta', 0.5, 0.0, 50.0, 0.05, '', 'weight on steering'),
                  P('r_rate', 5.0, 0.0, 200.0, 0.5, '', 'weight on steering change'),
                  P('min_speed', 2.0, 0.5, 10.0, 0.5, 'm/s', 'linearise at least at this speed')],
          refs=['KIT 2022 (MPC lateral)', 'ETH 2020 (NMPC)', 'Monash thesis (MPC)'])
class KinematicMPC(_Lateral):
    """Error-state LTV MPC solved as one least-squares problem each tick."""

    def __init__(self, params, **context):
        super(KinematicMPC, self).__init__(params, **context)
        self.prev = 0.0

    def reset(self):
        self.prev = 0.0

    def steer(self, path, v, ctx):
        p = self.params
        if len(path) < 3:
            return 0.0
        N, dt = int(p['horizon']), p['dt']
        v = max(v, p['min_speed'])
        L = self.wheelbase
        e_y, heading = cross_track(path, (0.0, 0.0))   # + = car is left of the path
        e_psi = -heading                               # car heading (0) minus path heading
        s = arc_length(path)
        k_all = curvature(path)
        k0 = int(np.argmin(np.hypot(path[:, 0], path[:, 1])))
        s_pred = s[k0] + v * dt * np.arange(1, N + 1)
        kappa = np.interp(s_pred, s, k_all)
        # x = [e_y, e_psi]; x+ = A x + B d + E kappa
        A = np.array([[1.0, v * dt], [0.0, 1.0]])
        B = np.array([0.0, v * dt / L])
        E = np.array([0.0, -v * dt])
        x0 = np.array([e_y, e_psi])
        # batch: X = Phi x0 + G U + H K
        Phi = np.zeros((2 * N, 2))
        G = np.zeros((2 * N, N))
        Hk = np.zeros((2 * N, N))
        Ak = np.eye(2)
        for i in range(N):
            Ak = A @ Ak if i else A.copy()
            Phi[2 * i:2 * i + 2] = Ak
        for i in range(N):
            Apow = np.eye(2)
            for j in range(i, -1, -1):
                G[2 * i:2 * i + 2, j] = Apow @ B
                Hk[2 * i:2 * i + 2, j] = Apow @ E
                Apow = A @ Apow
        Q = np.diag(np.tile([p['q_ey'], p['q_epsi']], N))
        R = p['r_delta'] * np.eye(N)
        D = np.eye(N) - np.eye(N, k=-1)
        Rd = p['r_rate'] * D.T @ D
        d_prev = np.zeros(N)
        d_prev[0] = self.prev
        free = Phi @ x0 + Hk @ kappa
        Hm = G.T @ Q @ G + R + Rd
        f = G.T @ Q @ free - p['r_rate'] * D.T @ d_prev
        try:
            U = -np.linalg.solve(Hm, f)
        except np.linalg.LinAlgError:
            return self.prev
        delta = _clip(float(U[0]), self.max_steer)
        self.prev = delta
        # predicted path (car frame) for display
        self.target = tuple(point_at(path, min(s[-1], v * dt * N)))
        return delta
