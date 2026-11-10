"""
Vehicle model: four independently spinning wheels on a planar chassis, with a
sprung body that pitches and rolls and moves load between the wheels.

Chassis (x, y, yaw, vx, vy, r)
    Planar rigid body driven by the sum of the four tyre forces. At low speed
    it blends into a kinematic bicycle (the dynamic model is singular as
    vx -> 0), exactly as before.

Wheels and tyres (like PhysX, which FSDS uses)
    Each wheel has its own angular velocity. Its longitudinal slip ratio
        kappa = (omega * R - v_long) / max(|v_long|, v_floor)
    sets the drive/brake force through a Pacejka-style curve, so too much
    throttle spins the rear wheels and too much brake locks them.
    Lateral force comes from the wheel's own slip angle. Each tyre has a
    friction ellipse (longitudinal force takes priority), so a spinning or
    locked wheel loses cornering grip, which is what makes power oversteer
    and locked-front-wheel understeer happen.
    The wheel-speed update is semi-implicit (linearised tyre slope), so it
    stays stable at 1 kHz even near standstill, where the slip dynamics are
    very stiff.

Differential (rear, like PhysX PxVehicleDifferential4WData)
    open           fixed torque split (default 50/50)
    limited_slip   open until the faster wheel spins `diff_bias` times faster
                   than the slower one, then torque shifts to the slower wheel
                   (PhysX's speed-ratio bias, default 1.3 like UE4)
    locked         both rear wheels turn at one speed (a spool)
    The left/right force difference yaws the car, so a locked diff pushes
    wide (understeer) and an open diff lets the inside wheel spin up.

Body attitude (roll, pitch)
    Second-order spring/damper driven by the inertial force at CoG height:
        Ix * roll''  =  m * ay * h - k_roll  * roll  - c_roll  * roll'
        Iy * pitch'' = -m * ax * h - k_pitch * pitch - c_pitch * pitch'
    ENU conventions: +roll = right side down (it happens in a LEFT turn),
    +pitch = nose down (it happens under braking). The spring/damper moments
    set the four wheel loads, and grip is load-sensitive.

Powertrain
    Engine torque from a piecewise-linear torque curve; engine rpm locked to
    the differential carrier (mean rear wheel speed) through the gear and
    final drive; slipping launch clutch below `launch_rpm`; soft rev limiter;
    automatic gearbox with a torque cut per shift; engine braking. With the
    clutch engaged, the engine's inertia is felt at the rear wheels
    (multiplied by gear ratio squared). Like a real gearbox controller, the
    shift points use road speed, so wheelspin can hit the limiter but can't
    make the box hunt between gears. Optional traction control trims torque
    to hold the rear slip near `traction_control_slip`. There is no reverse
    gear.

Controls follow FSDS/AirSim CarControls exactly:
    throttle  0..1
    brake     0..1
    steering -1..1, POSITIVE STEERS RIGHT (FSDS passes it straight to
              Unreal's SetSteeringInput, where +ve is right). In ENU that
              is a negative wheel angle.

All maths uses plain floats - at a 1 kHz step that is far faster than numpy.
"""
import bisect
import json
import math
from dataclasses import dataclass, field, fields

G = 9.81
RPM_PER_RAD_S = 60.0 / (2.0 * math.pi)
LIMITER_BAND = 200.0   # rpm over which the rev limiter fades torque out
V_FLOOR = 1.0          # m/s: slip-ratio denominator floor (standard low-speed fix)
FL, FR, RL, RR = 0, 1, 2, 3
DIFF_TYPES = ('open', 'limited_slip', 'locked')


@dataclass
class VehicleParams:
    mass: float = 255.0
    inertia_z: float = 120.0
    inertia_x: float = 40.0
    inertia_y: float = 110.0
    lf: float = 0.80                 # CoG -> front axle
    lr: float = 0.73                 # CoG -> rear axle
    track_width: float = 1.2
    cg_height: float = 0.25          # FSDS: CoG 25 cm above the pawn centre
    max_steer_deg: float = 25.0
    steer_rate_deg_s: float = 300.0  # steering actuator slew limit
    # engine + gearbox (defaults: ~40 Nm single-cylinder FS engine, 4 speeds,
    # rev limit gives ~27 m/s top speed like the FSDS car)
    torque_curve: tuple = ((3000.0, 30.0), (6000.0, 38.0), (9000.0, 42.0),
                           (11000.0, 40.0), (12500.0, 32.0))   # (rpm, Nm)
    idle_rpm: float = 3000.0
    launch_rpm: float = 7000.0       # clutch-slip rpm at full throttle from standstill
    max_rpm: float = 12500.0         # rev limiter
    gear_ratios: tuple = (6.0, 4.5, 3.5, 2.8)
    final_drive: float = 4.0
    drivetrain_efficiency: float = 0.9
    upshift_rpm: float = 11800.0
    downshift_rpm: float = 6000.0
    shift_time: float = 0.12         # s of zero drive torque per shift
    engine_brake_torque: float = 4.0  # Nm at max_rpm with the throttle closed
    engine_inertia: float = 0.01     # kg m^2 (crank + clutch), felt through the gearing on throttle
    traction_control: bool = False   # FSDS's PhysX car has none
    traction_control_slip: float = 0.12
    # differential (rear)
    diff_type: str = 'limited_slip'  # open | limited_slip | locked
    diff_bias: float = 1.3           # limited_slip: max fast/slow wheel speed ratio (UE4 default)
    diff_split: float = 0.5          # share of torque to the left wheel when open
    # brakes
    max_brake_force: float = 3000.0  # total, at the contact patches (~threshold braking on dry)
    brake_bias_front: float = 0.6
    # resistances
    drag_coefficient: float = 0.3
    frontal_area: float = 1.1
    air_density: float = 1.225
    rolling_resistance: float = 0.015
    # tyres
    mu: float = 1.5
    pacejka_b: float = 10.0          # lateral
    pacejka_c: float = 1.6
    pacejka_bx: float = 12.0         # longitudinal (peak at ~12% slip)
    pacejka_cx: float = 1.65
    load_sensitivity: float = 0.1    # mu falls 10% per +100% wheel load
    wheel_radius: float = 0.23
    wheel_inertia: float = 0.3       # kg m^2 per wheel (wheel + tyre + brake disc)
    # suspension
    roll_stiffness: float = 35000.0  # N m/rad  (~1 deg of roll per g)
    roll_damping: float = 1650.0     # N m s/rad (~0.7 of critical)
    roll_stiffness_front: float = 0.5  # share of roll moment carried by the front axle
    pitch_stiffness: float = 50000.0
    pitch_damping: float = 3300.0
    length: float = 1.8              # collision box, centred on the pawn centre
    width: float = 1.0
    kinematic_below: float = 1.5     # m/s: pure kinematic model below this
    dynamic_above: float = 3.0       # m/s: pure dynamic model above this

    @property
    def wheelbase(self):
        return self.lf + self.lr

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            raw = json.load(f)
        known = {f.name for f in fields(cls)}
        unknown = [k for k in raw if k not in known and not k.startswith('_')]
        if unknown:
            raise ValueError('Unknown vehicle parameter(s) in {}: {}'.format(path, ', '.join(unknown)))

        def convert(v):
            if isinstance(v, (list, tuple)):
                return tuple(convert(x) for x in v)
            if isinstance(v, (str, bool)):
                return v
            return float(v)
        params = cls(**{k: convert(v) for k, v in raw.items() if k in known})
        params.validate()
        return params

    def validate(self):
        rpms = [r for r, _ in self.torque_curve]
        if len(rpms) < 2 or rpms != sorted(rpms):
            raise ValueError('torque_curve needs >= 2 points with increasing rpm')
        if not self.gear_ratios or list(self.gear_ratios) != sorted(self.gear_ratios, reverse=True):
            raise ValueError('gear_ratios must be decreasing (1st gear first)')
        for lo, hi in zip(self.gear_ratios[1:], self.gear_ratios[:-1]):
            # after an upshift the rpm must land above the downshift point, or it hunts
            if self.upshift_rpm * lo / hi <= self.downshift_rpm:
                raise ValueError('upshift_rpm/downshift_rpm too close for gear step {:.2f} -> {:.2f}'
                                 .format(hi, lo))
        if self.diff_type not in DIFF_TYPES:
            raise ValueError('diff_type must be one of {}'.format(', '.join(DIFF_TYPES)))
        if self.diff_bias < 1.0:
            raise ValueError('diff_bias must be >= 1')

    def torque_at(self, rpm):
        """Linear interpolation of the torque curve, flat beyond its ends."""
        curve = self.torque_curve
        if rpm <= curve[0][0]:
            return curve[0][1]
        if rpm >= curve[-1][0]:
            return curve[-1][1]
        i = bisect.bisect_right([r for r, _ in curve], rpm)
        (r0, t0), (r1, t1) = curve[i - 1], curve[i]
        return t0 + (t1 - t0) * (rpm - r0) / (r1 - r0)

    def overall_ratio(self, gear):
        return self.gear_ratios[gear - 1] * self.final_drive

    def top_speed(self):
        """Speed at the rev limiter in top gear (ignoring drag and slip)."""
        return self.max_rpm / RPM_PER_RAD_S / self.overall_ratio(len(self.gear_ratios)) * self.wheel_radius


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0     # body frame, forward
    vy: float = 0.0     # body frame, left
    r: float = 0.0      # yaw rate
    delta: float = 0.0  # front wheel angle (ENU, +ve = left)
    ax: float = 0.0     # planar acceleration in the yaw frame, for the IMU
    ay: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    gear: int = 1
    rpm: float = 0.0
    shift_timer: float = 0.0
    # per wheel, order FL, FR, RL, RR
    omega: list = field(default_factory=lambda: [0.0] * 4)        # rad/s
    wheel_angle: list = field(default_factory=lambda: [0.0] * 4)  # accumulated, rad
    slip: list = field(default_factory=lambda: [0.0] * 4)         # longitudinal slip ratio
    fz: tuple = (0.0, 0.0, 0.0, 0.0)                               # wheel loads (N)

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)

    @property
    def wheel_spin(self):
        """Mean wheel angle (kept for older callers)."""
        return sum(self.wheel_angle) / 4.0

    def world_velocity(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return (self.vx * c - self.vy * s, self.vx * s + self.vy * c)


def _clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class Vehicle(object):
    def __init__(self, params=None):
        self.p = params or VehicleParams()
        self.p.validate()
        h = self.p.track_width / 2.0
        self.wheel_xy = ((self.p.lf, h), (self.p.lf, -h), (-self.p.lr, h), (-self.p.lr, -h))
        self.reset()

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        self.state = VehicleState(x=x, y=y, yaw=yaw, rpm=self.p.idle_rpm)
        self.state.fz = self._static_loads()

    def teleport(self, x, y, yaw):
        """Move without changing velocity (simSetVehiclePose)."""
        self.state.x, self.state.y, self.state.yaw = x, y, yaw

    def stop(self):
        """Kill all motion (used when the car hits a solid obstacle)."""
        s = self.state
        s.vx = s.vy = s.r = 0.0
        s.omega = [0.0] * 4

    def _static_loads(self):
        p = self.p
        f = p.mass * G * p.lr / p.wheelbase / 2.0
        r = p.mass * G * p.lf / p.wheelbase / 2.0
        return (f, f, r, r)

    def _mu(self, fz):
        """Load-sensitive friction coefficient for one wheel."""
        p = self.p
        nominal = p.mass * G / 4.0
        return max(0.1, p.mu * (1.0 - p.load_sensitivity * (fz / nominal - 1.0)))

    # ---- powertrain -------------------------------------------------------
    def _engine(self, dt, throttle, carrier_omega, vx):
        """Gearbox + engine. Returns (torque at the diff carrier, reflected inertia)."""
        p, s = self.p, self.state
        n_gears = len(p.gear_ratios)
        carrier_rpm = max(carrier_omega, 0.0) * RPM_PER_RAD_S
        road_rpm = max(vx, 0.0) / p.wheel_radius * RPM_PER_RAD_S   # rpm with zero slip

        if s.shift_timer > 0.0:
            s.shift_timer = max(0.0, s.shift_timer - dt)
        else:
            shift_rpm = road_rpm * p.overall_ratio(s.gear)
            if shift_rpm > p.upshift_rpm and s.gear < n_gears:
                s.gear += 1
                s.shift_timer = p.shift_time
            elif shift_rpm < p.downshift_rpm and s.gear > 1:
                s.gear -= 1
                s.shift_timer = p.shift_time
        ratio = p.overall_ratio(s.gear)
        locked = carrier_rpm * ratio

        # clutch slips below the launch rpm so the engine can pull away
        clutch = p.idle_rpm + throttle * (p.launch_rpm - p.idle_rpm)
        # the flywheel is felt at the wheels only while driving through the clutch
        engaged = locked >= clutch and s.shift_timer <= 0.0 and throttle > 0.05
        s.rpm = min(max(locked, clutch), p.max_rpm)
        inertia = p.engine_inertia * ratio * ratio if engaged else 0.0
        if s.shift_timer > 0.0:
            return 0.0, 0.0

        limiter = min(max((p.max_rpm - locked) / LIMITER_BAND, 0.0), 1.0)
        torque = throttle * p.torque_at(s.rpm) * limiter
        if p.traction_control:
            excess = max(s.slip[RL], s.slip[RR]) - p.traction_control_slip
            torque *= _clip(1.0 - 4.0 * excess, 0.0, 1.0)
        if throttle < 0.05 and vx > 1.0 and locked > p.idle_rpm:
            torque -= p.engine_brake_torque * locked / p.max_rpm
        return torque * ratio * p.drivetrain_efficiency, inertia

    def _diff_split(self, torque, w_left, w_right):
        """Torque to (left, right) rear wheels for an open or limited-slip diff."""
        p = self.p
        left = p.diff_split
        if p.diff_type == 'limited_slip' and torque > 0.0:
            fast, slow = max(w_left, w_right), min(w_left, w_right)
            ratio = fast / max(slow, 1e-3) if fast > 1.0 else 1.0
            if ratio > p.diff_bias:
                # past the bias the diff locks up: within 10% beyond it the
                # faster wheel's share fades to nothing and the slower
                # (gripping) wheel gets the torque, as in PhysX
                keep = _clip(1.0 - (ratio - p.diff_bias) / (0.1 * p.diff_bias), 0.0, 1.0)
                fast_share = (left if w_left >= w_right else 1.0 - left) * keep
                left = fast_share if w_left >= w_right else 1.0 - fast_share
        return torque * left, torque * (1.0 - left)

    # ---- tyres ------------------------------------------------------------
    def _long_force(self, kappa, cap):
        """Pacejka-style longitudinal force and its slope dF/dkappa (clamped >= 0)."""
        p = self.p
        bk = p.pacejka_bx * kappa
        a = math.atan(bk)
        f = cap * math.sin(p.pacejka_cx * a)
        slope = cap * math.cos(p.pacejka_cx * a) * p.pacejka_cx * p.pacejka_bx / (1.0 + bk * bk)
        return f, max(slope, 0.0)

    def _spin(self, dt, wheels, drive, brake_torque, inertia):
        """Semi-implicit update of a group of wheels sharing one speed.

        `wheels`: indices that turn together (1 wheel, or 2 for a locked diff).
        Linearising F(kappa) around the current slip keeps this stable however
        stiff the tyre is; brakes act as Coulomb friction (they can hold a
        wheel at zero but never spin it backwards).
        """
        p, s = self.p, self.state
        R = p.wheel_radius
        omega = s.omega[wheels[0]]
        net = drive
        stiff = 0.0
        for i in wheels:
            v, vd, cap = self._v_long[i], self._v_den[i], self._cap[i]
            f, slope = self._long_force((omega * R - v) / vd, cap)
            net -= R * f
            stiff += slope * R * R / vd
        denom = inertia + dt * stiff
        new = omega + dt * net / denom
        hold = dt * brake_torque / denom
        new = 0.0 if new <= hold else new - hold
        for i in wheels:
            s.omega[i] = new

    # ---- step ---------------------------------------------------------------
    def step(self, dt, throttle, brake, steering):
        p, s = self.p, self.state
        throttle = _clip(throttle, 0.0, 1.0)
        brake = _clip(brake, 0.0, 1.0)
        steering = _clip(steering, -1.0, 1.0)
        L, R = p.wheelbase, p.wheel_radius

        # steering actuator: +ve command = right = negative ENU wheel angle
        max_steer = math.radians(p.max_steer_deg)
        target = -steering * max_steer
        max_d = math.radians(p.steer_rate_deg_s) * dt
        s.delta += _clip(target - s.delta, -max_d, max_d)
        delta = s.delta
        cd, sd = math.cos(delta), math.sin(delta)

        # wheel loads from the suspension moments (static + transfer)
        sf, _, sr, _ = self._static_loads()
        m_pitch = p.pitch_stiffness * s.pitch + p.pitch_damping * s.pitch_rate
        m_roll = p.roll_stiffness * s.roll + p.roll_damping * s.roll_rate
        d_long = m_pitch / L / 2.0                                    # per wheel
        d_lat_f = p.roll_stiffness_front * m_roll / p.track_width
        d_lat_r = (1.0 - p.roll_stiffness_front) * m_roll / p.track_width
        fz = (max(0.0, sf + d_long - d_lat_f),       # +roll = right side down
              max(0.0, sf + d_long + d_lat_f),
              max(0.0, sr - d_long - d_lat_r),
              max(0.0, sr - d_long + d_lat_r))
        s.fz = fz
        self._cap = [self._mu(f) * f for f in fz]

        # contact-patch velocities in each wheel's own frame
        vx, vy, r = s.vx, s.vy, s.r
        v_long, v_lat = [0.0] * 4, [0.0] * 4
        for i, (wx, wy) in enumerate(self.wheel_xy):
            ux, uy = vx - r * wy, vy + r * wx
            if i < 2:                                                 # steered fronts
                v_long[i], v_lat[i] = ux * cd + uy * sd, -ux * sd + uy * cd
            else:
                v_long[i], v_lat[i] = ux, uy
        self._v_long = v_long
        self._v_den = [max(abs(v), V_FLOOR) for v in v_long]

        # engine -> differential -> rear wheels; brakes on all four
        carrier = 0.5 * (s.omega[RL] + s.omega[RR])
        t_axle, i_engine = self._engine(dt, throttle, carrier, vx)
        tb_front = brake * p.max_brake_force * p.brake_bias_front * R / 2.0
        tb_rear = brake * p.max_brake_force * (1.0 - p.brake_bias_front) * R / 2.0
        self._spin(dt, (FL,), 0.0, tb_front, p.wheel_inertia)
        self._spin(dt, (FR,), 0.0, tb_front, p.wheel_inertia)
        if p.diff_type == 'locked':
            self._spin(dt, (RL, RR), t_axle, 2.0 * tb_rear, 2.0 * p.wheel_inertia + i_engine)
        else:
            t_l, t_r = self._diff_split(t_axle, s.omega[RL], s.omega[RR])
            self._spin(dt, (RL,), t_l, tb_rear, p.wheel_inertia + 0.5 * i_engine)
            self._spin(dt, (RR,), t_r, tb_rear, p.wheel_inertia + 0.5 * i_engine)

        # tyre forces at the new wheel speeds; longitudinal has priority (friction ellipse)
        fx_body = fy_body = mz = 0.0
        for i, (wx, wy) in enumerate(self.wheel_xy):
            cap = self._cap[i]
            kappa = (s.omega[i] * R - v_long[i]) / self._v_den[i]
            s.slip[i] = kappa
            fx, _ = self._long_force(kappa, cap)
            alpha = math.atan2(v_lat[i], max(abs(v_long[i]), 0.5))
            fy = -cap * math.sin(p.pacejka_c * math.atan(p.pacejka_b * alpha))
            if cap > 0.0:
                used = min(1.0, abs(fx) / cap)
                fy *= math.sqrt(1.0 - used * used)
            if i < 2:                                                 # rotate fronts to body frame
                fx, fy = fx * cd - fy * sd, fx * sd + fy * cd
            fx_body += fx
            fy_body += fy
            mz += wx * fy - wy * fx

        f_resist = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * vx * vx
        if vx > 0.01:
            f_resist += p.rolling_resistance * p.mass * G
        fx_body -= f_resist

        dvx = fx_body / p.mass + vy * r
        dvy = fy_body / p.mass - vx * r
        dr = mz / p.inertia_z

        # blend factor: 0 = kinematic, 1 = dynamic. At low speed the tyre
        # slip angles are meaningless, so they must not feed back into vx.
        lo, hi = p.kinematic_below, p.dynamic_above
        lam = _clip((vx - lo) / (hi - lo), 0.0, 1.0) if hi > lo else 1.0
        dvx = lam * dvx + (1.0 - lam) * (fx_body / p.mass)

        new_vx = max(0.0, vx + dvx * dt)
        vy_dyn = vy + dvy * dt
        r_dyn = r + dr * dt

        # kinematic bicycle at the new speed (no slip)
        beta = math.atan(p.lr * math.tan(delta) / L)
        vy_kin = new_vx * math.tan(beta)
        r_kin = new_vx * math.cos(beta) * math.tan(delta) / L

        new_vy = lam * vy_dyn + (1.0 - lam) * vy_kin
        new_r = lam * r_dyn + (1.0 - lam) * r_kin

        # static friction: braked wheels that have stopped hold the car still
        # (slip-ratio forces fade to zero with speed, so without this the car
        # would creep towards zero forever)
        if brake > 0.05 and new_vx < 0.05 and not any(s.omega) and throttle < 0.05:
            new_vx = new_vy = new_r = 0.0

        # planar acceleration in the yaw frame (what the IMU sees, minus gravity and tilt)
        s.ax = (new_vx - vx) / dt - vy * r
        s.ay = (new_vy - vy) / dt + vx * r

        s.vx, s.vy, s.r = new_vx, new_vy, new_r
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        s.x += (s.vx * c - s.vy * sn) * dt
        s.y += (s.vx * sn + s.vy * c) * dt
        s.yaw = math.atan2(math.sin(s.yaw + s.r * dt), math.cos(s.yaw + s.r * dt))
        for i in range(4):
            s.wheel_angle[i] = (s.wheel_angle[i] + s.omega[i] * dt) % (2.0 * math.pi)

        # body attitude (semi-implicit Euler: rates first, then angles)
        h, m = p.cg_height, p.mass
        roll_acc = (m * s.ay * h - p.roll_stiffness * s.roll - p.roll_damping * s.roll_rate) / p.inertia_x
        pitch_acc = (-m * s.ax * h - p.pitch_stiffness * s.pitch - p.pitch_damping * s.pitch_rate) / p.inertia_y
        s.roll_rate += roll_acc * dt
        s.pitch_rate += pitch_acc * dt
        s.roll += s.roll_rate * dt
        s.pitch += s.pitch_rate * dt
