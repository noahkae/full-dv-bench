"""
ROS 2 adapter: publishes what fsds_ros2_bridge publishes, so the stack's
topic contract (docs/CONTRIBUTING.md) is unchanged.

    /lidar/<name>        sensor_msgs/PointCloud2   frame fsds/<name>
    /imu                 sensor_msgs/Imu           frame fsds/FSCar
    /gps                 sensor_msgs/NavSatFix     frame fsds/FSCar
    /gss                 geometry_msgs/TwistStamped (contract type; use
                         --gss-type covariance for TwistWithCovarianceStamped)
    /testing_only/odom   nav_msgs/Odometry         fsds/map -> fsds/FSCar
    /tf_static           car -> each lidar / camera, from settings.json
    /fsds/<cam>/image_color  sensor_msgs/Image bgr8, or 32FC1 metres for depth
                         cameras (ImageType 1/2) like FSDS     (if cameras render)
    /fsds/<cam>/camera_info  sensor_msgs/CameraInfo     plumb_bob, no distortion
    /clock               rosgraph_msgs/Clock       sim time, with every GSS sample
                         like FSDS (lets nodes run with use_sim_time)

If fs_msgs is installed (it is, wherever the FSDS bridge was built) we also
provide the FSDS extras:

    /testing_only/track      fs_msgs/Track (latched)
    /testing_only/extra_info fs_msgs/ExtraInfo, 1 Hz: doo_counter + lap times
    /wheel_states            fs_msgs/WheelStates, 100 Hz
    /signal/go               fs_msgs/GoSignal, 1 Hz (same stamp every time, like FSDS)
    /signal/finished         fs_msgs/FinishedSignal  (subscribed: the AS hands back control)
    /control_command         fs_msgs/ControlCommand  (subscribed; obeys API control)
    /reset                   fs_msgs/srv/Reset

Competition mode (--competition-mode) drops every /testing_only topic, as
the FSDS bridge's competition_mode parameter does.

Odometry twist is the true body-frame velocity (REP-103). The FSDS bridge
flips the sign of the lateral component; nothing in the stack reads it.

rclpy is imported here and only here, so the rest of fsds_lite runs without ROS.
"""
import array
import logging
import math
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, TwistStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, NavSatFix, NavSatStatus, PointCloud2, PointField
from tf2_msgs.msg import TFMessage

from .geometry import rpy_to_quat
from .simulator import (CameraSample, Controls, GpsSample, GssSample, ImuSample, LidarSample,
                        OdomSample, WheelStatesSample)

try:
    from fs_msgs.msg import (ControlCommand, ExtraInfo, FinishedSignal, GoSignal,
                             Track as TrackMsg, Cone as ConeMsg, WheelStates)
    from fs_msgs.srv import Reset as ResetSrv
    HAVE_FS_MSGS = True
except ImportError:
    HAVE_FS_MSGS = False

log = logging.getLogger('fsds_lite.ros')

_FIELDS = []
for _i, _n in enumerate('xyz'):
    _f = PointField()
    _f.name, _f.offset, _f.datatype, _f.count = _n, 4 * _i, PointField.FLOAT32, 1
    _FIELDS.append(_f)


def to_time(stamp):
    t = Time()
    t.sec = int(stamp)
    t.nanosec = int(round((stamp - int(stamp)) * 1e9)) % 1000000000
    return t


def _has_subscribers(pub):
    """Whether anything (a node, ros2 bag, rviz) currently subscribes to `pub`.
    Building a big message costs Python time under the GIL - in the workbench's
    --lite mode that's the GUI's time too - so skip it when no one listens."""
    try:
        return pub.get_subscription_count() > 0
    except Exception:
        return True


class RosBridge(object):
    def __init__(self, sim, sensors, namespace='', gss_type='twist', mission=None,
                 publish_cameras=True, competition_mode=False, own_rclpy=True):
        self.sim = sim
        self.own_rclpy = own_rclpy      # False when the host app owns rclpy.init/shutdown
        self.sensors = sensors
        self.competition = competition_mode
        self.vehicle_frame = 'fsds/' + sensors.vehicle_name
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('fsds_lite', namespace=namespace)
        n = self.node

        self.lidar_pubs = {l.name: n.create_publisher(PointCloud2, 'lidar/' + l.name, 10)
                           for l in sensors.lidars}
        self.imu_pub = n.create_publisher(Imu, 'imu', 10) if sensors.imu else None
        self.gps_pub = n.create_publisher(NavSatFix, 'gps', 10) if sensors.gps else None
        self.gss_cov = gss_type == 'covariance'
        if sensors.gss:
            self.gss_pub = n.create_publisher(
                TwistWithCovarianceStamped if self.gss_cov else TwistStamped, 'gss', 10)
        else:
            self.gss_pub = None
        self.odom_pub = None if competition_mode else n.create_publisher(Odometry, 'testing_only/odom', 10)
        self.clock_pub = n.create_publisher(Clock, '/clock', 10)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tf_pub = n.create_publisher(TFMessage, '/tf_static', latched)
        self._publish_static_tf()

        # cameras keep the /fsds prefix, like the FSDS bridge (absolute names)
        self.cam_pubs = {}
        if publish_cameras:
            for cam in sensors.cameras or []:
                self.cam_pubs[cam.name] = (
                    n.create_publisher(Image, '/fsds/{}/image_color'.format(cam.name), 1),
                    n.create_publisher(CameraInfo, '/fsds/{}/camera_info'.format(cam.name), 1))

        self.wheel_pub = None
        if HAVE_FS_MSGS:
            self.mission = mission or sim.track.mission
            self._go_stamp = None
            if not competition_mode:
                self.track_pub = n.create_publisher(TrackMsg, 'testing_only/track', latched)
                self._publish_track()
                self.extra_pub = n.create_publisher(ExtraInfo, 'testing_only/extra_info', 10)
                n.create_timer(1.0, self._extra_info)
            self.wheel_pub = n.create_publisher(WheelStates, 'wheel_states', 10)
            self.go_pub = n.create_publisher(GoSignal, 'signal/go', 1)
            n.create_timer(1.0, self._go)
            n.create_subscription(ControlCommand, 'control_command', self._on_control, 1)
            n.create_subscription(FinishedSignal, 'signal/finished', self._on_finished, 1)
            n.create_service(ResetSrv, 'reset', self._on_reset)
        else:
            log.warning('fs_msgs not found: /control_command, /signal/go, /signal/finished, '
                        '/wheel_states, /testing_only/track and /extra_info are disabled '
                        '(RPC control still works)')

        sim.add_listener(self._on_sample)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(n)
        self.thread = threading.Thread(target=self._spin, name='ros', daemon=True)

    def start(self):
        self.thread.start()
        log.info('ROS 2 bridge up (node %s)', self.node.get_fully_qualified_name())
        return self

    def _spin(self):
        try:
            self.executor.spin()
        except Exception:  # shutdown races are harmless
            pass

    def stop(self):
        self.executor.shutdown()
        self.node.destroy_node()
        if self.own_rclpy and rclpy.ok():
            rclpy.shutdown()

    # ---- sim -> ROS ----------------------------------------------------
    def _on_sample(self, s):
        if isinstance(s, LidarSample):
            self._lidar(s)
        elif isinstance(s, CameraSample):
            self._camera(s)
        elif isinstance(s, ImuSample) and self.imu_pub:
            m = Imu()
            m.header.stamp, m.header.frame_id = to_time(s.stamp), self.vehicle_frame
            o = m.orientation
            o.x, o.y, o.z, o.w = s.orientation
            v = m.angular_velocity
            v.x, v.y, v.z = s.angular_velocity
            a = m.linear_acceleration
            a.x, a.y, a.z = s.linear_acceleration
            self.imu_pub.publish(m)
        elif isinstance(s, GpsSample) and self.gps_pub:
            m = NavSatFix()
            m.header.stamp, m.header.frame_id = to_time(s.stamp), self.vehicle_frame
            m.status.status = NavSatStatus.STATUS_FIX
            m.status.service = NavSatStatus.SERVICE_GPS
            m.latitude, m.longitude, m.altitude = s.latitude, s.longitude, s.altitude
            cov = [0.0] * 9
            cov[0] = cov[4] = s.eph ** 2
            cov[8] = s.epv ** 2
            m.position_covariance = cov
            m.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
            self.gps_pub.publish(m)
        elif isinstance(s, GssSample) and self.gss_pub:
            if self.gss_cov:
                m = TwistWithCovarianceStamped()
                twist = m.twist.twist
                cov = [0.0] * 36
                for i in range(6):
                    cov[i * 7] = 0.1
                m.twist.covariance = cov
            else:
                m = TwistStamped()
                twist = m.twist
            m.header.stamp, m.header.frame_id = to_time(s.stamp), self.vehicle_frame
            twist.linear.x, twist.linear.y, twist.linear.z = s.linear_velocity
            twist.angular.x, twist.angular.y, twist.angular.z = s.angular_velocity
            self.gss_pub.publish(m)
            clock = Clock()
            clock.clock = m.header.stamp
            self.clock_pub.publish(clock)
        elif isinstance(s, WheelStatesSample) and self.wheel_pub is not None:
            m = WheelStates()
            m.header.stamp, m.header.frame_id = to_time(s.stamp), self.vehicle_frame
            m.fl_rpm, m.fr_rpm, m.rl_rpm, m.rr_rpm = (float(v) for v in s.rpm)
            (m.fl_rotation_angle, m.fr_rotation_angle,
             m.rl_rotation_angle, m.rr_rotation_angle) = (float(v) for v in s.rotation_angle)
            (m.fl_steering_angle, m.fr_steering_angle,
             m.rl_steering_angle, m.rr_steering_angle) = (float(v) for v in s.steering_angle)
            self.wheel_pub.publish(m)
        elif isinstance(s, OdomSample) and self.odom_pub is not None:
            m = Odometry()
            m.header.stamp, m.header.frame_id = to_time(s.stamp), 'fsds/map'
            m.child_frame_id = self.vehicle_frame
            p = m.pose.pose.position
            p.x, p.y, p.z = s.position
            o = m.pose.pose.orientation
            o.x, o.y, o.z, o.w = s.orientation
            t = m.twist.twist
            t.linear.x, t.linear.y, t.linear.z = s.linear_velocity
            t.angular.x, t.angular.y, t.angular.z = s.angular_velocity
            self.odom_pub.publish(m)

    def _lidar(self, s):
        pub = self.lidar_pubs.get(s.name)
        if pub is None or not _has_subscribers(pub):
            return
        m = PointCloud2()
        m.header.stamp, m.header.frame_id = to_time(s.stamp), 'fsds/' + s.name
        m.height, m.width = 1, len(s.points)
        m.fields = _FIELDS
        m.is_bigendian = False
        m.point_step = 12
        m.row_step = 12 * len(s.points)
        m.is_dense = True
        buf = array.array('B')
        buf.frombytes(s.points.astype('<f4').tobytes())
        m.data = buf
        pub.publish(m)

    def _camera(self, s):
        pubs = self.cam_pubs.get(s.name)
        if pubs is None or not (_has_subscribers(pubs[0]) or _has_subscribers(pubs[1])):
            return      # ~1.8 MB per frame: don't serialize it for nobody
        h, w = s.image.shape[:2]
        frame = 'fsds/' + s.name
        img = Image()
        img.header.stamp, img.header.frame_id = to_time(s.stamp), frame
        if s.image.dtype == 'float32':                 # depth camera, metres
            img.height, img.width, img.encoding = h, w, '32FC1'
            img.is_bigendian, img.step = 0, w * 4
        else:
            img.height, img.width, img.encoding = h, w, 'bgr8'
            img.is_bigendian, img.step = 0, w * 3
        buf = array.array('B')
        buf.frombytes(s.image.tobytes())
        img.data = buf
        pubs[0].publish(img)

        fx, fy, cx, cy = s.intrinsics
        info = CameraInfo()
        info.header = img.header
        info.height, info.width = h, w
        info.distortion_model = 'plumb_bob'
        info.d = [0.0] * 5
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        pubs[1].publish(info)

    def _publish_static_tf(self):
        msg = TFMessage()
        stamp = to_time(self.sim.stamp())
        frames = [(l.name, (l.x, l.y, l.z), (l.roll, l.pitch, l.yaw)) for l in self.sensors.lidars]
        frames += [(c.name, (c.x, c.y, c.z), (c.roll, c.pitch, c.yaw))
                   for c in self.sensors.cameras or []]
        for name, xyz, rpy in frames:
            t = TransformStamped()
            t.header.stamp, t.header.frame_id = stamp, self.vehicle_frame
            t.child_frame_id = 'fsds/' + name
            tr = t.transform.translation
            tr.x, tr.y, tr.z = (float(v) for v in xyz)
            q = t.transform.rotation
            q.x, q.y, q.z, q.w = rpy_to_quat(*(math.radians(v) for v in rpy))
            msg.transforms.append(t)
        self.tf_pub.publish(msg)

    # ---- fs_msgs extras -----------------------------------------------
    def _publish_track(self):
        msg = TrackMsg()
        colors = {'blue': ConeMsg.BLUE, 'yellow': ConeMsg.YELLOW,
                  'orange': ConeMsg.ORANGE_SMALL, 'big_orange': ConeMsg.ORANGE_BIG}
        for (x, y), c in zip(self.sim.track.positions, self.sim.track.colors):
            cone = ConeMsg()
            cone.location.x, cone.location.y = float(x), float(y)
            cone.color = colors.get(c, ConeMsg.UNKNOWN)
            msg.track.append(cone)
        self.track_pub.publish(msg)

    def _extra_info(self):
        ref = self.sim.referee
        m = ExtraInfo()
        m.doo_counter = int(ref.doo)
        m.laps = [float(t) for t in ref.laps]
        self.extra_pub.publish(m)

    def _on_finished(self, msg):
        self.sim.finish()

    def _go(self):
        m = GoSignal()
        if self._go_stamp is None:                    # FSDS: every GO carries the first stamp
            self._go_stamp = self.sim.stamp()
        m.header.stamp = to_time(self._go_stamp)
        m.mission = self.mission
        m.track = self.sim.track.name
        self.go_pub.publish(m)

    def _on_control(self, msg):
        self.sim.set_api_controls(Controls(throttle=msg.throttle, steering=msg.steering,
                                           brake=msg.brake))

    def _on_reset(self, request, response):
        self.sim.reset()
        self._go_stamp = None                         # a reset starts a new run
        return response
