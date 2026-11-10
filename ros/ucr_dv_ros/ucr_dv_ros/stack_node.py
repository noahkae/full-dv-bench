"""
ucr_dv_stack node: one ucr-dv-stack profile running on ROS 2 topics.

In  (fsds_ros2_bridge / fsds_lite contract, names from the profile's rig):
    /lidar/<sensor>               sensor_msgs/PointCloud2   (sensor frame)
    /fsds/<camera>/image_color    sensor_msgs/Image (bgr8)  + /fsds/<camera>/camera_info
    /fsds/<camera>_depth/image_color  32FC1 depth for stereo/depth sensors (optional)
    /gss                          geometry_msgs/TwistStamped (ground speed, body frame)
    /imu                          sensor_msgs/Imu
    /gps                          sensor_msgs/NavSatFix
    /testing_only/odom            nav_msgs/Odometry (only used if odometry = ground_truth)
Out:
    /perception/cones   geometry_msgs/PoseArray (car frame, z = colour id)
    /slam/map           geometry_msgs/PoseArray (map frame, z = colour id)
    /planning/path      geometry_msgs/PoseArray (car frame)
    /control_command    fs_msgs/ControlCommand  (if fs_msgs is installed; fsds_lite and FSDS accept it)
    /signal/finished    fs_msgs/FinishedSignal  (when the mission is done)

Every algorithm parameter is a ROS parameter (`planner.max_gap`,
`sensor.lidar.min_cluster_points`, ...): `ros2 param set` tunes it live,
exactly like the workbench sliders.

The node finds the ucr_dv package through $UCR_DV_ROOT (the ucr-dv-stack
folder) or, with `colcon build --symlink-install`, relative to this file.
"""
import math
import os
import sys
import threading


def _find_root():
    cands = [os.environ.get('UCR_DV_ROOT', '')]
    here = os.path.realpath(os.path.abspath(__file__))
    for _ in range(6):
        here = os.path.dirname(here)
        cands.append(here)
    for c in cands:
        if c and os.path.isdir(os.path.join(c, 'ucr_dv')) and os.path.isdir(os.path.join(c, 'fsds_lite')):
            return c
    return None


ROOT = _find_root()
if ROOT and ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from geometry_msgs.msg import Pose, PoseArray, TwistStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rcl_interfaces.msg import SetParametersResult  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, Imu, NavSatFix, PointCloud2  # noqa: E402

from .convert import (GeoToEnu, cloud_to_xyz, image_to_bgr, intrinsics_from_info, quat_to_yaw,  # noqa: E402
                      sensor_to_car)

try:
    from fs_msgs.msg import ControlCommand as FsControl, FinishedSignal
except ImportError:                      # pragma: no cover
    FsControl = FinishedSignal = None


class StackNode(Node):
    def __init__(self):
        super(StackNode, self).__init__('ucr_dv_stack')
        if ROOT is None:
            raise RuntimeError('cannot find ucr-dv-stack: set UCR_DV_ROOT to the folder containing ucr_dv/')
        from ucr_dv.core.profile import Profile, load_profile
        from ucr_dv.core.types import CameraImage, EgoMeasurements, GroundTruth, Observation, PointCloud, Pose2D
        from ucr_dv.deploy import flat_params
        from ucr_dv.pipeline import Stack
        self._T = dict(CameraImage=CameraImage, EgoMeasurements=EgoMeasurements, GroundTruth=GroundTruth,
                       Observation=Observation, PointCloud=PointCloud, Pose2D=Pose2D)
        name = self.declare_parameter('profile', 'baseline').value
        path = name
        if not os.path.isfile(path):
            share_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
            for cand in (os.path.join(share_cfg, name), os.path.join(share_cfg, name + '.json'),
                         os.path.join(ROOT, 'data', 'profiles', name + '.json')):
                if os.path.isfile(cand):
                    path = cand
                    break
        self.profile = load_profile(path) if os.path.isfile(path) else Profile(name)
        hz = self.declare_parameter('pipeline_hz', float(self.profile.pipeline_hz)).value
        event = self.declare_parameter('event', 'autocross').value
        self.stack = Stack(self.profile, event=event)
        for key, val in sorted(flat_params(self.profile).items()):
            self.declare_parameter(key, val)
        self.add_on_set_parameters_callback(self._on_params)

        self.lock = threading.Lock()
        self.frames = {}
        self.ego = EgoMeasurements()
        self.truth = None
        self.cam_info = {}
        self.depth = {}
        self.geo = GeoToEnu()
        self.t0 = None
        rig = self.profile.rig
        for s in rig.perception():
            m = s.mount
            mount = (m.x, m.y, m.z, math.radians(m.roll), math.radians(m.pitch), math.radians(m.yaw))
            if s.type == 'lidar':
                self.create_subscription(PointCloud2, '/lidar/' + s.name,
                                         lambda msg, n=s.name, mt=mount: self._cloud(n, mt, msg), qos_profile_sensor_data)
            elif s.type in ('camera', 'stereo'):
                self.create_subscription(Image, '/fsds/{}/image_color'.format(s.name),
                                         lambda msg, n=s.name, mt=mount: self._image(n, mt, msg), qos_profile_sensor_data)
                self.create_subscription(CameraInfo, '/fsds/{}/camera_info'.format(s.name),
                                         lambda msg, n=s.name: self.cam_info.__setitem__(n, intrinsics_from_info(msg)), 10)
                if s.type == 'stereo':
                    self.create_subscription(Image, '/fsds/{}_depth/image_color'.format(s.name),
                                             lambda msg, n=s.name: self.depth.__setitem__(n, image_to_bgr(msg)),
                                             qos_profile_sensor_data)
        self.create_subscription(TwistStamped, '/gss', self._gss, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu', self._imu, qos_profile_sensor_data)
        self.create_subscription(NavSatFix, '/gps', self._gps, qos_profile_sensor_data)
        if self.stack.uses_ground_truth:
            self.create_subscription(Odometry, '/testing_only/odom', self._odom, qos_profile_sensor_data)
        self.pub_cones = self.create_publisher(PoseArray, '/perception/cones', 10)
        self.pub_map = self.create_publisher(PoseArray, '/slam/map', 10)
        self.pub_path = self.create_publisher(PoseArray, '/planning/path', 10)
        self.pub_cmd = self.create_publisher(FsControl, '/control_command', 10) if FsControl else None
        self.pub_fin = self.create_publisher(FinishedSignal, '/signal/finished', 1) if FinishedSignal else None
        if self.pub_cmd is None:
            self.get_logger().warning('fs_msgs not found: no /control_command output (source the FSDS ROS 2 workspace)')
        self.finished_sent = False
        self.timer = self.create_timer(1.0 / float(hz), self._tick)
        self.get_logger().info('profile {} ({}), sensors: {}'.format(
            self.profile.name, path, ', '.join(s.name for s in rig.perception())))

    # ---- inputs ------------------------------------------------------------
    def _now(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = t
        return t - self.t0

    def _cloud(self, name, mount, msg):
        pts = sensor_to_car(cloud_to_xyz(msg), mount)
        with self.lock:
            self.frames[name] = self._T['PointCloud'](name, self._now(), pts, mount[:3])

    def _image(self, name, mount, msg):
        intr = self.cam_info.get(name)
        if intr is None:
            return
        img = image_to_bgr(msg)
        with self.lock:
            self.frames[name] = self._T['CameraImage'](name, self._now(), img, intr, mount,
                                                      depth=self.depth.get(name))

    def _gss(self, msg):
        self.ego.gss_velocity = (msg.twist.linear.x, msg.twist.linear.y)
        self.ego.wheel_speed = math.hypot(msg.twist.linear.x, msg.twist.linear.y)

    def _imu(self, msg):
        self.ego.yaw_rate = msg.angular_velocity.z
        self.ego.accel = (msg.linear_acceleration.x, msg.linear_acceleration.y)

    def _gps(self, msg):
        self.ego.gnss_xy = self.geo(msg.latitude, msg.longitude)

    def _odom(self, msg):
        p = msg.pose.pose
        v = msg.twist.twist.linear
        self.truth = self._T['GroundTruth'](pose=self._T['Pose2D'](p.position.x, p.position.y, quat_to_yaw(p.orientation)),
                                            speed=math.hypot(v.x, v.y), vx=v.x, vy=v.y,
                                            yaw_rate=msg.twist.twist.angular.z)

    def _on_params(self, params):
        for prm in params:
            parts = prm.name.split('.')
            if len(parts) == 2 and parts[0] in self.stack.stages:
                self.stack.set_params(parts[0], {parts[1]: prm.value})
            elif len(parts) == 3 and parts[0] == 'sensor':
                self.stack.set_params('sensor:' + parts[1], {parts[2]: prm.value})
        return SetParametersResult(successful=True)

    # ---- tick ----------------------------------------------------------------
    def _tick(self):
        with self.lock:
            frames, self.frames = self.frames, {}
        ego = self.ego
        obs = self._T['Observation'](t=self._now(), frames=frames,
                                     ego=self._T['EgoMeasurements'](**vars(ego)), truth=self.truth)
        self.ego.gnss_xy = None                    # GNSS fixes are consumed once
        out = self.stack.step(obs)
        stamp = self.get_clock().now().to_msg()
        self.pub_cones.publish(self._poses(out.cones, 'fsds/FSCar', stamp))
        self.pub_map.publish(self._poses(out.map_world, 'map', stamp))
        self.pub_path.publish(self._poses(np.column_stack([out.path, np.zeros(len(out.path))]) if len(out.path)
                                          else np.zeros((0, 3)), 'fsds/FSCar', stamp))
        c = out.command
        if self.pub_cmd is not None:
            msg = FsControl()
            msg.header.stamp = stamp
            msg.throttle = float(c.throttle)
            msg.brake = float(c.brake)
            msg.steering = float(max(-1.0, min(1.0, -c.steering / self.stack.vehicle.max_steer)))   # FSDS: +right
            self.pub_cmd.publish(msg)
        if c.finished and not self.finished_sent and self.pub_fin is not None:
            fin = FinishedSignal()
            fin.placeholder = True
            self.pub_fin.publish(fin)
            self.finished_sent = True

    @staticmethod
    def _poses(arr, frame, stamp):
        msg = PoseArray()
        msg.header.frame_id = frame
        msg.header.stamp = stamp
        for row in np.asarray(arr):
            p = Pose()
            p.position.x, p.position.y = float(row[0]), float(row[1])
            p.position.z = float(row[2]) if len(row) > 2 else 0.0
            p.orientation.w = 1.0
            msg.poses.append(p)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = StackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
