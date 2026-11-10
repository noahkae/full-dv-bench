"""
SimHost: builds a Simulator and all of its services from parsed args.

One place owns the wiring (track, sensors, vehicle, cameras, RPC, ROS bridge,
physics thread), so `python3 -m fsds_lite` and the workbench's `--lite` mode
start exactly the same sim. The workbench embeds it in-process:

    host = SimHost(parse_args(['--track', 'skidpad']), own_rclpy=False).start()
    host.sim            # the Simulator: reset(), enable_api_control(), ...
    host.sensors        # for SimView3D
    host.stop()

No Qt in here; the windows live in viewer.py / render/viewer3d.py.
"""
import dataclasses
import logging
import threading

from .rpc_server import RpcServer
from .sensors import load_settings
from .simulator import Simulator
from .track import load_track
from .vehicle import VehicleParams

log = logging.getLogger('fsds_lite')


class SimHost(object):
    def __init__(self, args, own_rclpy=True):
        """`own_rclpy=False` when the host process already called rclpy.init()
        (the workbench): stopping the sim then leaves ROS alone."""
        self.args = args
        self.own_rclpy = own_rclpy
        self.services = []
        self.physics = None
        self.sim = None
        self.sensors = None
        self.track = None
        self.ros = None
        self.rpc = None

    @property
    def mission(self):
        return self.track.mission

    def start(self):
        """Build and start everything. Raises OSError if the RPC port is taken."""
        args = self.args
        self.track = track = load_track(args.track, scenery=not args.no_scenery)
        self.sensors = sensors = load_settings(args.settings)
        if args.depth_cameras:
            sensors.cameras += [dataclasses.replace(c, name=c.name + '_depth', image_type=2)
                                for c in sensors.cameras if not c.is_depth]
        for cam in sensors.cameras:
            cam.rate_hz = args.camera_hz
        if args.no_cameras:
            sensors.cameras = []
        if args.no_imu_noise and sensors.imu:
            sensors.imu.noise = False
        params = VehicleParams.from_json(args.vehicle)
        if args.traction_control:
            params.traction_control = True
        self.sim = sim = Simulator(track, sensors, params, physics_hz=args.physics_hz,
                                   seed=args.seed)
        sim.realtime_factor = args.realtime_factor
        log.info('track %s (%s, %d lap%s): %d cones, %d trackside objects | lidars: %s | '
                 'cameras: %s | imu %s gps %s gss %s', track.name, track.mission, track.laps,
                 's' if track.laps > 1 else '', len(track), len(track.obstacles),
                 ', '.join(l.name for l in sensors.lidars) or 'none',
                 ', '.join(c.name for c in sensors.cameras) or 'none',
                 bool(sensors.imu), bool(sensors.gps), bool(sensors.gss))

        if sensors.cameras:
            try:
                from .render.camera_sensor import CameraRenderer
                self.services.append(CameraRenderer(
                    sim, sensors.cameras, samples=args.msaa,
                    sensor_mounts=[(l.x, l.y, l.z) for l in sensors.lidars]).start())
            except (ImportError, RuntimeError) as e:
                log.warning('cameras disabled - %s (pip install moderngl, or pass --no-cameras)', e)
                sensors.cameras = []

        if not args.no_rpc:
            try:
                self.rpc = RpcServer(sim, args.rpc_host, args.rpc_port,
                                     RpcServer.settings_text_from(args.settings)).start()
                self.services.append(self.rpc)
            except OSError as e:
                log.error('cannot bind RPC %s:%d (%s). Is FSDS or another fsds_lite running?',
                          args.rpc_host, args.rpc_port, e)
                self._stop_services()
                raise
        if not args.no_ros:
            try:
                from .ros_bridge import RosBridge
            except ImportError as e:
                log.warning('ROS 2 bridge disabled - could not import rclpy (%s). '
                            'Source /opt/ros/<distro>/setup.bash, or pass --no-ros.', e)
            else:
                self.ros = RosBridge(sim, sensors, args.namespace, args.gss_type, args.mission,
                                     publish_cameras=bool(sensors.cameras),
                                     competition_mode=args.competition_mode,
                                     own_rclpy=self.own_rclpy).start()
                self.services.append(self.ros)

        self.physics = threading.Thread(target=sim.run, name='physics', daemon=True)
        self.physics.start()
        return self

    def _stop_services(self):
        for s in reversed(self.services):
            try:
                s.stop()
            except Exception:
                pass
        self.services = []

    def stop(self):
        if self.sim is not None:
            self.sim.stop()
        self._stop_services()
