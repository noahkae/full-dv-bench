"""
AirSim-compatible msgpack-rpc server (default 127.0.0.1:41451).

Implements the subset of the FSDS/AirSim car API that our stack and the
FSDS Python client use, so `airsim.CarClient` / `fsds.FSDSClient` connect to
fsds_lite exactly as they would to FSDS.exe:

    ping, getServerVersion, getMinRequiredClientVersion, getMinRequiredServerVersion
    enableApiControl, isApiControlEnabled, armDisarm
    setCarControls, getCarControls, getCarState, reset
    simGetVehiclePose, simSetVehiclePose, simGetGroundTruthKinematics
    getLidarData, getImuData, getGpsData, getGroundSpeedSensorData
    getRefereeState, getSettingsString, simPause, simIsPaused, simContinueForTime
    simGetImages, simGetImage   (scene and depth images, if cameras are rendered)
    simGetWheelStates

Unknown methods return an msgpack-rpc error (the client raises) instead of
hanging. Like FSDS, positions are ENU relative to spawn - not NED.

Wire format: request [0, msgid, method, params] -> response [1, msgid, error, result];
notification [2, method, params] gets no response.
"""
import json
import logging
import socket
import socketserver
import struct
import threading
import zlib

import msgpack
import numpy as np

from .geometry import quat_to_yaw, yaw_to_quat
from .simulator import Controls

log = logging.getLogger('fsds_lite.rpc')

SERVER_VERSION = 1


def vec(x=0.0, y=0.0, z=0.0):
    return {'x_val': float(x), 'y_val': float(y), 'z_val': float(z)}


def quat(q):
    return {'w_val': float(q[3]), 'x_val': float(q[0]), 'y_val': float(q[1]), 'z_val': float(q[2])}


def pose(x, y, z, yaw):
    return {'position': vec(x, y, z), 'orientation': quat(yaw_to_quat(yaw))}


def encode_png(rgb):
    """Minimal PNG encoder (8-bit RGB) so simGetImage needs no imaging library."""
    h, w, _ = rgb.shape
    raw = b''.join(b'\x00' + rgb[i].tobytes() for i in range(h))

    def chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw, 1)) + chunk(b'IEND', b''))


class CarApi(object):
    """Method table. Each public method is callable over RPC by name."""

    def __init__(self, sim, settings_text=''):
        self.sim = sim
        self.settings_text = settings_text

    # --- connection / version ---
    def ping(self):
        return True

    def getServerVersion(self):
        return SERVER_VERSION

    def getMinRequiredClientVersion(self):
        return SERVER_VERSION

    def getMinRequiredServerVersion(self):
        return SERVER_VERSION

    # --- api control ---
    def enableApiControl(self, is_enabled, vehicle_name=''):
        self.sim.enable_api_control(is_enabled)
        log.info('API control -> %s', bool(is_enabled))

    def isApiControlEnabled(self, vehicle_name=''):
        return self.sim.api_control

    def armDisarm(self, arm, vehicle_name=''):
        return True

    def reset(self):
        self.sim.reset()
        log.info('reset')

    # --- car ---
    def setCarControls(self, controls, vehicle_name=''):
        c = Controls(
            throttle=float(controls.get('throttle', 0.0)),
            steering=float(controls.get('steering', 0.0)),
            brake=float(controls.get('brake', 0.0)),
            handbrake=bool(controls.get('handbrake', False)),
        )
        if not self.sim.set_api_controls(c):
            log.debug('setCarControls ignored: API control is disabled')

    def getCarControls(self, vehicle_name=''):
        c = self.sim.active_controls()
        return {'throttle': c.throttle, 'steering': c.steering, 'brake': c.brake,
                'handbrake': c.handbrake, 'is_manual_gear': False, 'manual_gear': 0,
                'gear_immediate': True}

    def _kinematics(self):
        with self.sim.lock:
            s = self.sim.vehicle.state
            vxw, vyw = s.world_velocity()
            return {
                'position': vec(s.x, s.y, 0.0),
                'orientation': quat(yaw_to_quat(s.yaw)),
                'linear_velocity': vec(vxw, vyw, 0.0),
                'angular_velocity': vec(0.0, 0.0, s.r),
                'linear_acceleration': vec(s.ax, s.ay, 0.0),
                'angular_acceleration': vec(),
            }

    def getCarState(self, vehicle_name=''):
        k = self._kinematics()
        return {
            'speed': self.sim.vehicle.state.speed,
            'gear': self.sim.vehicle.state.gear, 'rpm': self.sim.vehicle.state.rpm,
            'maxrpm': self.sim.vehicle.p.max_rpm, 'handbrake': False,
            'collision': {
                'has_collided': False, 'normal': vec(), 'impact_point': vec(),
                'position': vec(), 'penetration_depth': 0.0, 'time_stamp': 0,
                'object_name': '', 'object_id': -1,
            },
            'kinematics_estimated': k,
            'timestamp': int(self.sim.stamp() * 1e9),
        }

    def simGetGroundTruthKinematics(self, vehicle_name=''):
        return self._kinematics()

    def simGetVehiclePose(self, vehicle_name=''):
        s = self.sim.vehicle.state
        return pose(s.x, s.y, 0.0, s.yaw)

    def simSetVehiclePose(self, pose_in, ignore_collision=True, vehicle_name=''):
        p, o = pose_in['position'], pose_in['orientation']
        yaw = quat_to_yaw(o['x_val'], o['y_val'], o['z_val'], o['w_val'])
        self.sim.teleport(p['x_val'], p['y_val'], yaw)

    # --- sensors (latest sample) ---
    def _latest(self, name, default_kind):
        latest = self.sim.latest
        if name and name in latest:
            return latest[name]
        for sample in reversed(list(latest.values())):
            if type(sample).__name__ == default_kind:
                return sample
        raise RuntimeError('no {} data yet'.format(name or default_kind))

    def getLidarData(self, lidar_name='', vehicle_name=''):
        s = self._latest(lidar_name, 'LidarSample')
        return {'point_cloud': s.points.reshape(-1).tolist(),
                'time_stamp': int(s.stamp * 1e9),
                'pose': pose(*s.pose)}

    def getImuData(self, imu_name='', vehicle_name=''):
        s = self._latest(imu_name, 'ImuSample')
        return {'time_stamp': int(s.stamp * 1e9), 'orientation': quat(s.orientation),
                'angular_velocity': vec(*s.angular_velocity),
                'linear_acceleration': vec(*s.linear_acceleration)}

    def getGpsData(self, gps_name='', vehicle_name=''):
        s = self._latest(gps_name, 'GpsSample')
        return {'time_stamp': int(s.stamp * 1e9), 'is_valid': True, 'gnss': {
            'geo_point': {'latitude': s.latitude, 'longitude': s.longitude, 'altitude': s.altitude},
            'eph': s.eph, 'epv': s.epv, 'velocity': vec(*s.velocity),
            'fix_type': 3, 'time_utc': int(s.stamp * 1e3)}}

    def getGroundSpeedSensorData(self, vehicle_name=''):
        s = self._latest('', 'GssSample')
        return {'time_stamp': int(s.stamp * 1e9), 'linear_velocity': vec(*s.linear_velocity),
                'angular_velocity': vec(*s.angular_velocity)}

    def simGetImages(self, requests, vehicle_name=''):
        """Latest frame of each requested camera.

        Scene cameras: compress=True gives PNG, False raw BGR bytes. Depth
        cameras (ImageType 1/2 in settings.json): pixels_as_float=True gives
        metres in image_data_float, like AirSim. The image type a camera
        produces is fixed by its settings, as in FSDS.
        """
        out = []
        for r in requests:
            name = str(r.get('camera_name', ''))
            s = self.sim.latest.get(name)
            if s is None or type(s).__name__ != 'CameraSample':
                raise RuntimeError('no image from camera {!r} yet (are cameras enabled?)'.format(name))
            h, w = s.image.shape[:2]
            is_depth = s.image.dtype == np.float32
            if is_depth != (r.get('image_type', 0) in (1, 2)):
                raise ValueError('camera {!r} renders {} images'.format(
                    name, 'depth (image_type 1/2)' if is_depth else 'Scene (image_type 0)'))
            response = {
                'image_data_uint8': b'', 'image_data_float': [],
                'camera_position': vec(*s.position), 'camera_orientation': quat(s.orientation),
                'time_stamp': int(s.stamp * 1e9), 'message': '', 'pixels_as_float': is_depth,
                'compress': False, 'width': w, 'height': h, 'image_type': r.get('image_type', 0),
            }
            if is_depth:
                response['image_data_float'] = np.nan_to_num(s.image, posinf=1e4).reshape(-1).tolist()
            else:
                compress = bool(r.get('compress', True))
                response['compress'] = compress
                response['image_data_uint8'] = encode_png(s.image[:, :, ::-1]) if compress else s.image.tobytes()
            out.append(response)
        return out

    def simGetWheelStates(self, vehicle_name=''):
        s = self.sim.latest.get('wheel_states')
        if s is None:
            raise RuntimeError('no wheel states yet')
        wheels = {}
        for k, key in enumerate(('fl', 'fr', 'rl', 'rr')):
            wheels[key] = {'rpm': s.rpm[k], 'rotation_angle': s.rotation_angle[k],
                           'steering_angle': s.steering_angle[k]}
        wheels['time_stamp'] = int(s.stamp * 1e9)
        return wheels

    def simGetImage(self, camera_name, image_type=0, vehicle_name=''):
        return self.simGetImages([{'camera_name': camera_name, 'image_type': image_type,
                                   'compress': True}])[0]['image_data_uint8']

    def getRefereeState(self):
        t = self.sim.track
        with self.sim.lock:
            ref = self.sim.referee
            hit, laps = int(ref.doo), [float(x) for x in ref.laps]
        return {'doo_counter': hit, 'laps': laps,
                'initial_position': {'x': 0.0, 'y': 0.0},
                'cones': [{'x': float(x), 'y': float(y), 'color': c}
                          for (x, y), c in zip(t.positions, t.colors)]}

    def getSettingsString(self):
        return self.settings_text

    # --- time ---
    def simPause(self, is_paused):
        self.sim.set_paused(is_paused)

    def simIsPaused(self):
        return self.sim.paused

    def simContinueForTime(self, seconds):
        self.sim.continue_for(seconds)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        api = self.server.api
        sock = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        unpacker = msgpack.Unpacker(raw=False)
        packer = msgpack.Packer(use_bin_type=True)
        peer = '{}:{}'.format(*self.client_address[:2])
        log.info('client connected %s', peer)
        try:
            while True:
                data = sock.recv(65536)
                if not data:
                    break
                unpacker.feed(data)
                for msg in unpacker:
                    reply = self._dispatch(api, msg)
                    if reply is not None:
                        sock.sendall(packer.pack(reply))
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            log.info('client disconnected %s', peer)

    @staticmethod
    def _dispatch(api, msg):
        if not isinstance(msg, (list, tuple)) or not msg:
            return None
        if msg[0] == 0 and len(msg) == 4:
            _, msgid, method, params = msg
        elif msg[0] == 2 and len(msg) == 3:
            msgid, (_, method, params) = None, msg
        else:
            return None
        fn = getattr(api, method, None) if not str(method).startswith('_') else None
        if fn is None:
            log.warning('unsupported RPC method %r', method)
            err, result = 'fsds_lite does not implement {!r}'.format(method), None
        else:
            try:
                err, result = None, fn(*params)
            except Exception as e:  # report to the client, keep serving
                log.exception('RPC %s failed', method)
                err, result = '{}: {}'.format(type(e).__name__, e), None
        if msgid is None:
            return None
        return [1, msgid, err, result]


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True       # no stale-socket wait after a restart
    daemon_threads = True


class RpcServer(object):
    def __init__(self, sim, host='127.0.0.1', port=41451, settings_text=''):
        self.server = _Server((host, port), _Handler)
        self.server.api = CarApi(sim, settings_text)
        self.address = self.server.server_address
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.server.serve_forever, name='rpc', daemon=True)
        self.thread.start()
        log.info('AirSim RPC listening on %s:%d', *self.address)
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    # allow `with RpcServer(...) as srv:`
    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    @staticmethod
    def settings_text_from(path):
        try:
            with open(path) as f:
                return json.dumps(json.load(f))
        except (OSError, ValueError):
            return ''
