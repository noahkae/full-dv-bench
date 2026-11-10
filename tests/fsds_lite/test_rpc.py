"""
RPC compatibility. The raw-msgpack tests always run; the airsim-client test
runs wherever the real `airsim` package imports (e.g. the Galactic env).
"""
import socket

import msgpack
import pytest

from fsds_lite.rpc_server import RpcServer
from fsds_lite.simulator import Controls


@pytest.fixture
def server(sim):
    with RpcServer(sim, port=0) as srv:
        yield srv


class RawClient(object):
    def __init__(self, addr):
        self.sock = socket.create_connection(addr, timeout=3)
        self.unpacker = msgpack.Unpacker(raw=False)
        self.msgid = 0

    def call(self, method, *params):
        self.msgid += 1
        self.sock.sendall(msgpack.packb([0, self.msgid, method, list(params)], use_bin_type=True))
        while True:
            for msg in self.unpacker:
                assert msg[0] == 1 and msg[1] == self.msgid
                if msg[2] is not None:
                    raise RuntimeError(msg[2])
                return msg[3]
            self.unpacker.feed(self.sock.recv(65536))


def test_raw_protocol_round_trip(sim, server):
    c = RawClient(server.address)
    assert c.call('ping') is True
    assert c.call('getServerVersion') == 1
    c.call('setCarControls', {'throttle': 1.0, 'steering': 0.0, 'brake': 0.0}, 'FSCar')
    assert sim.api_controls.throttle == 0.0           # ignored: API control off
    c.call('enableApiControl', True, 'FSCar')
    assert c.call('isApiControlEnabled', 'FSCar') is True
    c.call('setCarControls', {'throttle': 1.0, 'steering': 0.2, 'brake': 0.0}, 'FSCar')
    sim.step(500)
    state = c.call('getCarState', 'FSCar')
    assert state['speed'] > 0.5
    assert state['kinematics_estimated']['position']['x_val'] > 0.0
    c.call('reset')
    assert c.call('isApiControlEnabled', 'FSCar') is False


def test_unknown_method_errors_instead_of_hanging(server):
    c = RawClient(server.address)
    with pytest.raises(RuntimeError, match='does not implement'):
        c.call('simSetWeatherParameter', 1, 0.5)
    assert c.call('ping') is True                     # connection survives


def test_lidar_over_rpc(sim, server):
    sim.step(150)
    c = RawClient(server.address)
    data = c.call('getLidarData', 'Lidar1', 'FSCar')
    assert len(data['point_cloud']) % 3 == 0 and len(data['point_cloud']) > 0


def test_real_airsim_client(sim, server):
    airsim = pytest.importorskip('airsim')
    client = airsim.CarClient(ip=server.address[0], port=server.address[1])
    client.confirmConnection()
    client.enableApiControl(True)
    c = airsim.CarControls()
    c.throttle, c.steering, c.brake = 0.5, 0.0, 0.0
    client.setCarControls(c)
    sim.step(1000)
    state = client.getCarState()
    assert state.speed > 1.0
    assert state.kinematics_estimated.position.x_val > 0.0
    client.reset()
    assert not client.isApiControlEnabled()


def test_referee_and_wheel_states_over_rpc(sim, server):
    sim.enable_api_control(True)
    sim.set_api_controls(Controls(throttle=1.0))
    while not sim.referee.mission_complete and sim.sim_time < 15.0:
        sim.step(10)
    sim.set_api_controls(Controls(brake=0.3, throttle=0.0))
    sim.step(200)
    c = RawClient(server.address)
    ref = c.call('getRefereeState')
    assert len(ref['laps']) == 1 and ref['doo_counter'] == 0
    ws = c.call('simGetWheelStates', 'FSCar')
    assert ws['rl']['rpm'] > 0 and set(ws) >= {'fl', 'fr', 'rl', 'rr', 'time_stamp'}
