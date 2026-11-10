"""
fsds_lite - a lightweight stand-in for the Formula Student Driverless Simulator.

The core (vehicle, sensors, track, simulator) is plain Python + numpy with no
ROS or Qt dependency, so it can be tested anywhere and embedded in the
workbench later. Adapters sit on top:

    rpc_server  - AirSim-compatible msgpack-rpc on :41451 (control_node, workbench)
    ros_bridge  - publishes the same topics as fsds_ros2_bridge (needs rclpy)
    viewer      - top-down PyQt5 view with keyboard driving (needs PyQt5)

Must stay Python 3.8 compatible (ROS 2 Galactic on Ubuntu 20.04).
"""

__version__ = "0.1.0"
