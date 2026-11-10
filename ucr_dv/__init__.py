"""
ucr_dv - the UCalgary Racing driverless development stack.

One algorithm library, several test benches:

    timeless      fast deterministic 2D bench (sim time, faster than real time)
    lite          3D bench: the vendored fsds_lite simulator (ray-cast LiDAR,
                  rendered cameras, 4-wheel dynamics), run in lockstep
    assetto_corsa planned final bench (interface stub only, see docs/ASSETTO_CORSA.md)

The same *stack profile* (sensor rig + algorithm choices + parameters, one
JSON file) runs on every bench, so a stack is developed in timeless, promoted
to 3D, and later to Assetto Corsa without changing code.

Pure Python + numpy, Python 3.8 compatible (ROS 2 Galactic / Ubuntu 20.04).
"""

__version__ = "0.1.0"
