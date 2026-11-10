"""
Run fsds_lite.

    python3 -m fsds_lite                       # 3D + top-down window, RPC, ROS (if sourced), cameras
    python3 -m fsds_lite --view 2d             # top-down only (no OpenGL needed)
    python3 -m fsds_lite --headless            # no window
    python3 -m fsds_lite --no-cameras          # skip camera rendering (saves GPU)
    python3 -m fsds_lite --track skidpad       # FS events: acceleration (default), skidpad,
                                               # autocross, trackdrive
    python3 -m fsds_lite --track my_map.csv    # any FSDS custom-map CSV
    python3 -m fsds_lite --no-ros              # RPC + window only (works without ROS)

Run from the sim/ directory, or put sim/ on PYTHONPATH.
"""
import argparse
import logging
import os
import signal
import sys
import threading

from .host import SimHost

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = logging.getLogger('fsds_lite')


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog='fsds_lite', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--track', default='acceleration',
                    help='acceleration | skidpad | autocross | trackdrive, a bundled track '
                         '(droneport), or an FSDS CSV path (default: acceleration)')
    ap.add_argument('--no-scenery', action='store_true',
                    help='no trackside objects (barriers, tyre stacks, poles, spare cones)')
    ap.add_argument('--competition-mode', action='store_true',
                    help='hide /testing_only topics (ground truth), like the FSDS bridge')
    ap.add_argument('--settings', default=os.path.join(HERE, 'data', 'vehicle', 'settings.json'),
                    help='FSDS settings.json defining the sensors')
    ap.add_argument('--vehicle', default=os.path.join(HERE, 'data', 'vehicle', 'vehicle.json'),
                    help='vehicle parameter JSON')
    ap.add_argument('--headless', action='store_true', help='no window')
    ap.add_argument('--view', choices=['3d', '2d'], default='3d',
                    help='3d adds a moderngl tab next to the top-down view (default: 3d)')
    ap.add_argument('--ui-scale', type=float, default=None,
                    help='window UI scale (default: auto - native Wayland scaling, else GDK_SCALE)')
    ap.add_argument('--no-cameras', action='store_true', help='do not render camera images')
    ap.add_argument('--depth-cameras', action='store_true',
                    help='also render a DepthPerspective twin of each camera (<name>_depth, 32FC1)')
    ap.add_argument('--no-imu-noise', action='store_true', help='perfect IMU (default: AirSim noise)')
    ap.add_argument('--traction-control', action='store_true', help='limit rear wheelspin')
    ap.add_argument('--camera-hz', type=float, default=30.0, help='camera frame rate (default 30)')
    ap.add_argument('--msaa', type=int, default=4, help='camera anti-aliasing samples (0 = off)')
    ap.add_argument('--no-ros', action='store_true', help='do not start the ROS 2 bridge')
    ap.add_argument('--no-rpc', action='store_true', help='do not start the AirSim RPC server')
    ap.add_argument('--rpc-host', default='127.0.0.1')
    ap.add_argument('--rpc-port', type=int, default=41451)
    ap.add_argument('--namespace', default='', help='ROS namespace for sensor topics')
    ap.add_argument('--gss-type', choices=['twist', 'covariance'], default='twist',
                    help='/gss message type: TwistStamped (contract) or TwistWithCovarianceStamped')
    ap.add_argument('--mission', default=None,
                    help='mission name in /signal/go (default: from the track)')
    ap.add_argument('--realtime-factor', type=float, default=1.0,
                    help='sim seconds per wall second (e.g. 2.0 = double speed)')
    ap.add_argument('--physics-hz', type=float, default=1000.0)
    ap.add_argument('--seed', type=int, default=None, help='RNG seed for sensor noise')
    ap.add_argument('-v', '--verbose', action='store_true')
    return ap.parse_args(argv)


def isolate_shader_cache():
    """Give a container's Mesa its own shader cache. Call before any GL context.

    distrobox/toolbox share $HOME, so by default the container's (older) Mesa
    and the host's - including the compositor's - both use
    ~/.cache/mesa_shader_cache and memory-map the same `index` file. When the
    container's Mesa rewrites it, the host compositor can die with SIGBUS inside
    its shader compiler (seen as Hyprland crashing when the workbench closes).
    Mesa before 21.3 (Ubuntu 20.04 ships 21.2) only reads the old name,
    MESA_GLSL_CACHE_DIR, so both are set. An explicit value from the user wins.
    """
    box = os.environ.get('CONTAINER_ID') or ('container' if os.path.exists('/run/.containerenv')
                                             else None)
    if not box:
        return
    cache = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    path = (os.environ.get('MESA_SHADER_CACHE_DIR') or os.environ.get('MESA_GLSL_CACHE_DIR')
            or os.path.join(cache, 'mesa_shader_cache_' + box))
    os.environ.setdefault('MESA_SHADER_CACHE_DIR', path)
    os.environ.setdefault('MESA_GLSL_CACHE_DIR', path)


def configure_qt_display(ui_scale=None):
    """Make the window come out the right size on HiDPI screens.

    Must run before the QApplication is created. Qt5 defaults to XWayland
    on a Wayland session, which can't see the monitor's scale, so at 2x
    everything renders at half size. Prefer native Wayland when the plugin is
    there (it reports the real scale); on X11 fall back to GDK_SCALE.
    An explicit QT_QPA_PLATFORM / QT_SCALE_FACTOR from the user always wins.
    """
    from PyQt5 import QtCore
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    if ui_scale:
        os.environ['QT_SCALE_FACTOR'] = str(ui_scale)
    if 'QT_QPA_PLATFORM' not in os.environ and os.environ.get('WAYLAND_DISPLAY'):
        plugins = QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)
        if os.path.exists(os.path.join(plugins, 'platforms', 'libqwayland-egl.so')):
            os.environ['QT_QPA_PLATFORM'] = 'wayland'
    if (os.environ.get('QT_QPA_PLATFORM', 'xcb').startswith('xcb')
            and 'QT_SCALE_FACTOR' not in os.environ):
        try:
            gdk = float(os.environ.get('GDK_SCALE', '1'))
        except ValueError:
            gdk = 1.0
        if gdk > 1.0:
            os.environ['QT_SCALE_FACTOR'] = str(gdk)


def main(argv=None):
    args = parse_args(argv)
    # Hand the GIL between threads every 1 ms instead of 5 ms, so the physics
    # thread isn't starved in bursts while the camera thread is busy.
    sys.setswitchinterval(0.001)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='[%(asctime)s] %(name)s: %(message)s', datefmt='%H:%M:%S')

    isolate_shader_cache()               # before the camera renderer's GL context
    host = SimHost(args)
    try:
        host.start()
    except OSError:
        return 1
    sim, sensors = host.sim, host.sensors

    def shutdown(*_):
        host.stop()

    try:
        if args.headless:
            stop = threading.Event()
            signal.signal(signal.SIGINT, lambda *_: stop.set())
            signal.signal(signal.SIGTERM, lambda *_: stop.set())
            log.info('running headless - Ctrl-C to quit')
            while not stop.wait(0.5):
                pass
        else:
            configure_qt_display(args.ui_scale)
            from PyQt5 import QtCore, QtWidgets
            use_3d = args.view == '3d'
            if use_3d:
                try:
                    from .render.viewer3d import configure_surface_format
                    configure_surface_format()
                except ImportError as e:
                    log.warning('3D view disabled (%s) - pip install moderngl', e)
                    use_3d = False
            app = QtWidgets.QApplication(sys.argv)
            log.info('window: Qt %s platform, UI scale %.2g', app.platformName(),
                     app.primaryScreen().devicePixelRatio())
            signal.signal(signal.SIGINT, lambda *_: app.quit())
            from .viewer import SimWindow
            win = SimWindow(sim, sensors, use_3d=use_3d)
            win.show()
            # let Python see Ctrl-C while Qt owns the main loop
            tick = QtCore.QTimer()
            tick.timeout.connect(lambda: None)
            tick.start(200)
            app.exec_()
    finally:
        shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
