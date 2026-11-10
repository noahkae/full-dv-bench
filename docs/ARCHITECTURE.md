# Architecture

## Data flow (one pipeline tick, `ucr_dv/pipeline.py: Stack.step`)

```
Backend.advance(dt) -> Observation
    frames: {sensor: ConeDetections | PointCloud | CameraImage}   (only sensors that fired)
    ego:    EgoMeasurements (wheel speed, GSS, yaw rate, accel, GNSS xy, steering)
    truth:  GroundTruth (metrics only - or odometry 'ground_truth', which marks the run uses_gt)

Stack.step(obs, dt)
  1 odometry.update(obs, dt)                         -> Pose2D (world/spawn frame)
  2 per sensor: detection frame passes through;      -> cones (N,4) car frame + covariance
    raw frame -> registered lidar/camera detector       (stored in the world frame, so older
                                                         10 Hz scans are motion compensated)
  3 fusion.fuse(cones_by_sensor, cov_by_sensor)      -> fused cones (car frame)
  4 mapping.localize(fused, pose) -> pose fix (ICP)  -> odometry.correct(pose)
    mapping.update(fused, pose); mapping.local(...)  -> cones for planning (map + fresh)
  5 mission.update(fused, distance, v, t)            -> laps, stopping, finished, speed cap
  6 planner.plan(local, ctx)                         -> path (car frame, starts at the car)
  7 speed.profile(path, v, ctx)                      -> profile + target speed
  8 lateral.steer(path, v, ctx)                      -> road-wheel angle (rad, +left)
  9 longitudinal.command(target, v, dt)              -> throttle, brake (vehicle pedal map)
-> StackOutput(command, cones, map, path, timings_ms, debug)

Backend.apply(command)
```

## Stage contracts

| stage | method | in -> out |
|---|---|---|
| lidar | `detect(PointCloud)` | car-frame points -> `(cones, debug)` |
| camera | `detect(CameraImage)` | BGR image (+depth), intrinsics, mount -> `(cones, debug)` |
| fusion | `fuse(cones_by_sensor, cov_by_sensor)` | dicts -> cones |
| odometry | `update(obs, dt)`, `correct(pose)`, `.speed`, `.distance` | -> Pose2D |
| mapping | `update(cones, pose)`, `local(cones, pose)`, optional `localize(cones, pose)` | -> world map / planning cones |
| mission | `update(cones, distance, v, t)`, optional `template()` | -> MissionState |
| planner | `plan(cones, ctx)` | car-frame cones -> car-frame path |
| speed | `profile(path, v, ctx)` | -> (per-point speeds, target) |
| lateral | `steer(path, v, ctx)`, `.target` | -> rad (+left) |
| longitudinal | `command(v_target, v, dt, a_ff)` | -> (throttle, brake, a_cmd) |

`ctx` carries `pose`, `speed`, `template`, `mission`, `speed_cap`, `vehicle`, `dt`.
Every component subclasses `registry.Component`: `params`, `set_params()` (live
tuning), `reset()`.

## Backends (`ucr_dv/backends/`)

`Backend`: `load(track, profile, vehicle, seed, stack)`, `reset(seed)`,
`apply(cmd)`, `advance(dt) -> Observation`, `status() -> EpisodeStatus`,
`car_state()`, `close()`, plus `Capabilities` (lockstep, realtime_only,
raw_lidar, raw_camera, ...) that the GUI and runner respect.

- `TimelessBackend` - 2D kinematic bicycle + grip limit + steering delay/lag/slew,
  drive force from the vehicle file, fsds_lite `Referee` for timing.
- `LiteBackend` - fsds_lite `Simulator` built from the rig (no settings.json),
  `sim.step(n)` lockstep, `LockstepCameraRenderer` (never drops, `wait_idle`),
  RNGs reseeded per run for determinism.
- `SensorSynth` (common.py) - detection-level frames + ego sensors from truth;
  used by timeless always, by 3D for `detection`-fidelity sensors, by AC later.
- `CourseMonitor` - cross-track error, progress, off-course (FS rule: all four
  wheels out), lost-car DNF, on the track's reference line (`tracks.track_reference`).

## Threading (GUI)

`RunController` runs `EpisodeRunner.run()` on a worker thread. The worker never
touches Qt; the GUI polls `runner.latest` at 30 Hz and posts commands (pause,
step, stop, speed, live params) through a queue. The fsds_lite `SimPanel`
renders from `sim.snapshot()` on the GUI thread; cameras render on their own
EGL thread. `configure_surface_format()` and `configure_qt_display()` run before
the `QApplication` exists (`app/main.py`).

## What was ported, and fixed

From ucr-dv-timeless: the kinematic model, steering actuator (delay bug fixed:
`deque.pop()` delayed 1 sample), the detector noise model as
`timeless_classic` (spurious-cone x/y swap fixed), greedy pairing planner,
P-steer controller, PID, grip-limited speed profile, map JSON format (moved to
the car-at-origin frame), FS scoring. Collision box now uses the car's real
footprint; seeded RNG everywhere; no Python >= 3.10 syntax.

From driverless-systems: the LiDAR grid detector (`lidar_grid`), the gate
planner (`gates`), the cone-merge mapping idea (`cone_map`), the topic
contract (ROS node), and fsds_lite itself (vendored).
