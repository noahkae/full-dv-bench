# Assetto Corsa bench (planned)

The third rung of the ladder: timeless 2D -> fsds_lite 3D -> **Assetto Corsa**.
Nothing here is implemented yet; `ucr_dv/backends/assetto_corsa.py` is a stub
whose capabilities and method contract are fixed, so the rest of the stack
(and the GUI) already knows how to drive it.

## Why AC is different from the other benches

| | timeless | fsds_lite 3D | Assetto Corsa |
|---|---|---|---|
| clock | own, lockstep | own, lockstep | **real time only** (it is a game) |
| determinism | yes | yes | no |
| cones | from the track file | from the track file | **none** - overlaid from a cone map |
| perception | detection models | ray-cast / rendered, or detection models | detection models on the overlay (screen-capture camera: later) |
| controls | throttle/brake/steer | throttle/brake/steer | virtual gamepad (steer lock + pedals) |
| frame | ENU | ENU | **left-handed, Y up** (x, z ground plane) |

## Readiness checklist (what the backend must implement)

- [ ] **Realtime path.** `Capabilities(lockstep=False, deterministic=False, realtime_only=True)`.
      `advance(dt)` waits for the wall clock and returns the latest telemetry; the runner already
      paces itself from `realtime_factor` and never queues late commands.
- [ ] **Telemetry in.** AC shared memory pages `acpmf_physics` (speed, velocity, accG, wheel slip,
      steerAngle, gas/brake, ~300 Hz), `acpmf_graphics` (car coordinates, lap times, status) and
      `acpmf_static` (car/track names, max rpm). Windows only: read with `mmap` on the AC PC, or run a
      tiny UDP forwarder there and receive it in WSL/Linux. Map to `EgoMeasurements` + `GroundTruth`.
- [ ] **Frames.** One `ac_to_enu(x, y, z)` (AC: x right, y up, z forward, left-handed) plus a
      spawn transform so the start pose is the origin facing +x, like every other track. Unit-test
      it with two known poses.
- [ ] **Cones.** AC tracks have no FS cones. A scenario = AC track + a cone map registered to it
      (generate from the AI line +/- half width, or lay them in a CSV in AC coordinates). The cone map
      becomes an fsds_lite `Track`, so `SensorSynth`, `CourseMonitor` and the `Referee` work unchanged.
- [ ] **Actuation.** Virtual controller (vJoy / ViGEm on Windows) or a Python app plugin.
      `ControlCommand.steering` (road-wheel rad, +left) -> AC steer input via the car's steer lock and
      ratio from `acpmf_static`; throttle/brake pass through. No speed mode: the stack's pedal map
      (`VehicleSpec.accel_to_pedals`) needs a vehicle file approximating the AC car.
- [ ] **Referee.** Same `Referee` on the overlaid timing lines; AC's own lap timer for cross-checks only.
- [ ] **Lifecycle.** `load` attaches to a running AC session (start it by hand); `reset` = restart
      session hotkey (may be unsupported: `teleport=False`); `close` releases the virtual controller.
- [ ] **No ground-truth leaks.** Profiles with `odometry = ground_truth` are blocked; the detection
      overlay is the one allowed use of truth (it *is* the sensor model).
- [ ] **Gate.** A profile must pass the 3D gate (`./ucr-dv gate --bench lite`) before AC runs.

## Suggested order

1. UDP telemetry forwarder + `ac_to_enu` with tests (drive by hand, watch the pose in the GUI).
2. Cone map for one AC track; `SensorSynth` + Stack in open loop (perception only, human driving).
3. Virtual controller; closed loop at low speed with the baseline profile.
4. Lap timing, gate, results comparison against the 3D bench (sim-to-sim gap).
