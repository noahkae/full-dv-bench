# ucr-dv-stack - UCalgary Racing driverless workbench

One app for the whole driverless stack: **choose sensors -> build perception ->
plan and control -> test in a fast 2D bench -> promote to the 3D simulator**
(and, later, Assetto Corsa). It merges `ucr-dv-timeless` (2D teaching sim),
`driverless-systems` (fsds_lite 3D sim + ROS 2 stack) and `fsd-research`
(the paper library) behind a single GUI.

```
./ucr-dv                      # the GUI (creates .venv on first run)
./ucr-dv list                 # profiles, tracks, algorithms
./ucr-dv run --profile baseline --track autocross timeless_map1 --seeds 0 1 2
./ucr-dv gate --profile baseline            # promotion gate (timeless)
./ucr-dv eval --profile lidar_camera --bench lite   # perception scores on raw 3D data
./ucr-dv export --profile baseline          # ROS 2 config for the car
```

Needs Python >= 3.8 (3.8 for ROS 2 Galactic / Ubuntu 20.04 / WSL; developed on
3.14), `numpy`, `PyQt5`, `pyqtgraph`, and optionally `scipy` (faster
clustering/triangulation) and `moderngl` (3D view + rendered cameras). On Arch:
`./ucr-dv` handles it with `uv`. Tests: `.venv/bin/python -m pytest tests`.

## The idea: one profile, every bench

A **stack profile** (`data/profiles/*.json`) is everything that defines your
autonomous system:

- **rig** - each sensor's type, mount (x, y, z, roll, pitch, yaw), specs and detection algorithm
- **stages** - one registered algorithm + parameters each for fusion, odometry,
  mapping, planner, speed profile, lateral control, longitudinal control and mission
- **vehicle** - one vehicle file (`data/vehicle/ucr_ev.json`) used by both benches

The same profile runs unchanged on:

| bench | what it is | sensors | speed |
|---|---|---|---|
| **timeless** | 2D, lockstep, deterministic (ported from ucr-dv-timeless, bugs fixed) | detection-level models of your rig | 5-30x real time |
| **3D (fsds_lite)** | vendored fsds_lite in lockstep: tyres, load transfer, ray-cast LiDAR, moderngl cameras | **raw** point clouds / images through your real detectors, or detection models per sensor | 1-3x |
| Assetto Corsa | planned - interface fixed, see `docs/ASSETTO_CORSA.md` | detection models on a cone overlay | real time |

### Promotion ladder

1. **timeless** - iterate fast; pass the **gate** (5 tracks x 3 seeds, all
   finished, mean DOO <= 2, mean OC <= 0.5, no simulator ground truth).
2. **3D, detection fidelity** - real dynamics with the same perception (set sensors to `detection`).
3. **3D, raw** - real point clouds and images through your detectors.
4. Assetto Corsa (later), then the car via the ROS 2 package.

## The GUI

| page | what you do there |
|---|---|
| Home | pick/duplicate profiles, see the pipeline, validation problems, gate status, recent runs |
| Sensor rig | add sensors from presets (VLP-16, Pandar40P, OS1-64, AMZ wing LiDAR, ZED 2i, mono 12 mm, ...), move them, edit specs; **live coverage heatmap**, FOV wedges, side view of laser rings vs cone height, detection range per sensor |
| Perception | detector per sensor (grid / DBSCAN LiDAR clustering, colour-blob camera detector) + fusion; **Evaluate** drives a ground-truth lap and scores precision, recall vs range, RMSE, colour accuracy, latency - on timeless or on raw 3D data |
| Planning & control | odometry (GT / dead reckoning / EKF+GNSS), cone map with ICP localisation, planners (boundaries, Delaunay, gates, greedy pairing, known-layout template), grip-limited speed profile, pure pursuit / Stanley / MPC / P-steer, PID + pedal map; **single-frame preview** to scrub along a track and see the path the stack would plan |
| Timeless bench | run / pause / single-step at any speed, live view (truth vs what the car believes), telemetry, **batch** runs and the **promotion gate** |
| 3D bench | the same, with the embedded fsds_lite 3D/top-down view, **sensor feeds** (camera images with detection boxes, LiDAR bird's-eye view) and per-sensor fidelity |
| Results | every saved run: filter, overlay trajectories and speed traces, compare metrics (e.g. the same profile on timeless vs 3D) |
| Research | the fsd-research papers indexed by pipeline stage; which algorithm here each paper backs |
| Deploy / AC | export to the ROS 2 package / an FSDS `settings.json`; the Assetto Corsa roadmap |

**Live tuning:** change any algorithm parameter while a bench is running and
it applies on the next tick (algorithm swaps apply to the next run).

## Starter profiles

| profile | rig | notes |
|---|---|---|
| `baseline` | VLP-16 + wheel speed + IMU | **passes the timeless gate**; completes every event on 3D with raw LiDAR (autocross 33.3 s, 0 cones) |
| `lidar_camera` | VLP-16 + wide mono + GNSS | camera colour + LiDAR range; finishes 3D autocross clean with rendered images; covariance-gated fusion is the open problem |
| `kit_style` | Pandar40P only | KIT 2021-style LiDAR-only; MPC / Delaunay are there to tune |
| `timeless_classic` | the original timeless detector model | the old teaching setup on the new pipeline |

## Layout

```
ucr_dv/                 the package (pure Python + numpy; no Qt below app/)
  core/                 types (frames, cones, commands), registry (plugins + param schemas), profile
  sensors/              rig spec + presets, detection-level sensor models
  perception/           lidar.py, camera.py, fusion.py, metrics.py
  estimation/           odometry.py, mapping.py (cone map + ICP localisation)
  planning/             planners.py, speed.py, mission.py, pathutil.py
  control/              lateral.py, longitudinal.py
  pipeline.py           Stack: sensor frames -> ControlCommand, bench-agnostic
  backends/             timeless.py, lite.py (fsds_lite), assetto_corsa.py (stub), common.py
  bench/                runner (episodes, threading), scoring, records, gate, perception_eval, session
  app/                  PyQt5 GUI (pages/, widgets, state)
  deploy.py, cli.py
fsds_lite/              vendored from driverless-systems/sim (see VENDORED.md)
ros/ucr_dv_ros/         ROS 2 (Galactic) ament_python package: stack_node runs a profile
data/                   tracks (FS builtins + droneport.csv + timeless maps), vehicle, profiles, rigs
docs/                   ARCHITECTURE.md, ASSETTO_CORSA.md
tests/                  ucr_dv/ (this package), fsds_lite/ (vendored suite)
runs/                   saved runs (git-ignored)
```

## Adding an algorithm

```python
from ucr_dv.core.registry import Component, P, register

@register('lateral', 'my_controller', label='My controller',
          params=[P('gain', 1.0, 0.0, 5.0, 0.05, '', 'what it does')], refs=['paper / idea'])
class MyController(Component):
    """One line shown in the GUI."""
    def steer(self, path, v, ctx):          # path in the car frame, returns road-wheel rad (+left)
        return self.params['gain'] * ...
```

Import it from `ucr_dv/core/registry.py:_load_all` (or the stage's module) and
it appears in every dropdown, the profile JSON, the ROS parameters and the CLI.
The contracts per stage are in `docs/ARCHITECTURE.md`.

## Conventions

ENU everywhere: x forward, y left, z up; yaw counter-clockwise. Cones are
`(N, 4)` arrays `[x, y, colour, confidence]` with the timeless colour ids
(yellow 0, blue 1, orange 2, big orange 3, unknown 4). Steering in a
`ControlCommand` is the road-wheel angle in radians, **positive = left**; each
backend converts (fsds_lite: -1..1 positive right).

## Known limitations (honest list)

- Fusion with a mono camera still loses the track on the tightest timeless
  hairpins; LiDAR-only rigs are currently the most robust.
- Skidpad finishes on the template planner but clips ~4 cones per run.
- MPC and Delaunay are implemented and tested for sign/shape, not tuned.
- Detection-level models are physics-based but uncalibrated against the real
  sensors; the Perception page's 3D evaluation is how to calibrate them.
- The ROS 2 node is untested on real ROS (no ROS on the dev machine); its
  message conversions are unit-tested.
- Assetto Corsa: interface and checklist only.
