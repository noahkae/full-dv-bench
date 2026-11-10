# Context for future sessions (ucr-dv-stack)

Condensed record of the session that built this app (2026-11-09/10). Read this
instead of re-exploring. Details: README.md, docs/ARCHITECTURE.md, VENDORED.md.

## What the user asked for
An all-in-one app/workbench for UCalgary Racing driverless development:
choose sensors (counts, types, placement, algorithms), develop perception, then
planning and control; test first in the 2D "timeless" bench, then in the 3D
bench; one frontend for the whole stack. Assetto Corsa will be the final sim
later — design for it but don't implement it. New folder: `Capstone/ucr-dv-stack`.

User decisions (via questions):
- **Vendor copies**: fsds_lite is copied in (not imported in place); timeless is ported. Never edit the original folders.
- **In-process now, ROS-ready**: algorithms run in-process; thin ROS 2 wrapper provided (untested, no ROS here).
- Depth priorities: sensor rig + perception, planning + control, benches + results. Research integration is light.
- Later request: GUI panes must be consistently sized (fixed — see below).

## Environment
- Dev machine: Arch Linux (user said "I'm using Arch btw"), system Python 3.14, no ROS.
- Team target: Ubuntu 20.04 / ROS 2 Galactic / Python 3.8 (WSL2) → all code must stay 3.8-compatible, numpy-1.17-safe (enforced by tests/ucr_dv/test_compat.py).
- Venvs in ucr-dv-stack: `.venv` (py3.14: numpy 2.x, scipy, PyQt5, pyqtgraph 0.14, moderngl, msgpack, pytest, matplotlib for debugging) and `.venv38` (py3.8: numpy 1.24, pyqtgraph 0.13.3).
- Launch: `./ucr-dv` (GUI), `./ucr-dv list|run|gate|eval|export` (CLI, ucr_dv/cli.py). Tests: `.venv/bin/python -m pytest tests` → 120 passed, 1 skipped (both venvs).
- Offscreen GUI checks: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m ucr_dv --page <page> --screenshot out.png`.

## Source folders (sibling, untouched)
- `ucr-dv-timeless`: 2D teaching sim (State, kinematic bicycle, GT+noise detector, greedy pairing planner, P-steer, MyMission.loop contract, map JSON). Bugs fixed in the port: steering deque pop() delay, spurious-cone x/y swap, collision box from wheelbase, global RNG, py3.10 syntax. User's own uncommitted edits there (gains.py, mission.py, ARCHITECTURE.md) predate this work.
- `driverless-systems`: ROS 2 stack (cone_detector grid clustering, gates planner, mapping merge, P-steer control_node via AirSim RPC), fsds_lite sim (Simulator lockstep `step(n)`, 4-wheel dynamics, ray-cast Lidar, moderngl CameraRenderer, SimPanel viewer), workbench.py.
- `fsd-research`: 10 FS driverless papers + index.html (recommendations: Delaunay/boundaries, adaptive pure pursuit, DBSCAN, camera-LiDAR fusion, EKF/GraphSLAM). Indexed in ucr_dv/research/library.py.
- `Formula-Student-Driverless-Simulator`: unmodified upstream FSDS clone.

## Architecture (ucr_dv/)
- `core/types.py`: ENU frames; cones = (N,4) [x,y,color,conf], colours YELLOW0 BLUE1 ORANGE2 BIG_ORANGE3 UNKNOWN4; ControlCommand.steering = road-wheel rad, **+left** (fsds_lite uses -1..1 +right).
- `core/registry.py`: `@register(stage, name, params=[P(...)], refs=[...])`, stages lidar/camera/fusion/odometry/mapping/planner/speed/lateral/longitudinal/mission; GUI forms + ROS params generated from schemas.
- `core/profile.py`: Profile JSON (rig + stages + vehicle + pipeline_hz + gates) in data/profiles/.
- `sensors/rig.py` (SensorSpec, presets from papers, to_lite_lidar/cameras), `sensors/models.py` (detection-level LiDAR/camera/stereo/timeless_classic models; coverage_grid).
- perception: lidar.py (lidar_grid, lidar_dbscan, plane_fit/RANSAC ground, big-cone by height), camera.py (camera_color_blob: HSV + ndimage label + stripe merge + ground-plane/depth ranging), fusion.py (concat, nearest_assoc with Mahalanobis + one-detection-per-sensor greedy association + lidar confirmation, priority), metrics.py.
- estimation: odometry.py (ground_truth [uses_gt], dead_reckoning, ekf), mapping.py (none, cone_map with range-weighted merge + damped ICP `localize` = lightweight SLAM).
- planning: planners.py (boundaries [robust default fallback], delaunay [virtual mids, skew, consistency, side inference], gates, greedy_pairing, track_template [skidpad/accel rules layout]), speed.py (grip_limited, constant), mission.py (big-orange line crossings both sides, run-out, event crossings counts).
- control: lateral.py (pure_pursuit, stanley, p_steer, mpc), longitudinal.py (pid_speed + vehicle pedal map, bang_bang).
- `pipeline.py`: Stack.step(obs, dt) → StackOutput; motion-compensated per-sensor tracks; model covariance for raw detections.
- backends: timeless.py (2D, grip limit, delay/lag/slew steering, fsds Referee), lite.py (fsds_lite lockstep, LockstepCameraRenderer, RNG reseed, per-sensor fidelity raw/detection), common.py (SensorSynth, CourseMonitor), assetto_corsa.py (stub; ac_to_enu UNVERIFIED), base.py (Backend, Capabilities, EpisodeStatus).
- bench: runner.py (EpisodeRunner thread, post() commands, Frame incl. raw frames), scoring.py (DOO 2 s, OC 10 s), records.py (runs/<id>/run.json + log.npz), gate.py (5 tracks × 3 seeds), perception_eval.py (GT-driver lap scoring), session.py (prepare/run_episode/make_backend).
- `vehicle.py`: VehicleSpec over fsds_lite VehicleParams; data/vehicle/ucr_ev.json (212 kg, 11.46:1 single gear, TC on) shared by both benches.
- `tracks.py`: builtins + FSDS CSV + timeless JSON (moved to car-at-origin); `track_reference` cached ON the track object (id()-cache caused a bug).
- `deploy.py`: export_ros (config/<profile>.json + flattened YAML), fsds_settings.
- app/: main.py (nav + pages wrapped in horizontal-only QScrollArea, Ctrl+S), state.py (AppState, RunController — signal renamed `run_event` because `event` shadowed QObject.event and crashed), widgets.py (ParamForm, AlgoPicker, TrackView, Telemetry, **Splitter** proportional/non-collapsible with min sizes), sensorview.py (camera boxes + LiDAR BEV), pages/{home,rig,perception,planning,bench,results,research,deploy}.py.
- fsds_lite/ vendored (only path patches; see VENDORED.md); ros/ucr_dv_ros/ (ament_python, stack_node.py, convert.py, launch/stack.launch.py).

## Frontend pass (2026-11-10)
- `app/theme.py`: colours via QPalette (no global `QWidget {background}` - that painted dark strips behind checkboxes/forms inside cards); QSS adds shape only. Arrow/check glyphs are drawn at startup into a temp dir (`theme._icons`). `theme.mono()` for logs/HUD.
- Sidebar is grouped nav buttons (BUILD / TEST / SHIP / REFERENCE; `main.NAV`), Ctrl+1..9 switch pages, profile card shows saved/unsaved + gate chips, Save is enabled only when dirty.
- `main.PageHost` wraps each page inside its scroll area: pages with word-wrapped labels are height-for-width, and a bare QScrollArea stretched them to their preferred height (taller than the window -> bottoms were clipped).
- `widgets.EmptyHint` (overlay text on empty lists/plots/tables), `widgets.chip/set_chip`.
- Home: `StepCard` workflow strip with live status per step, recent runs as a tree. Bench: track is previewed as soon as it is picked. Planning: stage pickers reflow 2 -> 1 column when the pane is narrow; preview controls are a grid.
- Offscreen GUI drive script pattern: build_window, `page.run()`, pump `processEvents` until `page.ctl.running` is False, `win.grab().save()`.

## Measured state at handoff
- `baseline` (VLP-16 only + track_template→boundaries + pure pursuit + dead reckoning + cone_map ICP): **passes timeless gate** (mean DOO 0.73, OC 0.07, recorded in profile); 3D raw LiDAR autocross 33.29 s 0 cones; accel 6.71 s.
- `lidar_camera` (VLP-16 + mono cam, EKF+GNSS, priority fusion): 3D raw autocross 32.90 s clean; timeless tight hairpins (map1/map2) still DNF — the open perception/fusion problem.
- `kit_style` (Pandar40P, pure pursuit + boundaries): good except accel 4 DOO. `timeless_classic`: map3 DNF, skidpad 12 DOO.
- Skidpad clips ~4 cones every run; MPC and Delaunay untuned (sign/shape tested only); detection models uncalibrated; ROS node untested on real ROS; AC not implemented (docs/ASSETTO_CORSA.md checklist).
- Speeds: timeless 5–30× real time (≈3× inside GUI); 3D ≈2.7× lidar-only, ≈1× with cameras.

## Lessons / gotchas
- Colourless planning needs side inference relative to previous path; geometry overrides wrong camera colours (>1.2 m wrong side).
- Fusion must not chain one sensor's detections via another's (greedy one-per-sensor association).
- ICP correction must rotate about the matched centroid, not the world origin.
- Python dict prints after os._exit need `-u`/flush; pytest basenames must be unique across tests/fsds_lite and tests/ucr_dv.
- QSplitter.setSizes before show + collapsible panes → slivers; use widgets.Splitter.

## Debug helpers (scratchpad, may be gone)
matrix.py / rigs.py (profile × track tables), dbg*.py (matplotlib failure plots), guirun.py (drive a GUI run offscreen), sizes.py (pane-size check at several window sizes).
