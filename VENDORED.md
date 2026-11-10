# Vendored code

`fsds_lite/` and `tests/fsds_lite/` are copies of
`../driverless-systems/sim/fsds_lite` and `../driverless-systems/sim/tests`
(copied 2026-11-09), with `config/settings.json`, `config/vehicle.json` and
`tracks/droneport.csv` moved into `data/vehicle/` and `data/tracks/`.

Local changes (paths only):
- `fsds_lite/track.py`: `TRACKS_DIR` -> `<root>/data/tracks`
- `fsds_lite/__main__.py`: default `--settings` / `--vehicle` -> `<root>/data/vehicle/`
- `tests/fsds_lite/conftest.py`: `SIM_DIR` / `SETTINGS` follow the new layout

Everything else is byte-identical. ucr_dv extends fsds_lite from outside
(`ucr_dv/backends/lite.py`) instead of editing it, so upstream fixes can be
re-copied. Check drift with:

    diff -r --exclude=__pycache__ ../driverless-systems/sim/fsds_lite fsds_lite

`data/tracks/timeless_map{1,2,3}.json` are copied unchanged from
`../ucr-dv-timeless/maps/`. The timeless car parameters
(`bin/car_params.bin`) are folded into `data/vehicle/ucr_ev.json`.
