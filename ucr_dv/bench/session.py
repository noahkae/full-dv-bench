"""
High-level entry points: build a backend + stack for a profile and run it.

    from ucr_dv.bench.session import run_episode
    result, runner = run_episode('baseline', 'autocross', bench='timeless', seed=0)
"""
from __future__ import annotations

from typing import Optional

from ..core.profile import Profile, load_profile
from ..pipeline import Stack
from ..tracks import load_track
from ..vehicle import VehicleSpec, load_vehicle
from .runner import EpisodeRunner, RunConfig

BENCHES = {'timeless': 'Timeless 2D', 'lite': 'fsds_lite 3D', 'assetto_corsa': 'Assetto Corsa (planned)'}


def make_backend(name, **kw):
    if name == 'timeless':
        from ..backends.timeless import TimelessBackend
        return TimelessBackend(**kw)
    if name == 'lite':
        from ..backends.lite import LiteBackend
        return LiteBackend(**kw)
    if name == 'assetto_corsa':
        from ..backends.assetto_corsa import AssettoCorsaBackend
        return AssettoCorsaBackend(**kw)
    raise KeyError('unknown bench {!r}'.format(name))


def prepare(profile, track_name, bench='timeless', seed=0, backend=None, backend_kw=None, scenery=None):
    """(runner, track, stack, backend), not started."""
    prof = profile if isinstance(profile, Profile) else load_profile(profile)
    if scenery is None:
        scenery = bench != 'timeless'
    track = load_track(track_name, scenery=scenery)
    vehicle = VehicleSpec(load_vehicle(prof.vehicle))
    stack = Stack(prof, event=track.mission, vehicle=vehicle, seed=seed)
    b = backend or make_backend(bench, **(backend_kw or {}))
    b.load(track, prof, vehicle, seed, stack=stack)
    runner = EpisodeRunner(b, stack, RunConfig(track=track_name, bench=bench, seed=seed), track=track)
    return runner, track, stack, b


def run_episode(profile, track_name, bench='timeless', seed=0, save=False, **kw):
    runner, track, stack, b = prepare(profile, track_name, bench, seed, **kw)
    result = runner.run()
    if save:
        from .records import save_run
        result['path'] = save_run(runner, stack.profile, track_name, bench, seed)
    if bench != 'timeless':
        b.close()
    return result, runner
