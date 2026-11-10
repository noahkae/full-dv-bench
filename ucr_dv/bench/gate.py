"""
Promotion gate: is a stack ready to move to the next bench?

    timeless -> 3D      every gate track completed on every seed, mean DOO
                        and OC under the limits, no stack exceptions, and
                        no simulator ground truth in the stack
    3D -> Assetto Corsa the same on the 3D bench (raw sensors) - checked
                        when the AC backend exists

Results are stored in the profile file under "gates" so the GUI can show
which rung a stack has reached.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List

from .session import run_episode

GATE_TRACKS = ['acceleration', 'autocross', 'timeless_map1', 'timeless_map2', 'timeless_map3']


@dataclass
class GateCriteria:
    tracks: List[str] = field(default_factory=lambda: list(GATE_TRACKS))
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    max_mean_doo: float = 2.0
    max_mean_oc: float = 0.5


def run_gate(profile, bench='timeless', criteria=None, progress=None, stop_flag=None):
    """Returns (passed, report dict). `progress(i, n, text)` is called per run."""
    c = criteria or GateCriteria()
    rows = []
    n = len(c.tracks) * len(c.seeds)
    i = 0
    for tr in c.tracks:
        for seed in c.seeds:
            if stop_flag is not None and stop_flag():
                return False, {'aborted': True, 'rows': rows}
            res, _ = run_episode(profile, tr, bench=bench, seed=seed)
            rows.append({'track': tr, 'seed': seed, 'complete': res['complete'], 'total': res['total'],
                         'doo': res['doo'], 'oc': res['off_course'], 'errors': res['stack_errors'],
                         'summary': res['summary']})
            i += 1
            if progress:
                progress(i, n, '{} seed {}: {}'.format(tr, seed, res['summary']))
    done = [r for r in rows if r['complete']]
    mean_doo = sum(r['doo'] for r in rows) / max(1, len(rows))
    mean_oc = sum(r['oc'] for r in rows) / max(1, len(rows))
    reasons = []
    if len(done) < len(rows):
        reasons.append('{} of {} runs did not finish'.format(len(rows) - len(done), len(rows)))
    if mean_doo > c.max_mean_doo:
        reasons.append('mean DOO {:.2f} > {:.2f}'.format(mean_doo, c.max_mean_doo))
    if mean_oc > c.max_mean_oc:
        reasons.append('mean OC {:.2f} > {:.2f}'.format(mean_oc, c.max_mean_oc))
    if any(r['errors'] for r in rows):
        reasons.append('the stack raised exceptions')
    if profile.uses_ground_truth:
        reasons.append('the stack uses simulator ground truth (odometry = ground_truth)')
    passed = not reasons
    report = {'bench': bench, 'passed': passed, 'reasons': reasons, 'rows': rows,
              'mean_doo': round(mean_doo, 2), 'mean_oc': round(mean_oc, 2),
              'mean_total': round(sum(r['total'] for r in done) / max(1, len(done)), 2) if done else None,
              'time': time.strftime('%Y-%m-%d %H:%M')}
    return passed, report


def record_gate(profile, report):
    gates = getattr(profile, 'gates', None) or {}
    gates[report['bench']] = {k: report[k] for k in ('passed', 'reasons', 'mean_doo', 'mean_oc',
                                                     'mean_total', 'time')}
    profile.gates = gates
    return gates
