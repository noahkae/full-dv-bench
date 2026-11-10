"""
Scoring, the same on every bench (FS Driverless rules, simplified):

    time      = run time from the referee (skidpad: mean of the timed laps)
    penalties = 2 s per cone down/out (DOO) + 10 s per off-course (OC)
    DNF       if the event wasn't completed, the car was lost or it stalled

`uses_gt` marks runs whose stack used simulator ground truth (GT odometry);
they are fine for development but cannot pass the promotion gate.
"""
from __future__ import annotations

import numpy as np

DOO_PENALTY = 2.0
OC_PENALTY = 10.0


def score(status, reason, event, uses_gt=False, log=None, stack_errors=0, error=None):
    complete = bool(status.complete)
    penalties = DOO_PENALTY * status.doo + OC_PENALTY * status.off_course
    dnf = None
    if not complete:
        dnf = {'lost': status.dnf or 'left the track', 'timeout': 'time limit reached',
               'stalled': 'car stopped making progress', 'error': 'stack crashed',
               'stopped': 'stopped by user'}.get(reason, 'event not completed')
    run_time = status.run_time
    total = (run_time + penalties) if (complete and run_time is not None) else None
    res = {
        'event': event, 'reason': reason, 'complete': complete, 'dnf': dnf,
        'run_time': None if run_time is None else round(run_time, 3),
        'laps': [round(x, 3) for x in status.laps], 'doo': status.doo, 'off_course': status.off_course,
        'crashes': status.crashes, 'penalties': round(penalties, 2),
        'total': None if total is None else round(total, 3),
        'finished_signal': bool(status.finished), 'uses_gt': bool(uses_gt),
        'max_cte': round(float(status.extras.get('max_cte', 0.0)), 3),
        'rms_cte': round(float(status.extras.get('rms_cte', 0.0)), 3),
        'stack_errors': stack_errors,
    }
    if log is not None and log.rows:
        a = log.arrays()
        res['max_speed'] = round(float(a['speed'].max()), 2)
        res['mean_speed'] = round(float(a['speed'].mean()), 2)
        res['compute_ms_p50'] = round(float(np.percentile(a['compute_ms'], 50)), 3)
        res['compute_ms_p95'] = round(float(np.percentile(a['compute_ms'], 95)), 3)
        res['max_lat_g'] = round(float(np.abs(a['ay']).max() / 9.81), 2)
        steer_rate = np.diff(a['steer_cmd']) / np.maximum(np.diff(a['t']), 1e-3)
        res['steer_rate_rms'] = round(float(np.sqrt(np.mean(steer_rate ** 2))) if len(steer_rate) else 0.0, 3)
    if error:
        res['error'] = error[-2000:]
    if complete:
        res['summary'] = '{:.2f} s + {:.0f} s penalties = {:.2f} s'.format(run_time, penalties, total)
    else:
        res['summary'] = 'DNF ({})'.format(dnf)
    return res
