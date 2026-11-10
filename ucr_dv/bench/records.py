"""
Run records on disk: runs/<timestamp>_<bench>_<track>_<profile>/
    run.json   result + config + the full profile used (reproducible)
    log.npz    per-tick arrays (RunLog.COLS)
"""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np

from .. import paths


def save_run(runner, profile, track, bench, seed, root=None):
    root = root or paths.RUNS
    os.makedirs(root, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S')
    name = '{}_{}_{}_{}_s{}'.format(stamp, bench, track, profile.name, seed)
    d = os.path.join(root, name)
    k = 1
    while os.path.exists(d):
        k += 1
        d = os.path.join(root, '{}-{}'.format(name, k))
    os.makedirs(d)
    rec = {'id': os.path.basename(d), 'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'bench': bench,
           'track': track, 'seed': seed, 'profile_name': profile.name, 'result': runner.result,
           'events': runner.log.events, 'profile': profile.to_dict()}
    with open(os.path.join(d, 'run.json'), 'w') as f:
        json.dump(rec, f, indent=1, default=_json_default)
    np.savez_compressed(os.path.join(d, 'log.npz'), **runner.log.arrays())
    return d


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def list_runs(root=None):
    root = root or paths.RUNS
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root), reverse=True):
        f = os.path.join(root, name, 'run.json')
        if os.path.isfile(f):
            try:
                with open(f) as fh:
                    rec = json.load(fh)
                rec['dir'] = os.path.join(root, name)
                out.append(rec)
            except (OSError, ValueError):
                pass
    return out


def load_log(run_dir):
    with np.load(os.path.join(run_dir, 'log.npz')) as z:
        return {k: z[k] for k in z.files}


def delete_run(run_dir):
    if os.path.isfile(os.path.join(run_dir, 'run.json')):
        shutil.rmtree(run_dir)
