"""
Headless command line (CI, sweeps over SSH, the WSL box without a display).

    python -m ucr_dv list
    python -m ucr_dv run  --profile baseline --track autocross --bench timeless --seeds 0 1 2 [--save]
    python -m ucr_dv gate --profile baseline [--bench timeless] [--record]
    python -m ucr_dv eval --profile lidar_camera --track autocross --bench lite
    python -m ucr_dv export --profile baseline          # ROS 2 config files
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog='ucr_dv')
    sub = ap.add_subparsers(dest='cmd')
    sub.add_parser('list', help='profiles, tracks, vehicles, algorithms')
    r = sub.add_parser('run', help='run episodes')
    r.add_argument('--profile', default='baseline')
    r.add_argument('--track', nargs='+', default=['autocross'])
    r.add_argument('--bench', default='timeless', choices=['timeless', 'lite'])
    r.add_argument('--seeds', nargs='+', type=int, default=[0])
    r.add_argument('--save', action='store_true')
    r.add_argument('--json', action='store_true')
    g = sub.add_parser('gate', help='run the promotion gate')
    g.add_argument('--profile', default='baseline')
    g.add_argument('--bench', default='timeless', choices=['timeless', 'lite'])
    g.add_argument('--record', action='store_true', help='store the result in the profile file')
    e = sub.add_parser('eval', help='perception evaluation along the reference line')
    e.add_argument('--profile', default='baseline')
    e.add_argument('--track', default='autocross')
    e.add_argument('--bench', default='timeless', choices=['timeless', 'lite'])
    e.add_argument('--distance', type=float, default=200.0)
    x = sub.add_parser('export', help='write ROS 2 config for a profile')
    x.add_argument('--profile', default='baseline')
    args = ap.parse_args(argv)

    from .core.profile import list_profiles, load_profile
    if args.cmd == 'list' or args.cmd is None:
        from . import paths
        from .core import registry
        from .tracks import list_tracks
        print('profiles :', ', '.join(list_profiles()))
        print('tracks   :', ', '.join(list_tracks()))
        print('vehicles :', ', '.join(paths.list_vehicles()))
        for k in registry.STAGES:
            print('{:<13}: {}'.format(k, ', '.join(registry.entries(k))))
        return 0
    prof = load_profile(args.profile)
    if args.cmd == 'run':
        from .bench.session import run_episode
        bad = 0
        for tr in args.track:
            for s in args.seeds:
                res, _ = run_episode(prof, tr, bench=args.bench, seed=s, save=args.save)
                bad += not res['complete']
                if args.json:
                    print(json.dumps({'track': tr, 'seed': s, **res}, default=str))
                else:
                    print('{:<14} seed {:<3} {:<50} DOO {:<3} OC {:<2} maxCTE {:.2f}  {:.0f}x'.format(
                        tr, s, res['summary'], res['doo'], res['off_course'], res['max_cte'], res['speedup']))
        return 1 if bad else 0
    if args.cmd == 'gate':
        from .bench.gate import record_gate, run_gate
        passed, rep = run_gate(prof, bench=args.bench, progress=lambda i, n, t: print('[{}/{}] {}'.format(i, n, t)))
        print('GATE {}: {}'.format('PASSED' if passed else 'NOT PASSED', '; '.join(rep['reasons']) or 'all criteria met'))
        if args.record:
            record_gate(prof, rep)
            prof.save()
        return 0 if passed else 1
    if args.cmd == 'eval':
        from .bench.perception_eval import evaluate
        rep, _, _ = evaluate(prof, args.track, bench=args.bench, distance=args.distance)
        for name, s in rep['sensors'].items():
            print('{:<10} P {:.3f}  R {:.3f}  RMSE {:.3f} m  colour {}  recall/range {}'.format(
                name, s['precision'], s['recall'], s['rmse_m'], s['color_accuracy'], s['recall_by_range']))
        print('timings ms:', rep['timings_ms'])
        return 0
    if args.cmd == 'export':
        from .deploy import export_ros
        for p in export_ros(prof):
            print('wrote', p)
        return 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
