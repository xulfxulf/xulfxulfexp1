import argparse
import math
import sys
from view4.geometry import check_geometry


def verify(result):
    check_geometry(result['config'])
    if (result['experiment'], result['seed'], result['epochs'], result['status']) != ('E0', 1, 10, 'complete'):
        raise RuntimeError('Incomplete CUHK E0')
    if result['config']['dataset'] != 'CUHK-PEDES' or result['test_at_best_validation'] is not None:
        raise RuntimeError('Baseline must be CUHK validation-only')
    if len(result['history']) != 10:
        raise RuntimeError('Missing epochs')
    for row in result['history']:
        if row['batches'] != 212 or not all(math.isfinite(row['validation'][k]) for k in ('r1', 'mAP')):
            raise RuntimeError('Missing batches/nonfinite metrics')
    best = max(result['history'], key=lambda row: row['validation']['r1'])
    if best['epoch'] != result['best_validation']['epoch']:
        raise RuntimeError('Incorrect best-R1 checkpoint selection')
    return dict(guard='passed', validation=result['best_validation'], test_evaluated=False)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'train', 'audit'):
        s = sub.add_parser(name)
        s.add_argument('--config', required=True)
        s.add_argument('--output-dir', required=True)
        if name == 'train':
            s.add_argument('--resume')
        if name == 'audit':
            s.add_argument('--steps', type=int, choices=[8], default=8)
    s = sub.add_parser('verify-result')
    s.add_argument('--result', required=True)
    args = p.parse_args()
    if args.command == 'verify-result':
        from view4.common import read_json
        print(verify(read_json(args.result)))
        return
    from view4.config import load_config
    cfg = load_config(args.config)
    if args.command == 'prepare':
        from view4.prepare import prepare
        prepare(cfg, args.output_dir, ' '.join(sys.argv))
    else:
        from view4.train import train
        train(cfg, 'E0', 1, 'audit' if args.command == 'audit' else 'formal', args.output_dir,
              ' '.join(sys.argv), resume=getattr(args, 'resume', None),
              audit_steps=args.steps if args.command == 'audit' else None)


if __name__ == '__main__':
    main()
