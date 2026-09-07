"""Persist a completed new RAM run's records/best without deleting any source."""
import argparse
import os
from pathlib import Path
import shutil

RAM_ROOT = Path('/dev/shm/TBPS_ViewResearch_v014_20260907')
DISK_ROOT = Path('/root/autodl-tmp/TBPS_ViewResearch_v014_20260907')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['e0_validation_run', 'validation_run'], required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    from view4.common import file_record, load_training_checkpoint, read_json, utcnow, write_json
    source = RAM_ROOT / args.phase
    destination = Path(args.output_dir)
    if not destination.is_absolute() or destination.resolve() != DISK_ROOT / args.phase:
        raise ValueError('Wrong persistent destination')
    if source.resolve() != source or destination.exists():
        raise ValueError('Unexpected symlink or existing output; preserve all records')
    result = read_json(source/'result.json')
    if args.phase == 'e0_validation_run':
        from run_cuhk_e0 import verify
        verify(result)
    else:
        from research.runner import verify_result
        verify_result(source/'result.json')
    last = load_training_checkpoint(source/'checkpoints/last.pth')
    if last['next_epoch'] != 6 or last['next_step'] != 0 or last.get('checkpoint_role') != 'resumable':
        raise ValueError('Last checkpoint is not the complete resumable state')
    if 'optimizer' not in last or len(last['history']) != 5:
        raise ValueError('Missing optimizer/history in full last')
    del last
    best_path = source/'checkpoints/best.pth'
    best = load_training_checkpoint(best_path)
    if (best['best']['epoch'] != result['best_validation']['epoch'] or
            best.get('checkpoint_role') != 'selection_only' or 'optimizer' in best):
        raise ValueError('Best checkpoint/selection metadata mismatch')
    name = best['best_checkpoint']
    if Path(name).name != name or name != 'best_epoch_%03d.pth' % best['best']['epoch']:
        raise ValueError('Invalid committed best checkpoint name')
    del best
    files = [p for p in source.iterdir() if p.is_file()]
    required = sum(p.stat().st_size for p in files) + best_path.stat().st_size
    if shutil.disk_usage(DISK_ROOT).free < required + 1024**3:
        raise RuntimeError('Insufficient persistent space for best and records plus1GiB reserve')
    staging = destination.with_name(destination.name+'.persisting')
    staging.mkdir(exist_ok=False)
    for path in files:
        shutil.copy2(path, staging/path.name)
    (staging/'checkpoints').mkdir()
    copied = staging/'checkpoints'/name
    shutil.copy2(best_path, copied)
    if copied.stat().st_size != best_path.stat().st_size:
        raise IOError('Best checkpoint copy length mismatch')
    with open(copied,'rb') as stream:
        os.fsync(stream.fileno())
    os.link(copied, staging/'checkpoints/best.pth')
    write_json(staging/'storage_record.json', dict(source=str(source), persistent_destination=str(destination),
        copied_best=file_record(best_path), full_last_source=file_record(source/'checkpoints/last.pth'),
        full_last_local_archive_required=True, source_deleted=False, ended_at=utcnow()))
    os.rename(staging,destination)
    fd=os.open(str(DISK_ROOT),os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print('PERSISTED',destination,flush=True)


if __name__ == '__main__':
    main()
