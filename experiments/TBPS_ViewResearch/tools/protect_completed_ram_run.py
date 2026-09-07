"""Protect full last on disk and package a completed RAM run; never delete."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import tarfile


def record(path):
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def save_new(path, value):
    with open(path, 'x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['v014', 'v015', 'v016', 'v017'], required=True)
    parser.add_argument('--phase', choices=['validation_run', 'e0_validation_run'], default='validation_run')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.phase == 'e0_validation_run' and args.version != 'v017':
        raise ValueError('Only V017 adds a baseline archive in this helper')
    name = 'TBPS_ViewResearch_%s_20260907' % args.version
    source = Path('/dev/shm') / name / args.phase
    disk = Path('/root/autodl-tmp') / name / args.phase
    if Path(args.output_dir) != disk or source.resolve() != source or disk.resolve() != disk:
        raise ValueError('Unexpected archive source/destination or symlink')
    result = json.loads((source/'result.json').read_text())
    persistent = json.loads((disk/'result.json').read_text())
    if (result != persistent or result['status'] != 'complete' or
            result['epochs'] != 5 or len(result['history']) != 5 or
            result['test_at_best_validation'] is not None):
        raise ValueError('Only identical completed validation-only five-epoch runs')
    last = source/'checkpoints/last.pth'
    target = disk/'checkpoints/last.pth'
    receipt_path = disk/'full_last_protection.json'
    prefix = 'e0' if args.phase == 'e0_validation_run' else 'method'
    archive = source.parent/(prefix+'_completed_archive.tar')
    for path in (last, source/'checkpoints/best.pth'):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Unexpected checkpoint source')
    if any(path.exists() for path in (target, receipt_path, archive, archive.with_suffix('.tar.partial'))):
        raise FileExistsError('Preserve existing protection/archive records')
    if shutil.disk_usage(disk).free < last.stat().st_size + 1024**3:
        raise RuntimeError('Insufficient persistent space plus1GiB reserve')
    before = record(last)
    temporary = target.with_suffix('.pth.protecting')
    if temporary.exists():
        raise FileExistsError('Previous incomplete protection exists')
    shutil.copy2(last, temporary)
    if temporary.stat().st_size != before['size'] or record(last) != before:
        raise IOError('Checkpoint changed or copy length mismatch')
    with open(temporary, 'rb') as stream:
        os.fsync(stream.fileno())
    os.rename(temporary, target)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_new(receipt_path, dict(status='protected_pending_local_archive', source=before,
        persistent=record(target), ended_at=now, source_deleted=False, no_file_hashes=True))
    transfer = source/'transfer_manifest.json'
    save_new(transfer, dict(source_run=str(source), source_deleted=False,
        checkpoint_sizes={n: (source/'checkpoints'/n).stat().st_size for n in ('best.pth','last.pth')},
        created_at=now, no_file_hashes=True))
    files = [p for p in source.iterdir() if p.is_file()]
    required = sum(p.stat().st_size for p in files) + sum(
        (source/'checkpoints'/n).stat().st_size for n in ('best.pth','last.pth'))
    if shutil.disk_usage(source).free < required + 1024**3:
        raise RuntimeError('Insufficient RAM disk space for archive plus1GiB reserve')
    partial = archive.with_suffix('.tar.partial')
    with tarfile.open(partial, 'w', dereference=True) as stream:
        for path in sorted(files):
            stream.add(path, arcname=args.phase+'/'+path.name, recursive=False)
        for name in ('best.pth','last.pth'):
            stream.add(source/'checkpoints'/name, arcname=args.phase+'/checkpoints/'+name,
                       recursive=False)
    os.rename(partial, archive)
    print(json.dumps(dict(status='ready_for_download', archive=record(archive),
                          persistent_last=record(target), source_deleted=False)), flush=True)


if __name__ == '__main__':
    main()
