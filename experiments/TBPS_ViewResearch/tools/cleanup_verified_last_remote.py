"""Delete one exactly whitelisted last.pth after a local verification receipt."""
import argparse
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/root/autodl-tmp')
RUNS = {
    'TBPS_ViewResearch_v001_20260907': ('validation_run',),
    'TBPS_ViewResearch_v002_20260907': ('validation_run',),
    'TBPS_ViewResearch_v003_20260907': ('validation_run',),
    'TBPS_ViewResearch_v004_20260907': ('validation_run',),
    'TBPS_ViewResearch_v005_20260907_r2': ('e0_validation_run', 'validation_run'),
    'TBPS_ViewResearch_v006_20260907': ('validation_run',),
    'TBPS_ViewResearch_v007_20260907': ('e0_validation_run', 'validation_run'),
    'TBPS_ViewResearch_v010_20260907': ('validation_run',),
    'TBPS_ViewResearch_v011_20260907': ('validation_run',),
    'TBPS_ViewResearch_v012_20260907': ('validation_run',),
    'TBPS_ViewResearch_v013_20260907': ('e0_validation_run', 'validation_run'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verified-receipt', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--execute-user-approved-cleanup', action='store_true')
    args = parser.parse_args()
    if not args.execute_user_approved_cleanup:
        raise ValueError('Explicit archive-then-delete approval is required')
    receipt = json.loads(Path(args.verified_receipt).read_text())
    src = receipt['source']
    path = Path(src['path'])
    allowed = {ROOT/name/run/'checkpoints/last.pth' for name, runs in RUNS.items() for run in runs}
    if path not in allowed or path.resolve() != path or path.is_symlink():
        raise ValueError('Not an exact authorized current-round checkpoint path')
    if (receipt['status'] != 'passed' or not receipt['optimizer_present'] or
            not receipt['all_rank_rng_present'] or receipt['local_bytes'] != src['bytes']):
        raise ValueError('No successful full local archive verification')
    st = path.stat()
    if (st.st_size, st.st_mtime_ns, st.st_nlink) != (src['bytes'], src['mtime_ns'], 1):
        raise ValueError('Source checkpoint changed or has shared hardlinks')
    run = path.parent.parent
    result = json.loads((run/'result.json').read_text())
    if result['status'] != 'complete' or result['experiment'] != receipt['experiment']:
        raise ValueError('Source run is incomplete or different')
    # Refuse deletion if this checkpoint is currently open or its run is resumed.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            for fd in (process/'fd').iterdir():
                try:
                    if fd.resolve() == path:
                        raise RuntimeError('Checkpoint is open by process '+process.name)
                except FileNotFoundError:
                    pass
            raw = (process/'cmdline').read_bytes()
            if str(run).encode() in raw and b'--resume' in raw:
                raise RuntimeError('Source run has an active resume process')
        except (FileNotFoundError, PermissionError):
            continue
    best = path.parent/'best.pth'
    if not best.is_file() or best.resolve() == path:
        raise ValueError('Distinct preserved best checkpoint is missing')
    import torch
    selected = torch.load(best, map_location='cpu', weights_only=False)
    if selected['best'] != result['best_validation']:
        raise ValueError('Preserved best checkpoint is not the recorded selection')
    del selected
    output = Path(args.output_dir)
    if not output.is_absolute() or output.resolve() != ROOT/'TBPS_ViewResearch_last_cleanup_20260907':
        raise ValueError('Unexpected cleanup record directory')
    output.mkdir(exist_ok=True)
    target = output/(src['root']+'_'+run.name+'.json')
    if target.exists():
        raise FileExistsError('Cleanup already recorded')
    before = shutil.disk_usage(ROOT).free
    record = dict(status='verified_ready', checkpoint=str(path), local_archive=receipt['local_checkpoint'],
                  bytes=st.st_size, preserved_best=str(best), raw_datasets_untouched=True)
    target.write_text(json.dumps(record, indent=2))
    # No recursive deletion: only the single exact allowlisted checkpoint.
    path.unlink()
    record.update(status='deleted_after_local_verification', disk_free_before=before,
                  disk_free_after=shutil.disk_usage(ROOT).free, best_still_exists=best.exists())
    target.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
