"""Validate a downloaded trusted checkpoint archive, without file hashes."""
import argparse
import json
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-dir', required=True)
    args = parser.parse_args()
    root = Path(args.archive_dir).resolve()
    result = json.loads((root/'result.json').read_text(encoding='utf-8'))
    if (result['status'] != 'complete' or result['epochs'] != 5 or
            result['test_at_best_validation'] is not None or len(result['history']) != 5):
        raise ValueError('Only completed five-epoch validation-only archives are accepted')
    last_path, best_path = root/'checkpoints/last.pth', root/'checkpoints/best.pth'
    last = torch.load(last_path, map_location='cpu', weights_only=False)
    if (last['next_epoch'], last['next_step'], last.get('checkpoint_role')) != (6, 0, 'resumable'):
        raise ValueError('Full last is not a completed resumable checkpoint')
    if not last.get('optimizer',{}).get('state') or len(last['rng_by_rank']) != 4:
        raise ValueError('Missing AdamW moments or per-rank RNG state')
    shape_map = {key:(tuple(value.shape),str(value.dtype)) for key,value in last['model'].items()}
    last_meta = {key:last[key] for key in ('experiment','seed','config','best','best_checkpoint')}
    del last
    best = torch.load(best_path, map_location='cpu', weights_only=False)
    if best.get('checkpoint_role') != 'selection_only' or 'optimizer' in best:
        raise ValueError('Expected compact selection-only best')
    if any(best[key] != value for key,value in last_meta.items()):
        raise ValueError('Best and last have inconsistent selection/configuration metadata')
    if (best['next_epoch'] != best['best']['epoch']+1 or best['next_step'] != 0 or
            best['best'] != result['best_validation']):
        raise ValueError('Best checkpoint does not match recorded validation selection')
    if shape_map != {key:(tuple(value.shape),str(value.dtype)) for key,value in best['model'].items()}:
        raise ValueError('Best/last model architecture or dtype mismatch')
    for name in ('last.pth','best.pth'):
        expected = root/'transfer_manifest.json'
        if expected.exists():
            sizes=json.loads(expected.read_text(encoding='utf-8'))['checkpoint_sizes']
            if (root/'checkpoints'/name).stat().st_size != sizes[name]:
                raise IOError('Transfer file size mismatch: '+name)
    receipt=dict(status='passed',archive_dir=str(root),experiment=best['experiment'],
        best_validation=best['best'],next_resume_epoch=6,model_keys=len(shape_map),
        last_bytes=last_path.stat().st_size,best_bytes=best_path.stat().st_size,
        torch_version=torch.__version__,python=sys.version,no_file_hashes=True,
        optimizer_present=True,all_rank_rng_present=True,test_evaluated=False)
    target=root/'archive_verification.json'
    with open(target,'x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
