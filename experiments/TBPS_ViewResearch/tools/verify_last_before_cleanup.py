"""Verify a trusted local last checkpoint before explicitly authorized cleanup."""
import argparse
import json
from pathlib import Path

import torch


def check_finite(value):
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError('Nonfinite archived tensor')
        return value.numel()
    if isinstance(value, dict):
        return sum(check_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(check_finite(item) for item in value)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive-dir', required=True)
    parser.add_argument('--checkpoint-subdir', choices=['.', 'checkpoints'], default='.')
    args = parser.parse_args()
    root = Path(args.archive_dir).resolve()
    source = json.loads((root/'source.json').read_text(encoding='utf-8-sig'))
    result = json.loads((root/'result.json').read_text(encoding='utf-8'))
    path = root/args.checkpoint_subdir/'last.pth'
    if path.stat().st_size != source['bytes']:
        raise ValueError('Transfer length mismatch')
    epochs = result['epochs']
    if result['status'] != 'complete' or epochs not in (5, 10) or len(result['history']) != epochs:
        raise ValueError('Only completed five/ten-epoch runs are authorized')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if (saved['next_epoch'], saved['next_step']) != (epochs + 1, 0):
        raise ValueError('Checkpoint is not the completed resumable state')
    if saved.get('checkpoint_role') not in (None, 'resumable'):
        raise ValueError('Selection-only checkpoint cannot substitute for last')
    for key in ('experiment', 'seed', 'config', 'history'):
        if saved[key] != result[key]:
            raise ValueError('Checkpoint/result mismatch: '+key)
    if saved['best'] != result['best_validation']:
        raise ValueError('Checkpoint best selection mismatch')
    if not saved.get('optimizer', {}).get('state') or len(saved['rng_by_rank']) != 4:
        raise ValueError('Missing optimizer moments or rank RNG state')
    counts = {name:check_finite(saved[name]) for name in ('model', 'optimizer')}
    receipt = dict(status='passed', local_checkpoint=str(path), source=source,
                   local_bytes=path.stat().st_size, experiment=saved['experiment'],
                   model_keys=len(saved['model']), next_epoch=epochs + 1, next_step=0,
                   optimizer_present=True, all_rank_rng_present=True,
                   finite_tensor_elements=counts, content_hashes_used=False)
    with open(root/'verification.json', 'x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
