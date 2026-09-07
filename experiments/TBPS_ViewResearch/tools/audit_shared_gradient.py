"""Read-only, train-only gradient audit of an immutable research checkpoint."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--code-root', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--batches', type=int, default=3)
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.code_root).resolve()))
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from research.runner import load_config
    from research.model import build_model
    from research.data import enriched_catalog, SupportDataset, SupportSampler, to_device
    from view4.common import manifest, write_json, rank, seed32, load_training_checkpoint, reset_training_rng, utcnow
    from view4.config import official_config
    from view4.runtime import init_training, rank0_call
    from view4.upstream import bootstrap

    cfg = load_config(args.config)
    device, control = init_training(cfg)
    bootstrap(cfg['nltk_data'])
    out = Path(args.output_dir).resolve()
    def create():
        if out.exists():
            raise FileExistsError('Preserve previous gradient audit')
        out.mkdir(parents=True)
        write_json(out/'run_manifest.json', manifest(' '.join(sys.argv),
                   [args.config, args.checkpoint], out, cfg['seed']))
    rank0_call(create, control)
    catalog, _ = enriched_catalog(cfg)
    dataset = SupportDataset(catalog, cfg['dataset_root'], cfg['delta'])
    reset_training_rng(cfg['seed'])
    model = build_model(official_config(cfg, device), cfg, dataset.num_train_ids).to(device)
    checkpoint = load_training_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=True)
    selected_epoch = checkpoint['best']['epoch']
    del checkpoint
    model.train()
    reset_training_rng(seed32(cfg['seed'], 'gradient_audit'))
    params = [(n, p) for n, p in model.named_parameters()
              if n.startswith('visual.transformer.resblocks.11.') or n == 'visual.proj' or n.startswith('experts.')]
    groups = {name: [i for i, (n, _) in enumerate(params) if predicate(n)] for name, predicate in {
        'visual_last_block': lambda n: n.startswith('visual.transformer.resblocks.11.'),
        'visual_projection': lambda n: n == 'visual.proj',
        'residual_heads': lambda n: n.startswith('experts.')}.items()}
    captured = {}
    original = model.auxiliary
    def capture(*values):
        shared, decor, stats = original(*values)
        captured.update(shared=shared, decor=decor)
        return shared, decor, stats
    model.auxiliary = capture
    with np.load(str(Path(cfg['prepared_dir'])/'plans/seed_1/E0_epoch_3.npz')) as archive:
        indices = archive['pair_indices'][:args.batches]
    sampler = SupportSampler(indices, cfg['seed'], 3, rank(), dataset, cfg['max_supports_per_rank'])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    rows = []
    for step, host in enumerate(loader):
        batch = to_device(host, device, seed32(cfg['seed'], 3, step, rank(), 'text_batch'))
        with torch.cuda.amp.autocast():
            output = model(batch, .5, 0., 0.)
        terms = {'retrieval': output['loss_total'], **captured}
        vectors = {}
        for name, value in terms.items():
            grads = torch.autograd.grad(value * 4096., [p for _, p in params],
                                        retain_graph=True, allow_unused=True)
            full = []
            for g, (_, parameter) in zip(grads, params):
                g = torch.zeros_like(parameter) if g is None else g
                g = g.float() / 4096.
                dist.all_reduce(g)
                g /= dist.get_world_size()
                if not torch.isfinite(g).all():
                    raise FloatingPointError('Nonfinite scaled gradient: ' + name)
                full.append(g.flatten())
            vectors[name] = {group: torch.cat([full[i] for i in index]) for group, index in groups.items()}
            del grads, full
        summary = {}
        for group in groups:
            a, b, c = [vectors[name][group] for name in ('retrieval', 'shared', 'decor')]
            na, nb, nc = [float(x.norm()) for x in (a, b, c)]
            summary[group] = dict(retrieval_norm=na, shared_norm=nb, decor_norm=nc,
                 shared_cosine=float(torch.nn.functional.cosine_similarity(a[None], b[None])) if na*nb else None,
                 weighted_shared_to_retrieval=cfg['shared_weight']*nb/na if na else None,
                 weighted_decor_to_retrieval=cfg['decorrelation_weight']*nc/na if na else None,
                 shared_weight_for_10pct_norm=.1*na/nb if nb else None)
        rows.append(dict(batch=step, groups=summary, supports=int(output['support_pairs'])))
        if rank() == 0:
            print(json.dumps(rows[-1]), flush=True)
        captured.clear()
        del output, terms, vectors, batch, a, b, c
    if rank() == 0:
        write_json(out/'result.json', dict(status='complete', checkpoint_epoch=selected_epoch,
                   batches=rows, parameter_scope={g: [params[i][0] for i in idx] for g, idx in groups.items()},
                   train_only=True, optimizer_steps=0, ended_at=utcnow()))
        info = json.loads((out/'run_manifest.json').read_text())
        info.update(status='complete', ended_at=utcnow())
        write_json(out/'run_manifest.json', info)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
