"""Read-only matched train gradients for two R-ITC target floors."""
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
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.code_root).resolve()))
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from research.runner import load_config
    from research.model import build_model
    from research.data import enriched_catalog, SupportDataset, SupportSampler, to_device, validate_pk_plan
    from view4.common import file_record, manifest, write_json, rank, seed32, reset_training_rng, utcnow
    from view4.config import official_config
    from view4.runtime import init_training, rank0_call
    from view4.upstream import bootstrap

    cfg = load_config(args.config)
    bootstrap(cfg['nltk_data'])
    device, control = init_training(cfg)
    out = Path(args.output_dir).resolve()
    checkpoint_record = file_record(args.checkpoint)
    plan_path = Path(cfg['prepared_dir'])/'plans/seed_1/PK_epoch_1.npz'

    def prepare():
        out.mkdir(parents=True, exist_ok=False)
        info = manifest(' '.join(sys.argv), [args.config,args.checkpoint,plan_path], out, cfg['seed'])
        info.update(train_only=True,optimizer_updates=0,epsilons=[.01,.001],batches=10,
                    stochastic_forward_rng_matched=True,beta=.5,shared_weight=.4,decorrelation_weight=.0003)
        write_json(out/'run_manifest.json', info)
    rank0_call(prepare, control)
    catalog, sources = enriched_catalog(cfg)
    dataset = SupportDataset(catalog,cfg['dataset_root'],cfg['delta'])
    reset_training_rng(cfg['seed'])
    model = build_model(official_config(cfg,device),cfg,dataset.num_train_ids).to(device).train()
    saved = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved['config'] != cfg or saved['input_sources'] != sources:
        raise ValueError('Checkpoint/config/input mismatch')
    model.load_state_dict(saved['model'],strict=True)
    del saved
    selectors = {
        'visual_last_block': lambda n:n.startswith('visual.transformer.resblocks.11.'),
        'visual_projection': lambda n:n=='visual.proj',
        'text_last_block': lambda n:n.startswith('encode_text.transformer.resblocks.11.'),
        'text_projection': lambda n:n=='encode_text.text_projection',
        'residual_heads': lambda n:n.startswith('experts.'),
        'patch_queries': lambda n:n=='expert_queries',
        'logit_scale': lambda n:n=='logit_scale'}
    parameters = [(n,p) for n,p in model.named_parameters() if any(f(n) for f in selectors.values())]
    groups = {g:[i for i,(n,_) in enumerate(parameters) if select(n)] for g,select in selectors.items()}
    if not all(groups.values()):
        raise ValueError('Missing gradient group')
    with np.load(str(plan_path),allow_pickle=False) as archive:
        indices = archive['pair_indices'][:10].copy()
    validate_pk_plan(indices,dataset,10,4,80,160)
    sampler = SupportSampler(indices,cfg['seed'],1,rank(),dataset,cfg['max_supports_per_rank'])
    rows = []
    for step,host in enumerate(DataLoader(dataset,batch_sampler=sampler,num_workers=0)):
        batch = to_device(host,device,seed32(cfg['seed'],1,step,rank(),'text_batch'))
        vectors, losses = {}, {}
        for epsilon in (.01,.001):
            model.eps = epsilon
            reset_training_rng(seed32(cfg['seed'],step,'matched_ritc_floor'))
            with torch.cuda.amp.autocast():
                output = model(batch,.5,.4,.0003)
                loss = output['loss_total']
            gradients = torch.autograd.grad(loss*4096.,[p for _,p in parameters],allow_unused=True)
            flattened = []
            for grad,(_,parameter) in zip(gradients,parameters):
                grad = torch.zeros_like(parameter) if grad is None else grad
                grad = (grad.float()/4096.).contiguous()
                dist.all_reduce(grad)
                grad /= dist.get_world_size()
                if not torch.isfinite(grad).all():
                    raise FloatingPointError('Nonfinite diagnostic gradient')
                flattened.append(grad.flatten())
            vectors[epsilon] = {g:torch.cat([flattened[i] for i in ix]) for g,ix in groups.items()}
            metrics = torch.stack([output[k].detach().float() for k in ('loss_total','nitc','ritc')])
            dist.all_reduce(metrics)
            losses[str(epsilon)] = dict(zip(('loss_total','nitc','ritc'),(metrics/dist.get_world_size()).tolist()))
            del gradients,flattened,output,loss
        if losses['0.01']['nitc'] != losses['0.001']['nitc']:
            raise RuntimeError('Unmatched main forward/dropout in epsilon comparison')
        measurements = {}
        for group in groups:
            a,b = vectors[.01][group],vectors[.001][group]
            na,nb = float(a.norm()),float(b.norm())
            measurements[group] = dict(original_norm=na,lower_floor_norm=nb,
                cosine=float(torch.nn.functional.cosine_similarity(a[None],b[None])) if na*nb else None,
                lower_to_original_norm=nb/na if na else None)
        row = dict(batch=step,losses=losses,groups=measurements)
        rows.append(row)
        if rank()==0:
            print(json.dumps(row),flush=True)
        del vectors,batch,a,b
    if rank()==0:
        if file_record(args.checkpoint) != checkpoint_record:
            raise RuntimeError('Read-only checkpoint changed')
        summary = {}
        for g in groups:
            summary[g] = {}
            for key in ('cosine','lower_to_original_norm'):
                values = [r['groups'][g][key] for r in rows if r['groups'][g][key] is not None]
                summary[g][key] = dict(count=len(values),mean=float(np.mean(values)),min=min(values),max=max(values)) if values else None
        write_json(out/'result.json',dict(status='complete',train_only=True,optimizer_updates=0,
            test_evaluated=False,checkpoint=checkpoint_record,rows=rows,summary=summary,
            pair_indices_by_batch_and_rank=indices.tolist(),ended_at=utcnow()))
        info = json.loads((out/'run_manifest.json').read_text())
        info.update(status='complete',ended_at=utcnow())
        write_json(out/'run_manifest.json',info)
        print('SUMMARY',json.dumps(summary),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
