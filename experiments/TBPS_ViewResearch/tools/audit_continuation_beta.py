"""Matched train-only gradients for restarted versus retained TBPS beta."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--code-root',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(args.code_root).resolve()))
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from research.runner import load_config
    from research.model import build_model
    from research.data import enriched_catalog,SupportDataset,SupportSampler,to_device
    from view4.common import (file_record,manifest,write_json,rank,seed32,reset_training_rng,utcnow)
    from view4.config import official_config
    from view4.runtime import init_training,rank0_call
    from view4.upstream import bootstrap

    cfg=load_config(args.config)
    bootstrap(cfg['nltk_data'])
    device,control=init_training(cfg)
    output=Path(args.output_dir).resolve()
    checkpoint_record=file_record(cfg['initializer'])
    plan_path=Path(cfg['prepared_dir'])/'plans/seed_1/E0_epoch_1.npz'
    def prepare():
        output.mkdir(parents=True,exist_ok=False)
        info=manifest(' '.join(sys.argv),[args.config,cfg['initializer'],plan_path],output,cfg['seed'])
        info.update(train_only=True,optimizer_updates=0,betas=[0.,.5],batches=10,
                    stochastic_forward_rng_matched=True,shared_weight=0.,decorrelation_weight=0.)
        write_json(output/'run_manifest.json',info)
    rank0_call(prepare,control)
    catalog,_=enriched_catalog(cfg)
    dataset=SupportDataset(catalog,cfg['dataset_root'],cfg['delta'])
    reset_training_rng(cfg['seed'])
    model=build_model(official_config(cfg,device),cfg,dataset.num_train_ids,
                      initializer=cfg['initializer']).to(device).train()
    predicates={
        'visual_last_block':lambda n:n.startswith('visual.transformer.resblocks.11.'),
        'visual_projection':lambda n:n=='visual.proj',
        'text_last_block':lambda n:n.startswith('encode_text.transformer.resblocks.11.'),
        'text_projection':lambda n:n=='encode_text.text_projection',
        'residual_heads':lambda n:n.startswith('experts.')}
    parameters=[(n,p) for n,p in model.named_parameters() if any(f(n) for f in predicates.values())]
    groups={g:[i for i,(n,_) in enumerate(parameters) if select(n)] for g,select in predicates.items()}
    if not all(groups.values()):
        raise ValueError('Missing declared gradient parameter group')
    with np.load(str(plan_path),allow_pickle=False) as archive:
        indices=archive['pair_indices'][:10].copy()
    sampler=SupportSampler(indices,cfg['seed'],1,rank(),dataset,cfg['max_supports_per_rank'])
    rows=[]
    for step,host in enumerate(DataLoader(dataset,batch_sampler=sampler,num_workers=0)):
        batch=to_device(host,device,seed32(cfg['seed'],1,step,rank(),'text_batch'))
        vectors,losses={},{}
        for beta in (0.,.5):
            reset_training_rng(seed32(cfg['seed'],step,'matched_beta_gradient'))
            with torch.cuda.amp.autocast():
                result=model(batch,beta,0.,0.)
                loss=result['loss_total']
            grads=torch.autograd.grad(loss*4096.,[p for _,p in parameters],allow_unused=True)
            flattened=[]
            for grad,(_,parameter) in zip(grads,parameters):
                grad=torch.zeros_like(parameter) if grad is None else grad
                grad=(grad.float()/4096.).contiguous()
                dist.all_reduce(grad)
                grad/=dist.get_world_size()
                if not torch.isfinite(grad).all():
                    raise FloatingPointError('Nonfinite beta-comparison gradient')
                flattened.append(grad.flatten())
            vectors[beta]={g:torch.cat([flattened[i] for i in ix]) for g,ix in groups.items()}
            value=loss.detach().float()
            dist.all_reduce(value)
            losses[str(beta)]=float(value/dist.get_world_size())
            del grads,flattened,result,loss
        measurements={}
        for group in groups:
            a,b=vectors[0.][group],vectors[.5][group]
            na,nb=float(a.norm()),float(b.norm())
            measurements[group]=dict(beta0_norm=na,beta05_norm=nb,
                cosine=float(torch.nn.functional.cosine_similarity(a[None],b[None])) if na*nb else None,
                restarted_to_retained_norm=na/nb if nb else None,
                gradient_change_relative_to_retained=float((a-b).norm())/nb if nb else None)
        row=dict(batch=step,losses=losses,groups=measurements)
        rows.append(row)
        if rank()==0:
            print(json.dumps(row),flush=True)
        del vectors,batch,a,b
    if rank()==0:
        if file_record(cfg['initializer'])!=checkpoint_record:
            raise RuntimeError('Read-only source checkpoint changed')
        summary={}
        for group in groups:
            values=[row['groups'][group] for row in rows]
            summary[group]={key:dict(mean=float(np.mean([x[key] for x in values])),
                                    min=min(x[key] for x in values),max=max(x[key] for x in values))
                            for key in ('cosine','restarted_to_retained_norm','gradient_change_relative_to_retained')
                            if all(x[key] is not None for x in values)}
        write_json(output/'result.json',dict(status='complete',train_only=True,optimizer_updates=0,
            test_evaluated=False,checkpoint=checkpoint_record,rows=rows,summary=summary,
            pair_indices_by_batch_and_rank=indices.tolist(),
            parameter_scope={g:[parameters[i][0] for i in ix] for g,ix in groups.items()},ended_at=utcnow()))
        info=json.loads((output/'run_manifest.json').read_text())
        info.update(status='complete',ended_at=utcnow())
        write_json(output/'run_manifest.json',info)
        print('SUMMARY',json.dumps(summary),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
