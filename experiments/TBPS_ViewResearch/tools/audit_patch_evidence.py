"""Train-only attention and residual contribution audit; never update weights."""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code-root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', action='append', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.code_root).resolve()))
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from research.runner import load_config
    from research.model import build_model
    from research.data import enriched_catalog, SoftGalleryDataset
    from view4.config import official_config
    from view4.common import manifest, read_json, write_json, load_training_checkpoint, reset_training_rng, utcnow
    from view4.runtime import init_evaluation
    from view4.upstream import bootstrap
    from text_utils.tokenizer import tokenize

    cfg = load_config(args.config)
    bootstrap(cfg['nltk_data'])
    device = init_evaluation('cuda:0')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    indices_path = Path(cfg['prepared_dir'])/'expert_diagnostic_subset.json'
    indices = read_json(indices_path)
    catalog, _ = enriched_catalog(cfg)
    records = catalog['train']
    if len(indices) != len(set(indices)) or any(i < 0 or i >= len(records) for i in indices):
        raise ValueError('Invalid train-only subset indices')
    write_json(output/'run_manifest.json', manifest(' '.join(sys.argv),
               [args.config, indices_path, *args.checkpoint], output, cfg['seed']))
    data = SoftGalleryDataset(records, cfg['dataset_root'])
    captions = [records[i]['captions'][0] for i in indices]
    ids = torch.tensor([records[i]['person_id'] for i in indices])
    rows = []
    for path in args.checkpoint:
        saved = load_training_checkpoint(path)
        if (saved['config']['dataset'] != cfg['dataset'] or
            saved['config'].get('input_resolution',[224,224]) != cfg.get('input_resolution',[224,224])):
            raise ValueError('Checkpoint dataset/geometry mismatch')
        reset_training_rng(cfg['seed'])
        model = build_model(official_config(cfg,device),cfg,len(set(r['person_id'] for r in records)))
        missing = model.load_state_dict(saved['model'],strict=False)
        expected = {key for key in model.state_dict() if key.startswith('experts.') or key == 'expert_queries'}
        if saved['experiment'] == 'E0':
            if set(missing.missing_keys) != expected or missing.unexpected_keys:
                raise ValueError('E0 encoder/key mismatch')
        elif missing.missing_keys or missing.unexpected_keys:
            raise ValueError('Incomplete trained expert checkpoint')
        row = dict(checkpoint=str(Path(path).resolve()), experiment=saved['experiment'],
                   checkpoint_epoch=saved['next_epoch']-1 if saved['next_step']==0 else saved['next_epoch'],
                   next_step=saved['next_step'], training_selected_epoch=saved['best']['epoch'])
        del saved
        model = model.to(device).eval()
        shared, fused, attention, heads, rho, text = [], [], [], [], [], []
        with torch.no_grad():
            for batch in DataLoader(Subset(data,indices),batch_size=32,shuffle=False,num_workers=0):
                h,dense = model.encode_image(batch['image'].to(device),return_dense=True)
                v,r = model.fuse(h,batch['view_prob'].to(device),dense[:,1:])
                _,a = model.pool_patches(dense[:,1:])
                shared.append(F.normalize(h,dim=-1).float().cpu())
                fused.append(v.float().cpu())
                attention.append(a.cpu())
                heads.append(model.expert_outputs(h,dense[:,1:]).cpu())
                rho.append((model.residual_alpha*r.norm(dim=-1)/h.float().norm(dim=-1).clamp_min(1e-12)).cpu())
            for start in range(0,len(captions),128):
                tokens=tokenize(captions[start:start+128],context_length=77).to(device)
                text.append(F.normalize(model.encode_text(tokens),dim=-1).float().cpu())
        h,v,a,r,ratios,t = map(torch.cat,(shared,fused,attention,heads,rho,text))
        if not all(torch.isfinite(value).all() for value in (h,v,a,r,ratios,t)):
            raise FloatingPointError('Nonfinite diagnostic embeddings')
        score_h,score_v = t@h.t(),t@v.t()
        same = ids[:,None].eq(ids[None,:])
        def retrieval(scores):
            positive=scores.masked_fill(~same,float('-inf')).max(1).values
            negative=scores.masked_fill(same,float('-inf')).max(1).values
            return dict(subset_r1=float(same.gather(1,scores.argmax(1)[:,None]).float().mean()*100),
                        positive_minus_top_negative_mean=float((positive-negative).mean()),
                        paired_score_mean=float(scores.diag().mean()))
        row.update(train_images=len(indices),shared=retrieval(score_h),fused=retrieval(score_v),
                   normalized_attention_entropy_mean=(-(a*a.clamp_min(1e-12).log()).sum(-1)/math.log(a.shape[-1])).mean(0).tolist(),
                   attention_max_weight_mean=a.max(-1).values.mean(0).tolist(),
                   query_norms=model.expert_queries.detach().norm(dim=1).cpu().tolist(),
                   rho_quantiles=torch.quantile(ratios,torch.tensor([.1,.5,.9,.95])).tolist(),
                   paired_score_improved_fraction=float((score_v.diag()>score_h.diag()).float().mean()),
                   head_pairs={})
        for x,y in itertools.combinations(range(3),2):
            valid=(r[:,x].norm(dim=-1)>1e-8)&(r[:,y].norm(dim=-1)>1e-8)
            row['head_pairs']['%d_%d'%(x,y)]=dict(
                attention_cosine=float(F.cosine_similarity(a[:,x],a[:,y],dim=-1).mean()),
                attention_top_patch_overlap=float((a[:,x].argmax(-1)==a[:,y].argmax(-1)).float().mean()),
                output_cos2=float(F.cosine_similarity(r[valid,x],r[valid,y],dim=-1).square().mean()) if valid.any() else None)
        rows.append(row)
        print(json.dumps(row),flush=True)
        del model,h,v,a,r,ratios,t,shared,fused,attention,heads,rho,text
        torch.cuda.empty_cache()
    write_json(output/'result.json',dict(status='complete',train_only=True,test_evaluated=False,
               optimizer_updates=0,train_indices=indices,measurements=rows,
               warning='Training-subset scores are mechanistic diagnostics, not held-out performance',ended_at=utcnow()))
    info=read_json(output/'run_manifest.json')
    info.update(status='complete',ended_at=utcnow())
    write_json(output/'run_manifest.json',info)


if __name__ == '__main__':
    main()
