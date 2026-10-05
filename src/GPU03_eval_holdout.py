#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
import torch.nn.functional as F
from radfusion_gpu_common import *
ap=argparse.ArgumentParser(); ap.add_argument('--bundles',default='/NHNHOME/BASE/RadFusion/data/radiation/GPU_BUNDLES'); ap.add_argument('--transport-checkpoint',required=True); ap.add_argument('--context',default='GSE162931|GBM827'); ap.add_argument('--batch',type=int,default=1024); ap.add_argument('--out',default='/NHNHOME/BASE/RadFusion/results/gpu5d/final_eval.json'); args=ap.parse_args()
device=torch.device('cuda',0); tck=torch.load(args.transport_checkpoint,map_location='cpu'); eck=torch.load(tck['encoder_checkpoint'],map_location='cpu'); encoder,ecfg=model_from_checkpoint_dict(eck); encoder=encoder.to(device).eval(); [p.requires_grad_(False) for p in encoder.parameters()]; tcfg=tck['transport_config']; transport=ConditionTransport(**tcfg).to(device).eval(); transport.load_state_dict(tck['transport'],strict=True)
found=None
for d in sorted(Path(args.bundles).iterdir()):
    if not (d/'matrix.npz').exists(): continue
    X=sp.load_npz(d/'matrix.npz').tocsr(); m=np.load(d/'meta.npz',allow_pickle=False); context=m['context_id'].astype(str)
    if args.context not in set(context): continue
    ci=np.flatnonzero((context==args.context)&m['is_control'].astype(bool)); ti_all=np.flatnonzero((context==args.context)&m['is_irradiated'].astype(bool))
    if len(ci) and len(ti_all): found=(d.name,X,m,ci,ti_all); break
if found is None: raise RuntimeError(f'Holdout context not found: {args.context}')
study,X,m,ci,ti_all=found; rng=np.random.default_rng(20260906); sample_ids=m['sample_id'].astype(str); results=[]
def enc_rows(idx):
    x=torch.from_numpy(X[idx].toarray().astype(np.float32,copy=False)).to(device); x=normalize_dense(x)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): return encoder.encode(x).float()
for sid in sorted(set(sample_ids[ti_all])):
    ti=np.flatnonzero((m['context_id'].astype(str)==args.context)&m['is_irradiated'].astype(bool)&(sample_ids==sid)); b=min(args.batch,len(ci),len(ti)); cidx=rng.choice(ci,b,replace=False if len(ci)>=b else True); tidx=rng.choice(ti,b,replace=False if len(ti)>=b else True); zc=enc_rows(cidx); zt=enc_rows(tidx); j=tidx[0]
    cv=condition_vector(np.full(b,float(m['dose_gy'][j]),np.float32),np.full(b,float(m['time_h'][j]),np.float32),np.full(b,float(m['fraction'][j]),np.float32),np.full(b,float(m['radiation_code'][j]),np.float32),device)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): zp=transport(zc,cv).float()
    im=F.mse_loss(zc.mean(0),zt.mean(0)).item(); mm=F.mse_loss(zp.mean(0),zt.mean(0)).item(); i_mmd=rbf_mmd(zc,zt).item(); m_mmd=rbf_mmd(zp,zt).item(); results.append({'study':study,'context':args.context,'treated_sample':sid,'n_eval':b,'identity_mean_mse':im,'model_mean_mse':mm,'mean_mse_improvement_fraction':1-mm/max(im,1e-12),'identity_mmd':i_mmd,'model_mmd':m_mmd,'mmd_improvement_fraction':1-m_mmd/max(i_mmd,1e-12)})
out={'holdout_context':args.context,'results':results}; p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
