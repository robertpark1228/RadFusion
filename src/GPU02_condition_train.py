#!/usr/bin/env python3
import argparse,time
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from radfusion_gpu_common import *
ap=argparse.ArgumentParser(); ap.add_argument('--bundles',default='/NHNHOME/BASE/RadFusion/data/radiation/GPU_BUNDLES'); ap.add_argument('--encoder-checkpoint',required=True); ap.add_argument('--out',required=True); ap.add_argument('--batch',type=int,default=256); ap.add_argument('--lr',type=float,default=2e-4); ap.add_argument('--wall-hours',type=float,required=True); ap.add_argument('--seed',type=int,default=20260906); ap.add_argument('--holdout-context',default='GSE162931|GBM827'); ap.add_argument('--save-minutes',type=float,default=30); args=ap.parse_args()
dist,rank,world,local_rank=dist_info(); seed_all(args.seed,rank); device=torch.device('cuda',local_rank); out=Path(args.out)
if rank==0: out.mkdir(parents=True,exist_ok=True)
barrier(dist); ck=torch.load(args.encoder_checkpoint,map_location='cpu'); encoder,enc_cfg=model_from_checkpoint_dict(ck); encoder=encoder.to(device).eval(); [p.requires_grad_(False) for p in encoder.parameters()]
transport=ConditionTransport(latent=enc_cfg['latent'],cond_dim=5,hidden=2048,blocks=4).to(device); transport=DDP(transport,device_ids=[local_rank],broadcast_buffers=False) if world>1 else transport; raw=transport.module if hasattr(transport,'module') else transport; opt=torch.optim.AdamW(transport.parameters(),lr=args.lr,weight_decay=0.01,fused=True)
groups=[]
for d in sorted(Path(args.bundles).iterdir()):
    if not d.is_dir() or not (d/'matrix.npz').exists(): continue
    X=sp.load_npz(d/'matrix.npz').tocsr(); m=np.load(d/'meta.npz',allow_pickle=False); context=m['context_id'].astype(str); is_c=m['is_control'].astype(bool); is_t=m['is_irradiated'].astype(bool); sample_ids=m['sample_id'].astype(str)
    for ctx in sorted(set(context)):
        if ctx==args.holdout_context: continue
        ci=np.flatnonzero((context==ctx)&is_c)
        for sid in sorted(set(sample_ids[(context==ctx)&is_t])):
            ti=np.flatnonzero((context==ctx)&is_t&(sample_ids==sid))
            if len(ci)<32 or len(ti)<32: continue
            j=ti[0]; groups.append({'study':d.name,'context':ctx,'sample':sid,'X':X,'control_idx':ci,'treated_idx':ti,'dose':float(m['dose_gy'][j]),'time':float(m['time_h'][j]),'fraction':float(m['fraction'][j]),'rcode':float(m['radiation_code'][j])})
if not groups: raise RuntimeError('No trainable control/treated groups')
if rank==0:
    print('TRAIN_GROUPS',len(groups))
    for g in groups: print(g['context'],g['sample'],g['dose'],g['time'],len(g['control_idx']),len(g['treated_idx']))
rng=np.random.default_rng(args.seed+rank*911); start=time.monotonic(); deadline=start+args.wall_hours*3600; last_save=start; step=0
def enc_rows(X,idx):
    x=torch.from_numpy(X[idx].toarray().astype(np.float32,copy=False)).to(device,non_blocking=True); x=normalize_dense(x)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16): return encoder.encode(x).float()
def save_ck(tag):
    if rank!=0:return
    atomic_torch_save({'transport':raw.state_dict(),'transport_config':{'latent':enc_cfg['latent'],'cond_dim':5,'hidden':2048,'blocks':4,'dropout':0.05},'encoder_checkpoint':str(args.encoder_checkpoint),'encoder_config':enc_cfg,'step':step,'holdout_context':args.holdout_context,'args':vars(args)},out/tag)
while time.monotonic()<deadline:
    g=groups[int(rng.integers(0,len(groups)))]; b=min(args.batch,len(g['control_idx']),len(g['treated_idx'])); ci=rng.choice(g['control_idx'],size=b,replace=False); ti=rng.choice(g['treated_idx'],size=b,replace=False); zc=enc_rows(g['X'],ci); zt=enc_rows(g['X'],ti)
    c=condition_vector(np.full(b,g['dose'],np.float32),np.full(b,g['time'],np.float32),np.full(b,g['fraction'],np.float32),np.full(b,g['rcode'],np.float32),device)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        zp=transport(zc,c).float(); mean_loss=F.mse_loss(zp.mean(0),zt.mean(0)); std_loss=F.mse_loss(zp.std(0),zt.std(0)); mmd=rbf_mmd(zp,zt); delta_reg=(zp-zc).pow(2).mean(); loss=3*mean_loss+std_loss+2*mmd+0.01*delta_reg
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(transport.parameters(),1.0); opt.step(); step+=1
    if rank==0 and step%20==0: print(f'step={step} loss={float(loss):.6f} mean={float(mean_loss):.6f} std={float(std_loss):.6f} mmd={float(mmd):.6f} group={g["context"]}::{g["sample"]}',flush=True)
    now=time.monotonic()
    if now-last_save>=args.save_minutes*60: barrier(dist); save_ck('latest.pt'); barrier(dist); last_save=now
barrier(dist); save_ck('latest.pt')
if rank==0: save_ck(f'final_step_{step:09d}.pt'); print('CONDITION_TRAIN_COMPLETE',step)
barrier(dist); cleanup_dist(dist)
