#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from radfusion_gpu_common import *
ap=argparse.ArgumentParser(); ap.add_argument('--root',default='/NHNHOME/BASE/RadFusion/data/reference/R3_BREAST_SPARSE_SHARDS'); ap.add_argument('--batch',type=int,default=64); ap.add_argument('--hidden',type=int,default=3584); ap.add_argument('--blocks',type=int,default=6); ap.add_argument('--latent',type=int,default=1024); ap.add_argument('--steps',type=int,default=80); ap.add_argument('--seed',type=int,default=20260906); ap.add_argument('--out',default='/NHNHOME/BASE/RadFusion/results/gpu5d/io_benchmark.json'); args=ap.parse_args()
dist,rank,world,local_rank=dist_info(); seed_all(args.seed,rank); device=torch.device('cuda',local_rank); records=sidecar_records(args.root); stream=InfiniteShardStream(records,args.batch,rank,world,args.seed)
model=RadEncoderDAE(14817,args.hidden,args.blocks,args.latent).to(device); model=DDP(model,device_ids=[local_rank]) if world>1 else model; opt=torch.optim.AdamW(model.parameters(),lr=1e-4,fused=True)
for _ in range(5):
    x=normalize_dense(stream.next_batch().to(device,non_blocking=True)); xi,m=corrupt_input(x)
    with torch.autocast('cuda',dtype=torch.bfloat16): pred,_=model(xi); loss=dae_loss(pred,x,m)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
torch.cuda.synchronize(); t0=time.time(); seen=0
for _ in range(args.steps):
    x=normalize_dense(stream.next_batch().to(device,non_blocking=True)); xi,m=corrupt_input(x)
    with torch.autocast('cuda',dtype=torch.bfloat16): pred,_=model(xi); loss=dae_loss(pred,x,m)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); seen+=x.shape[0]
torch.cuda.synchronize(); elapsed=time.time()-t0
if rank==0:
    raw=model.module if hasattr(model,'module') else model; result={'world':world,'batch_per_gpu':args.batch,'global_batch':args.batch*world,'steps':args.steps,'elapsed_sec':elapsed,'cells_per_sec_per_rank':seen/elapsed,'approx_global_cells_per_sec':seen*world/elapsed,'model_parameters':parameter_count(raw),'max_memory_gb_rank0':torch.cuda.max_memory_allocated()/2**30,'config':vars(args)}; p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
barrier(dist); cleanup_dist(dist)
