#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from radfusion_gpu_common import *
ap=argparse.ArgumentParser(); ap.add_argument('--root',default='/NHNHOME/BASE/RadFusion/data/reference/R3_BREAST_SPARSE_SHARDS'); ap.add_argument('--out',required=True); ap.add_argument('--hidden',type=int,required=True); ap.add_argument('--blocks',type=int,required=True); ap.add_argument('--latent',type=int,required=True); ap.add_argument('--batch',type=int,default=64); ap.add_argument('--grad-accum',type=int,default=2); ap.add_argument('--lr',type=float,default=1e-4); ap.add_argument('--weight-decay',type=float,default=0.01); ap.add_argument('--mask-prob',type=float,default=0.15); ap.add_argument('--wall-hours',type=float,required=True); ap.add_argument('--seed',type=int,default=20260906); ap.add_argument('--save-minutes',type=float,default=30); ap.add_argument('--resume',default=''); args=ap.parse_args()
dist,rank,world,local_rank=dist_info(); seed_all(args.seed,rank); device=torch.device('cuda',local_rank); out=Path(args.out)
if rank==0: out.mkdir(parents=True,exist_ok=True)
barrier(dist); records=sidecar_records(args.root); stream=InfiniteShardStream(records,args.batch,rank,world,args.seed)
cfg=dict(genes=14817,hidden=args.hidden,blocks=args.blocks,latent=args.latent,dropout=0.05); model=RadEncoderDAE(**cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,fused=True); step=0
if args.resume:
    ck=torch.load(args.resume,map_location='cpu'); model.load_state_dict(ck['model']); step=int(ck.get('step',0))
    if 'optimizer' in ck: opt.load_state_dict(ck['optimizer'])
    if rank==0: print('RESUMED',args.resume,'step',step)
model=DDP(model,device_ids=[local_rank],broadcast_buffers=False) if world>1 else model; raw=model.module if hasattr(model,'module') else model
if rank==0:
    meta={'model_config':cfg,'parameters':parameter_count(raw),'world':world,'batch_per_gpu':args.batch,'grad_accum':args.grad_accum,'global_effective_batch':args.batch*world*args.grad_accum,'args':vars(args)}; (out/'run_config.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2))
start=time.monotonic(); deadline=start+args.wall_hours*3600; last_save=start; last_log=start; running_loss=0.0; running_n=0; model.train()
def save_ck(tag='latest.pt'):
    if rank!=0: return
    atomic_torch_save({'model':raw.state_dict(),'optimizer':opt.state_dict(),'step':step,'model_config':cfg,'args':vars(args),'saved_unix':time.time()},out/tag)
while time.monotonic()<deadline:
    opt.zero_grad(set_to_none=True); accum=0.0
    for _ in range(args.grad_accum):
        xb=stream.next_batch().to(device,non_blocking=True); target=normalize_dense(xb); inp,mask=corrupt_input(target,args.mask_prob)
        with torch.autocast('cuda',dtype=torch.bfloat16): pred,_=model(inp); loss=dae_loss(pred,target,mask)/args.grad_accum
        loss.backward(); accum+=float(loss.detach())*args.grad_accum
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1; running_loss+=accum; running_n+=1; now=time.monotonic()
    if rank==0 and (step%25==0 or now-last_log>60):
        print(f'step={step} hours={(now-start)/3600:.3f}/{args.wall_hours} loss={running_loss/max(running_n,1):.6f} memGB={torch.cuda.max_memory_allocated()/2**30:.2f}',flush=True); running_loss=0.0; running_n=0; last_log=now
    if now-last_save>=args.save_minutes*60:
        barrier(dist); save_ck('latest.pt'); barrier(dist); last_save=now
barrier(dist); save_ck('latest.pt')
if rank==0: save_ck(f'final_step_{step:09d}.pt'); print('PRETRAIN_COMPLETE',step,'hours',(time.monotonic()-start)/3600)
barrier(dist); cleanup_dist(dist)
