#!/usr/bin/env python3
import os, json, time, random
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

def dist_info():
    if 'RANK' in os.environ:
        import torch.distributed as dist
        dist.init_process_group('nccl')
        rank = dist.get_rank(); world = dist.get_world_size(); local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return dist, rank, world, local_rank
    return None, 0, 1, 0

def barrier(dist):
    if dist is not None: dist.barrier()

def cleanup_dist(dist):
    if dist is not None: dist.destroy_process_group()

def seed_all(seed, rank=0):
    s = int(seed) + int(rank) * 100003
    random.seed(s); np.random.seed(s % (2**32-1)); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def normalize_dense(x):
    lib = x.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.log1p(x / lib * 1.0e4)

class ResidualBlock(nn.Module):
    def __init__(self, d, dropout=0.05):
        super().__init__(); self.norm=nn.LayerNorm(d); self.fc1=nn.Linear(d,d*2); self.fc2=nn.Linear(d*2,d); self.drop=nn.Dropout(dropout)
    def forward(self,x):
        h=self.norm(x); h=F.gelu(self.fc1(h), approximate='tanh'); h=self.drop(h); h=self.fc2(h); return x+self.drop(h)

class RadEncoderDAE(nn.Module):
    def __init__(self, genes=14817, hidden=3584, blocks=6, latent=1024, dropout=0.05):
        super().__init__(); self.genes=genes; self.hidden=hidden; self.blocks_n=blocks; self.latent=latent
        self.in_norm=nn.LayerNorm(genes); self.in_proj=nn.Linear(genes,hidden)
        self.blocks=nn.ModuleList([ResidualBlock(hidden,dropout) for _ in range(blocks)])
        self.to_latent=nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden,latent))
        self.from_latent=nn.Linear(latent,hidden)
        self.out_blocks=nn.ModuleList([ResidualBlock(hidden,dropout) for _ in range(max(1,blocks//3))])
        self.out_norm=nn.LayerNorm(hidden); self.out_proj=nn.Linear(hidden,genes)
    def encode(self,x):
        h=self.in_proj(self.in_norm(x))
        for b in self.blocks: h=b(h)
        return self.to_latent(h)
    def decode(self,z):
        h=self.from_latent(z)
        for b in self.out_blocks: h=b(h)
        return self.out_proj(self.out_norm(h))
    def forward(self,x):
        z=self.encode(x); return self.decode(z), z

class ConditionTransport(nn.Module):
    def __init__(self, latent=1024, cond_dim=5, hidden=2048, blocks=4, dropout=0.05):
        super().__init__(); self.cond=nn.Sequential(nn.Linear(cond_dim,128),nn.GELU(),nn.Linear(128,256),nn.GELU()); self.inp=nn.Linear(latent+256,hidden)
        self.blocks=nn.ModuleList([ResidualBlock(hidden,dropout) for _ in range(blocks)]); self.out=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,latent))
    def forward(self,z,c):
        h=self.inp(torch.cat([z,self.cond(c)],dim=-1))
        for b in self.blocks: h=b(h)
        return z+self.out(h)

def model_from_checkpoint_dict(ckpt):
    cfg=ckpt['model_config']; m=RadEncoderDAE(**cfg); m.load_state_dict(ckpt['model'],strict=True); return m,cfg

def atomic_torch_save(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(obj,tmp); os.replace(tmp,path)

def sidecar_records(root):
    out=[]
    for p in sorted(Path(root).rglob('part-*.json')):
        try: x=json.loads(p.read_text())
        except Exception: continue
        if x.get('status')!='OK': continue
        mf=Path(x['matrix_file'])
        if mf.exists(): out.append((p,x))
    return out

class InfiniteShardStream:
    def __init__(self,records,batch,rank,world,seed):
        self.records=records[rank::world]; self.batch=batch; self.rng=np.random.default_rng(seed+rank*17011); self.order=np.arange(len(self.records)); self.ptr=0; self.X=None; self.row_order=None; self.row_ptr=0
        if not self.records: raise RuntimeError(f'rank {rank}: no shards')
    def _load_next(self):
        if self.ptr==0: self.rng.shuffle(self.order)
        ridx=self.order[self.ptr]; self.ptr=(self.ptr+1)%len(self.order); _,rec=self.records[ridx]
        self.X=sp.load_npz(rec['matrix_file']).tocsr(); self.row_order=np.arange(self.X.shape[0]); self.rng.shuffle(self.row_order); self.row_ptr=0
    def next_batch(self):
        while True:
            if self.X is None or self.row_ptr+self.batch>self.X.shape[0]: self._load_next()
            idx=self.row_order[self.row_ptr:self.row_ptr+self.batch]; self.row_ptr+=self.batch
            if len(idx)!=self.batch: continue
            return torch.from_numpy(self.X[idx].toarray().astype(np.float32,copy=False))

def corrupt_input(x,mask_prob=0.15):
    nz=x>0; m=nz & (torch.rand_like(x)<mask_prob); out=x.clone(); out[m]=0; out=out+torch.randn_like(out)*0.01; return out,m

def dae_loss(pred,target,mask):
    all_loss=F.smooth_l1_loss(pred,target)
    if mask.any(): return 0.30*all_loss+0.70*F.smooth_l1_loss(pred[mask],target[mask])
    return all_loss

def rbf_mmd(x,y):
    n=min(x.shape[0],y.shape[0],256); x=x[:n]; y=y[:n]
    with torch.no_grad():
        cat=torch.cat([x,y],0); d=torch.cdist(cat,cat); med=torch.median(d[d>0]); sigma2=(med*med).clamp_min(1e-4)
    return torch.exp(-torch.cdist(x,x).pow(2)/(2*sigma2)).mean()+torch.exp(-torch.cdist(y,y).pow(2)/(2*sigma2)).mean()-2*torch.exp(-torch.cdist(x,y).pow(2)/(2*sigma2)).mean()

def condition_vector(dose_gy,time_h,fraction,radiation_code,device):
    d=torch.as_tensor(dose_gy,device=device,dtype=torch.float32); t=torch.as_tensor(time_h,device=device,dtype=torch.float32); f=torch.as_tensor(fraction,device=device,dtype=torch.float32); r=torch.as_tensor(radiation_code,device=device,dtype=torch.float32)
    return torch.stack([torch.log1p(d.clamp_min(0)),torch.log1p(t.clamp_min(0)),f/10.0,r,d/20.0],dim=-1)

def parameter_count(model): return sum(p.numel() for p in model.parameters())
