#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np, scipy.sparse as sp, anndata as ad
ROOT=Path('/NHNHOME/BASE/RadFusion'); OUT=ROOT/'data/radiation/GPU_BUNDLES'; OUT.mkdir(parents=True,exist_ok=True)
SOURCES=[ROOT/'data/radiation/R1_STANDARDIZED/GSE310219/GSE310219_MCF10A_R3_14817_raw.h5ad',ROOT/'data/radiation/R1_STANDARDIZED/GSE255800/GSE255800_BEAS2B_R3_14817_raw.h5ad',ROOT/'data/radiation/R1_STANDARDIZED/GSE162931/GSE162931_GBM_R3_14817_raw.h5ad']
def arr_str(obs,col,default=''):
    return np.array([default]*len(obs),dtype='U64') if col not in obs.columns else obs[col].astype(str).to_numpy(dtype='U128')
for src in SOURCES:
    if not src.exists(): raise SystemExit(f'MISSING: {src}')
    a=ad.read_h5ad(src); X=a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    if X.shape[1]!=14817: raise RuntimeError(f'{src}: unexpected genes {X.shape}')
    study=str(a.obs['study_id'].iloc[0]); d=OUT/study; d.mkdir(exist_ok=True)
    keep=a.obs['use_primary_radiation'].astype(bool).to_numpy() if 'use_primary_radiation' in a.obs.columns else np.ones(a.n_obs,dtype=bool)
    X=X[keep].tocsr().astype(np.int32); obs=a.obs.iloc[np.flatnonzero(keep)].copy(); status=arr_str(obs,'radiation_status'); is_c=np.char.equal(status.astype('U'),'control'); is_t=np.char.equal(status.astype('U'),'irradiated')
    if 'patient_model' in obs.columns: context=np.char.add(np.char.add(arr_str(obs,'study_id'),'|'),arr_str(obs,'patient_model'))
    elif 'cell_line' in obs.columns: context=np.char.add(np.char.add(arr_str(obs,'study_id'),'|'),arr_str(obs,'cell_line'))
    else: context=arr_str(obs,'study_id')
    dose=obs['dose_gy'].astype(float).fillna(0).to_numpy() if 'dose_gy' in obs else np.zeros(len(obs)); time_h=obs['time_post_ir_hours'].astype(float).fillna(0).to_numpy() if 'time_post_ir_hours' in obs else np.zeros(len(obs)); frac=obs['fraction_number'].astype(float).fillna(0).to_numpy() if 'fraction_number' in obs else np.where(is_t,1.0,0.0)
    rt=arr_str(obs,'radiation_type','none'); rcode=np.zeros(len(obs),dtype=np.float32)
    for i,s in enumerate(rt):
        z=s.lower(); rcode[i]=0.0 if z in ('none','','nan') else (1.0 if any(k in z for k in ('gamma','xray','x-ray','ionizing')) else 2.0)
    sp.save_npz(d/'matrix.npz',X,compressed=True)
    np.savez_compressed(d/'meta.npz',study_id=arr_str(obs,'study_id'),sample_id=arr_str(obs,'sample_id'),context_id=context.astype('U128'),condition=arr_str(obs,'condition'),radiation_status=status.astype('U64'),is_control=is_c,is_irradiated=is_t,dose_gy=dose.astype(np.float32),time_h=time_h.astype(np.float32),fraction=frac.astype(np.float32),radiation_code=rcode)
    summary={'study':study,'cells':int(X.shape[0]),'genes':int(X.shape[1]),'control_cells':int(is_c.sum()),'irradiated_cells':int(is_t.sum()),'contexts':sorted(set(context.astype(str))),'source':str(src)}
    (d/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
print('GPU_BUNDLES_READY',OUT)
