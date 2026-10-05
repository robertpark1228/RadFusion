"""Extract cell embeddings or predict embeddings under irradiation."""
import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from radfusion_gpu_common import model_from_checkpoint_dict, ConditionTransport, normalize_dense, condition_vector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts', required=True, help='SciPy CSR NPZ; columns must match reference/vocabulary.tsv')
    p.add_argument('--encoder', required=True)
    p.add_argument('--transport')
    p.add_argument('--dose-gy', type=float)
    p.add_argument('--time-h', type=float)
    p.add_argument('--fraction', type=float)
    p.add_argument('--radiation-code', type=float)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--out', required=True, help='Output cell embeddings in NPY format')
    a = p.parse_args()
    if a.batch < 1:
        p.error('--batch must be positive')
    conditions = [a.dose_gy, a.time_h, a.fraction, a.radiation_code]
    if a.transport and any(v is None or not np.isfinite(v) or v < 0 for v in conditions):
        p.error('transport requires four finite nonnegative condition values; use the training radiation-code mapping')
    x = sp.load_npz(a.counts).tocsr()
    if x.shape[0] == 0 or not np.isfinite(x.data).all() or (x.data < 0).any():
        p.error('counts must contain nonempty, finite, nonnegative raw counts')
    ck = torch.load(a.encoder, map_location='cpu', weights_only=True)
    encoder, cfg = model_from_checkpoint_dict(ck)
    if x.shape[1] != cfg['genes']:
        p.error(f"Expected {cfg['genes']} ordered genes, got {x.shape[1]}")
    device = torch.device(a.device)
    encoder = encoder.to(device).eval()
    transport = None
    if a.transport:
        tc = torch.load(a.transport, map_location='cpu', weights_only=True)
        transport = ConditionTransport(**tc['transport_config']).to(device).eval()
        transport.load_state_dict(tc['transport'], strict=True)
    result = np.lib.format.open_memmap(a.out, mode='w+', dtype=np.float32, shape=(x.shape[0], cfg['latent']))
    with torch.inference_mode():
        for start in range(0, x.shape[0], a.batch):
            batch = torch.from_numpy(x[start:start+a.batch].toarray().astype(np.float32)).to(device)
            z = encoder.encode(normalize_dense(batch))
            if transport is not None:
                c = condition_vector(*[np.full(len(batch), v, np.float32) for v in conditions], device)
                z = transport(z, c)
            result[start:start+len(batch)] = z.cpu().numpy()
    result.flush()
    print(f'Saved {x.shape[0]} latent vectors to {Path(a.out)}')


if __name__ == '__main__':
    main()
