
#!/usr/bin/env python3
import argparse, json, numpy as np, torch
from pathlib import Path
from utils.model_utils import SparseAutoencoderSAE, MonosemanticSAE
OUT = Path('outputs')
def nearest_neighbor_seq(decoded_act, acts, seqs):
    d2 = np.sum((acts - decoded_act[None, :])**2, axis=1)
    idx = int(np.argmin(d2)); return seqs[idx], idx, float(d2[idx])
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', choices=['regular','mono'], default='mono')
    ap.add_argument('--latent-dim-index', type=int, default=0)
    ap.add_argument('--steps', type=int, default=9)
    ap.add_argument('--step-size', type=float, default=1.0)
    args = ap.parse_args()
    acts = np.load(OUT/'activations.npy'); seqs = np.load(OUT/'sequences.npy', allow_pickle=True).tolist()
    if args.suffix=='regular':
        cfg = json.load(open(OUT/'sae_regular_config.json'))
        model = SparseAutoencoderSAE(cfg['input_dim'], cfg['latent_dim'], cfg['hidden'])
        model.load_state_dict(torch.load(OUT/'sae_regular.pt', map_location='cpu'))
    else:
        cfg = json.load(open(OUT/'sae_mono_config.json'))
        model = MonosemanticSAE(cfg['input_dim'], cfg['latent_dim'], cfg['hidden'], topk=cfg.get('topk'))
        model.load_state_dict(torch.load(OUT/'sae_mono.pt', map_location='cpu'))
    model.eval()
    z0 = np.zeros(cfg['latent_dim'], dtype=np.float32); half = args.steps//2; zs = []
    for k in range(-half, half+1):
        z = z0.copy(); z[args.latent_dim_index] = k * args.step_size; zs.append(z)
    zs = torch.from_numpy(np.stack(zs))
    with torch.no_grad(): dec = model.decode(zs).numpy()
    with open(OUT/f'latent_traversal_{args.suffix}.txt','w') as f:
        f.write('step\tseq_index\tl2dist\tsequence\n')
        for i in range(dec.shape[0]):
            seq, idx, d = nearest_neighbor_seq(dec[i], acts, seqs)
            f.write(f'{i}\t{idx}\t{d:.4f}\t{seq}\n')
    print('Saved traversal file.')
if __name__ == '__main__':
    main()
