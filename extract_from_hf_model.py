
#!/usr/bin/env python3
import argparse, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm

def read_sequences(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--seqfile', required=True)
    ap.add_argument('--layer', type=int, default=-1)
    ap.add_argument('--pool', choices=['mean','cls'], default='mean')
    ap.add_argument('--outdir', default='outputs')
    ap.add_argument('--pca', action='store_true')
    ap.add_argument('--pca-dim', type=int, default=512)
    args = ap.parse_args()

    OUT = Path(args.outdir); OUT.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModel.from_pretrained(args.model); mdl.eval()

    seqs = read_sequences(args.seqfile)
    acts = []
    for seq in tqdm(seqs, desc='Extract'):
        toks = tok(seq, return_tensors='pt', add_special_tokens=True)
        with torch.no_grad():
            out = mdl(**toks, output_hidden_states=True)
        h = out.hidden_states[args.layer]
        pooled = h.mean(dim=1) if args.pool=='mean' else h[:,0,:]
        acts.append(pooled.squeeze(0).numpy())

    A = np.stack(acts)
    if args.pca:
        from sklearn.decomposition import PCA
        p = PCA(n_components=args.pca_dim).fit(A)
        A = p.transform(A)
        np.save(OUT/'pca_components.npy', p.components_)
        np.save(OUT/'pca_mean.npy', p.mean_)
    np.save(OUT/'activations.npy', A)
    np.save(OUT/'sequences.npy', np.array(seqs, dtype=object))
    print('Saved outputs/activations.npy and outputs/sequences.npy')

if __name__ == '__main__':
    main()
