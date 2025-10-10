#!/usr/bin/env python3
\"\"\"Generate synthetic 'model activations' by passing one-hot sequence embeddings through a small encoder.
This simulates internal activations of a larger model; replace with your real activations if available.\"\"\"
import numpy as np
from pathlib import Path
from utils.data_utils import generate_synthetic_dataset, one_hot_encode
import torch
import torch.nn as nn

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def build_small_encoder(input_dim=1000, hidden=512, out_dim=256):
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
        nn.ReLU()
    )

def main():
    np.random.seed(0)
    torch.manual_seed(0)
    # synthetic sequences
    seqs, ids = generate_synthetic_dataset(n=1000, L=50, alphabet="ACDEFGHIKLMNPQRSTVWY", motif="C")
    X = one_hot_encode(seqs, "ACDEFGHIKLMNPQRSTVWY").astype(np.float32)  # N x (L*A)
    input_dim = X.shape[1]
    encoder = build_small_encoder(input_dim=input_dim, hidden=512, out_dim=256)
    with torch.no_grad():
        activations = encoder(torch.from_numpy(X)).numpy()
    # save
    np.save(OUT / "activations.npy", activations)
    np.save(OUT / "sequences.npy", np.array(seqs))
    np.save(OUT / "ids.npy", np.array(ids))
    print("Saved synthetic activations shape", activations.shape)

if __name__ == '__main__':
    main()
