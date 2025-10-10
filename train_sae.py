#!/usr/bin/env python3
\"\"\"Train a sparse autoencoder on activations (L1 sparsity on latent) to learn pseudo-dictionary features.\"\"\"
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.model_utils import SparseAutoencoderSAE
import json

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def train(model, X, epochs=50, batch_size=128, lr=1e-3, sparsity_coef=1e-3):
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(X.astype('float32')))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for ep in range(1, epochs+1):
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad()
            recon, z = model(xb)
            loss_recon = mse(recon, xb)
            loss_sparse = sparsity_coef * torch.mean(torch.abs(z))
            loss = loss_recon + loss_sparse
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        if ep % 10 == 0 or ep==1:
            print(f"Epoch {ep}/{epochs} avg_loss={total/len(X):.6f}")
    # save
    torch.save(model.state_dict(), OUT / "sae_activations_model.pt")
    with open(OUT / "sae_config.json", "w") as f:
        json.dump({"latent_dim": model.latent_dim, "input_dim": model.input_dim}, f)
    print("Saved model and config")

def main():
    acts = np.load(OUT / "activations.npy")
    model = SparseAutoencoderSAE(input_dim=acts.shape[1], latent_dim=64, hidden=512)
    train(model, acts, epochs=60, batch_size=128, lr=1e-3, sparsity_coef=1e-3)

if __name__ == '__main__':
    main()
