#!/usr/bin/env python3
\"\"\"Train a minimal sparse autoencoder on synthetic protein sequences.\"\"\"
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from utils.data_utils import generate_synthetic_dataset, one_hot_encode, save_fasta
from utils.model_utils import SparseAutoencoder, train_autoencoder

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def main():
    # config (minimal)
    cfg = {
        "n_sequences": 500,
        "seq_len": 50,
        "alphabet": "ACDEFGHIKLMNPQRSTVWY",
        "latent_dim": 16,
        "batch_size": 64,
        "epochs": 40,
        "lr": 1e-3,
        "sparsity_coef": 1e-3,
        "device": "cpu"
    }
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # Generate synthetic sequences and a simple synthetic label (sum of motif counts)
    seqs, ids = generate_synthetic_dataset(cfg["n_sequences"], cfg["seq_len"], cfg["alphabet"], motif="C")
    # motif 'C' count used as a dummy label (for surrogate training later)
    labels = np.array([s.count("C") for s in seqs], dtype=float)

    X = one_hot_encode(seqs, cfg["alphabet"])
    X = X.astype(np.float32)

    # Train SAE
    model = SparseAutoencoder(input_dim=X.shape[1], latent_dim=cfg["latent_dim"])
    device = torch.device(cfg["device"])
    train_autoencoder(model, X, device=device, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                      lr=cfg["lr"], sparsity_coef=cfg["sparsity_coef"], out_dir=OUT)

    # Save examples and labels for downstream scripts
    np.save(OUT / "sequences.npy", np.array(seqs))
    np.save(OUT / "ids.npy", np.array(ids))
    np.save(OUT / "labels.npy", labels)
    save_fasta(seqs, ids, OUT / "sequences.fasta")

    # Save config
    with open(OUT / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

if __name__ == '__main__':
    main()
