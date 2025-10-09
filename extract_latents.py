#!/usr/bin/env python3
\"\"\"Load trained SAE and extract latent representations for the dataset.\"\"\"
import numpy as np
import torch
from pathlib import Path
from utils.data_utils import load_saved_onehot
from utils.model_utils import SparseAutoencoder

OUT = Path("outputs")

def main():
    cfg = np.load(OUT / "config.json", allow_pickle=True).item() if (OUT / "config.json").exists() else {}
    device = torch.device(cfg.get("device", "cpu"))
    # load model
    model = SparseAutoencoder(input_dim=cfg.get("seq_len", 50) * 20, latent_dim=cfg.get("latent_dim", 16))
    model_path = OUT / "sae_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # load data one-hot
    X = load_saved_onehot(OUT / "sequences.npy", cfg.get("alphabet", "ACDEFGHIKLMNPQRSTVWY"))
    X = X.astype(np.float32)
    with torch.no_grad():
        latents = model.encode(torch.from_numpy(X)).cpu().numpy()
    np.save(OUT / "latents.npy", latents)
    print("Saved latents.npy with shape", latents.shape)

if __name__ == '__main__':
    main()
