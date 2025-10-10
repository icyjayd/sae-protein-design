#!/usr/bin/env python3
\"\"\"Encode activations with trained SAE, save latent codes and optionally perform L1-based thresholding to get sparse codes.\"\"\"
import numpy as np
from pathlib import Path
import torch
from utils.model_utils import SparseAutoencoderSAE
OUT = Path("outputs")

def main():
    acts = np.load(OUT / "activations.npy")
    import json
    cfg = json.load(open(OUT / "sae_config.json"))
    model = SparseAutoencoderSAE(input_dim=acts.shape[1], latent_dim=cfg.get("latent_dim", 64), hidden=512)
    model.load_state_dict(torch.load(OUT / "sae_activations_model.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        zs = model.encode(torch.from_numpy(acts.astype('float32'))).numpy()
    # simple sparsification: threshold small values to zero to mimic monosemantic activations
    thresh = np.percentile(np.abs(zs), 70)
    sparse_codes = zs * (np.abs(zs) >= thresh)
    np.save(OUT / "sparse_codes.npy", sparse_codes)
    # treat decoder columns as "atoms" by passing unit vectors through decoder
    atoms = []
    for i in range(model.latent_dim):
        unit = torch.zeros((1, model.latent_dim))
        unit[0, i] = 1.0
        with torch.no_grad():
            atom = model.decode(unit).numpy().squeeze(0)
        atoms.append(atom)
    atoms = np.stack(atoms)
    np.save(OUT / "sae_atoms.npy", atoms)
    print("Saved sparse_codes.npy and sae_atoms.npy with shapes", sparse_codes.shape, atoms.shape)

if __name__ == '__main__':
    main()
