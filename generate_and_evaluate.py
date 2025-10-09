#!/usr/bin/env python3
\"\"\"Decode random latents, train a simple surrogate, and score generated sequences.\"\"\"
import numpy as np
from pathlib import Path
import joblib
import torch
from sklearn.ensemble import RandomForestRegressor
from utils.model_utils import SparseAutoencoder, decode_latent_batch
from utils.data_utils import sequences_from_onehot, save_fasta

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def main():
    latents = np.load(OUT / "latents.npy")
    seqs = np.load(OUT / "sequences.npy", allow_pickle=True)
    labels = np.load(OUT / "labels.npy", allow_pickle=True)
    cfg = np.load(OUT / "config.json", allow_pickle=True).item()
    # Train surrogate on latents -> labels
    # simple split
    n = len(latents)
    idx = np.arange(n)
    np.random.shuffle(idx)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    Xtr, ytr = latents[tr], labels[tr]
    Xte, yte = latents[te], labels[te]

    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(Xtr, ytr)
    joblib.dump(rf, OUT / "surrogate_rf.joblib")
    print("Surrogate RF trained, test R2:", rf.score(Xte, yte))

    # Generate candidate latents by adding noise to real latents
    rng = np.random.RandomState(0)
    cand = latents[rng.choice(len(latents), size=50)] + 0.1 * rng.randn(50, latents.shape[1])

    # decode with model decoder
    device = torch.device(cfg.get("device", "cpu"))
    model = SparseAutoencoder(input_dim=Xtr.shape[1] * 1, latent_dim=cfg.get("latent_dim", 16))
    model_path = OUT / "sae_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # decode and map to sequences (approx)
    candidates = []
    for z in cand:
        seq = decode_latent_batch(model, torch.from_numpy(z.astype(np.float32)).unsqueeze(0)).squeeze(0)
        candidates.append(seq)
    save_fasta(candidates, [f"cand_{i}" for i in range(len(candidates))], OUT / "generated_sequences.fasta")

    # featurize candidates using encoder to get latent (sanity) and score with surrogate
    with torch.no_grad():
        zs = model.encode(torch.from_numpy(np.stack([np.load(OUT / 'latents.npy')[0]] * len(candidates), dtype=np.float32)))
    # (Here we reuse the noisy cand as features since this is a minimal example)
    preds = rf.predict(cand)
    import pandas as pd
    df = pd.DataFrame({"id": [f"cand_{i}" for i in range(len(candidates))], "predicted_label": preds})
    df.to_csv(OUT / "predictions.csv", index=False)
    print("Saved predictions.csv and generated_sequences.fasta")

if __name__ == '__main__':
    main()
