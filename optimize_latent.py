#!/usr/bin/env python3
\"\"\"Simple random/evolutionary search over latent space using surrogate for scoring.\"\"\"
import numpy as np
from pathlib import Path
import joblib

OUT = Path("outputs")

def main():
    rf = joblib.load(OUT / "surrogate_rf.joblib")
    latents = np.load(OUT / "latents.npy")
    rng = np.random.RandomState(1)
    best = None
    best_score = -1e9
    # simple evolutionary loop: mutate random latents and keep top
    for gen in range(50):
        parent = latents[rng.randint(len(latents))]
        pop = parent + 0.2 * rng.randn(20, latents.shape[1])
        scores = rf.predict(pop)
        idx = np.argmax(scores)
        if scores[idx] > best_score:
            best_score = scores[idx]
            best = pop[idx]
    np.save(OUT / "optimized_latent.npy", best)
    print("Saved optimized latent with predicted score", best_score)

if __name__ == '__main__':
    main()
