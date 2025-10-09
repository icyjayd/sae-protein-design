#!/usr/bin/env python3
\"\"\"Simulated active learning loop that uses surrogate predictions as synthetic labels for retraining.\"\"\"
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor

OUT = Path("outputs")

def main():
    latents = np.load(OUT / "latents.npy")
    labels = np.load(OUT / "labels.npy")
    n_init = 20
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(latents))
    pool = idx[n_init:]
    train_idx = idx[:n_init].tolist()

    rf = RandomForestRegressor(n_estimators=50, random_state=0)
    rf.fit(latents[train_idx], labels[train_idx])
    history = [rf.score(latents, labels)]
    for round in range(5):
        # generate candidates by random perturbation of pool samples
        cand_idx = rng.choice(pool, size=10, replace=False)
        cand = latents[cand_idx] + 0.1 * rng.randn(10, latents.shape[1])
        preds = rf.predict(cand)
        # pick top 2 by predicted value and add (simulated label = surrogate pred)
        top = np.argsort(preds)[-2:]
        for t in top:
            train_idx.append(cand_idx[t])
        # retrain
        rf.fit(latents[train_idx], labels[train_idx])
        history.append(rf.score(latents, labels))
    np.save(OUT / "active_learning_history.npy", np.array(history))
    print("Active learning simulated. history:", history)

if __name__ == '__main__':
    main()
