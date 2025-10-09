#!/usr/bin/env python3
\"\"\"Fit DictionaryLearning on latents (or embeddings).\"\"\"
import numpy as np
from pathlib import Path
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import Lasso
import joblib

OUT = Path("outputs")

def main():
    latents = np.load(OUT / "latents.npy")
    n_atoms = min(32, latents.shape[0] // 2)
    dict_learner = DictionaryLearning(n_components=n_atoms, alpha=1.0, max_iter=500, random_state=0)
    codes = dict_learner.fit_transform(latents)
    atoms = dict_learner.components_
    np.save(OUT / "dictionary_atoms.npy", atoms)
    np.save(OUT / "sparse_codes.npy", codes)
    joblib.dump(dict_learner, OUT / "dictionary_model.joblib")
    print("Saved dictionary atoms and codes. atoms shape:", atoms.shape, "codes shape:", codes.shape)

if __name__ == '__main__':
    main()
