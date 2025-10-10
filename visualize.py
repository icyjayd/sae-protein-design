#!/usr/bin/env python3
\"\"\"Quick visualizations: PCA of activations, sample atom heatmap.\"\"\"
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
OUT = Path("outputs")

def main():
    acts = np.load(OUT / "activations.npy")
    atoms = np.load(OUT / "sae_atoms.npy")
    pca = PCA(n_components=2).fit_transform(acts)
    plt.figure(figsize=(6,5))
    plt.scatter(pca[:,0], pca[:,1], s=6)
    plt.title("PCA of activations")
    plt.savefig(OUT / "activations_pca.png")
    # atom heatmap (first 16 atoms)
    plt.figure(figsize=(8,6))
    plt.imshow(atoms[:16], aspect='auto')
    plt.colorbar()
    plt.title("First 16 SAE atoms (decoder columns)")
    plt.xlabel("activation feature index")
    plt.ylabel("atom index")
    plt.savefig(OUT / "atoms_heatmap.png")
    print('Saved activations_pca.png and atoms_heatmap.png')

if __name__ == '__main__':
    main()
