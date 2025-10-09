#!/usr/bin/env python3
\"\"\"Basic visualizations: PCA of latents and a simple latent traversal demo.\"\"\"
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
OUT = Path("outputs")

def main():
    latents = np.load(OUT / "latents.npy")
    pca = PCA(n_components=2)
    Z = pca.fit_transform(latents)
    plt.figure(figsize=(6,5))
    plt.scatter(Z[:,0], Z[:,1], s=10)
    plt.title("PCA of latent space")
    plt.savefig(OUT / "latent_pca.png")
    # latent traversal demo
    base = latents[0]
    traversal = [base + (i-5)*0.5 * (np.arange(len(base))==0).astype(float) for i in range(11)]
    traversal = np.stack(traversal)
    np.save(OUT / "latent_traversal.npy", traversal)
    plt.figure(figsize=(6,3))
    plt.plot(traversal)
    plt.title("Example latent traversal (atom 0)")
    plt.savefig(OUT / "latent_traversal_plot.png")
    print("Saved plots to outputs/")

if __name__ == '__main__':
    main()
