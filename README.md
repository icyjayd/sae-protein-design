# SAE + Dictionary Learning Minimal Skeleton

This repository contains a minimal, runnable skeleton for a Sparse Autoencoder (PyTorch) + Dictionary Learning pipeline for protein sequence design. It uses synthetic data so you can run the entire pipeline without external resources. The goal is to provide a starting point you can expand later.

Run example quickstart:
```
python train_sae.py
python extract_latents.py
python learn_dictionary.py
python generate_and_evaluate.py
python visualize_results.py
```

Scripts:
- train_sae.py: trains a minimal SAE on synthetic sequences and saves model + latents.
- extract_latents.py: loads model and dataset to produce saved latents.npy.
- learn_dictionary.py: fits a scikit-learn DictionaryLearning model on the latents and saves atoms + codes.
- generate_and_evaluate.py: shows decoding of random latent vectors, trains a simple surrogate, and scores generated sequences.
- optimize_latent.py: simple random/evolutionary search placeholder for latent optimization.
- active_learning_sim.py: small simulated active-learning loop using surrogate predictions as synthetic labels.
- visualize_results.py: basic plots (PCA of latent space, latent traversal).

Requirements: Python 3.8+, PyTorch, scikit-learn, numpy, pandas, matplotlib, seaborn, python-docx (optional).

This is a minimal skeleton intended for extension.
