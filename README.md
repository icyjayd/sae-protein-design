# SAE on Model Activations (Anthropic-style pseudo-dictionary learning)

Minimal runnable codebase that demonstrates:
- Generating synthetic "model activations" using a small transformer-like encoder (to simulate activations from a larger model).
- Training a Sparse Autoencoder (SAE) on those activations with an L1 sparsity penalty to learn monosemantic / dictionary-like features.
- Extracting atoms (decoder basis) and sparse codes (latent activations) for analysis.

Run quickstart:
```
python generate_activations.py        # create synthetic activations (or replace with real activations.npy)
python train_sae.py                   # train SAE on activations and save model + outputs
python extract_codes.py               # get sparse codes and atoms (pseudo-dictionary)
python visualize.py                   # quick visualization (PCA + atom heatmap)
```

Requirements:
- Python 3.8+
- PyTorch, numpy, scikit-learn, matplotlib

Notes:
- If you have real model activations, save them as `outputs/activations.npy` (shape N x D) and skip `generate_activations.py`.
- This is a minimal educational skeleton; extend with your real model hooks and AlphaFold/FoldX evaluations as needed.
