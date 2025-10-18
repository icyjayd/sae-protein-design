# Protein Representation & Steering Pipeline
### (Sparse Autoencoders + ESM2 Embeddings)

---

## High-Level Summary

This repository implements a complete research pipeline for analyzing, manipulating, and steering protein representations using Sparse Autoencoders (SAEs) trained or fine-tuned on ESM2 embeddings.

It supports:
- Extracting embeddings from pretrained ESM2 or InterPLM models.
- Training and/or fine-tuning sparse and monosemantic autoencoders.
- Encoding sequences into latent space and analyzing latent–property correlations.
- Performing latent traversal and optimization-based steering to alter protein properties.
- Decoding modified latents back to activation or sequence space.
- Computing alignment-based reconstruction metrics.
- Generating comprehensive visual and statistical reports.

End-to-end execution is automated via `run_pipeline.py`, which loads pretrained InterPLM SAEs or trains local ones, runs analysis, and produces a PDF report.

---

## Module-by-Module Documentation

### 1. Core Models
#### `utils/model_utils.py`
Defines:
- `SparseAutoencoderSAE` – linear autoencoder with ReLU layers.
- `MonosemanticSAE` – adds top-k activation masking for sparsity.
  - Penalties:
    - `latent_decorrelation_loss()`
    - `decoder_orthonormal_loss()`
    - `decoder_unitnorm_loss()`
- Includes `get_device()` for GPU/CPU detection.

---

### 2. Representation Extraction
#### `extract_from_hf_model.py`
- Loads pretrained ESM2 models (Hugging Face).
- Extracts hidden activations and pooled embeddings.
- Saves `activations.npy`, `labels.npy`, and metadata.

#### `generate_activations.py`
- Generates synthetic activations for testing.
- Produces `activations.npy` and `labels.npy`.

---

### 3. Training
#### `train.py`
- Trains regular and monosemantic SAEs.
- Supports:
  - L1 sparsity
  - Latent decorrelation
  - Decoder orthonormality
  - Unit-norm constraints
- Saves weights and config JSONs (`sae_regular.pt`, `sae_mono.pt`).

---

### 4. Latent Code Extraction & Manipulation
#### `extract_codes.py`
- Encodes activations into latent space.
- Applies threshold sparsification.
- Saves latent codes (`sparse_codes_*.npy`) and decoder atoms (`sae_atoms_*.npy`).

#### `latent_traversal.py`
- Traverses along individual latent axes (±δ).
- Decodes each traversal point to activation space.
- Supports nearest-neighbor projection.

#### `optimize_latent.py`
- Evolutionary optimization of latent vectors.
- Uses 5 RF regressors to compute mean/std and risk-adjusted scores.
- Saves best latent (`opt_best_latent_*.npy`) and history.

#### `steer_latent_tool.py`
- Steers a specific latent dimension for a given sequence.
- Decodes to activation + sequence via ESM2.
- Saves results to `steering_result.json`.

---

### 5. Analysis and Visualization
#### `analysis_metrics.py`
- Computes:
  - Spearman correlations (latent–property)
  - Sparsity and diversity metrics
  - Surrogate model fits
- Produces:
  - CSV + JSON summaries
  - QQ plots + correlation plots
  - Optional combined PDF report

#### `qq_plot_pvalues.py`
- Reads correlation CSVs and generates Q-Q plots of p-values.

---

### 6. Orchestration
#### `run_pipeline.py`
- Main orchestrator:
  - Handles InterPLM and ESM2 model selection.
  - Copies artifacts `temp/` to structured `outputs_/`.
  - Runs training, extraction, analysis, and report generation.
- Outputs `pipeline_report.pdf` summarizing correlations and metrics.

---

### 7. Evaluation Utilities
#### `utils/grade_reconstructions.py`
- Computes sequence similarity metrics:
  - Identity, conservative substitutions (BLOSUM62), normalized alignment, Levenshtein similarity.
  - Aggregates into a final weighted score.
- `mean_grade()` writes per-sequence CSV reports.

#### `utils/esm_utils.py`
- Simplifies ESM2 usage:
  - `load_esm2_model()`, `encode_sequence()`, `decode_activation()`.

---

### 8. Testing
#### `tests/test_reconstruction.py`
- Pytest integration.
- Loads InterPLM SAEs and ESM2.
- Encodes+decodes protein sequences.
- Verifies mean reconstruction score > 0.95.
- Writes reconstruction reports to CSV.

---

## Typical Workflow

```bash
# 1. Extract activations
python extract_from_hf_model.py --model facebook/esm2_t6_8M_UR50D --out temp/

# 2. Train SAEs
python train.py --mode both --epochs 60

# 3. Extract latent codes
python extract_codes.py --mode monosemantic --threshold-pct 70

# 4. Analyze correlations
python analysis_metrics.py --outdir outputs/

# 5. Visualize p-values
python qq_plot_pvalues.py

# 6. Steer or optimize
python steer_latent_tool.py --seq-idx 0 --latent 12 --delta 1.5
python optimize_latent.py --iters 100 --risk 0.2

# 7. Full pipeline run
python run_pipeline.py --from_hf_model esm2-8m --gene GFP --property fluorescence
```

---

## Key Design Ideas

- Sparse, interpretable latents via monosemantic SAEs.
- Latent–property correlation discovery.
- Steering via latent modification.
- Verification with reconstruction & surrogate predictors.
- Extensible modular design for incremental experimentation.

---

## Outputs & Artifacts

| File | Description |
|------|--------------|
| `activations.npy` | ESM2 representations |
| `labels.npy` | Experimental properties |
| `sparse_codes_mono.npy` | Latent codes |
| `sae_mono.pt` | Monosemantic SAE weights |
| `latent_property_correlation_mono.csv` | Correlation metrics |
| `qqplot_latent_property_pvals_mono.png` | QQ plot |
| `opt_best_latent_mono.npy` | Optimized latent |
| `pipeline_report.pdf` | Summary report |

---

## Dependencies

- `torch`, `numpy`, `scikit-learn`, `matplotlib`, `pandas`
- `transformers`, `interplm`
- `pytest`

---

## Current Status

- Full functional pipeline (embeddings → latents → steering)
- Hugging Face + local model interoperability
- Visualization + metrics reporting
- Reconstruction quality testing
- Next: unify QQ plots, add property-prediction loop

---

## Next Steps

1. Merge `qq_plot_pvalues.py` into `analysis_metrics.py`.
2. Add property-predictor validation for steered sequences.
3. Improve decoding fidelity (LM sampling).
4. Auto-generate `run_config.json` for reproducibility.
5. Integrate experimental datasets (fluorescence, stability).

---

## Directory Structure
```
project_root/
├── utils/
│   ├── model_utils.py
│   ├── esm_utils.py
│   ├── grade_reconstructions.py
├── tests/
│   └── test_reconstruction.py
├── extract_from_hf_model.py
├── generate_activations.py
├── train.py
├── extract_codes.py
├── analysis_metrics.py
├── qq_plot_pvalues.py
├── latent_traversal.py
├── optimize_latent.py
├── steer_latent_tool.py
├── run_pipeline.py
├── outputs/
└── temp/
```

---

## One-Sentence Summary
A modular system for interpreting and steering protein representations via sparse autoencoders trained on ESM2 embeddings, analyzing latent–property links, and regenerating steered sequences for in-silico validation.

