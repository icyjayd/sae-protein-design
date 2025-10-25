# 🧬 SAE Protein Design

### **AI-driven discovery in protein sequence space**

**SAE Protein Design** is an experimental research framework for exploring and steering protein representations using *Sparse Autoencoders (SAEs)* and *agentic AI orchestration*.  
It integrates latent-space manipulation, property prediction, and experiment automation into one cohesive system that learns how to generate, evaluate, and refine synthetic proteins.

The goal: **controllable protein engineering** — guiding sequence generation through interpretable latent features.

---

## 🚀 Quick Summary

| Component | Intended Purpose |
|------------|----------|
| **SAE module** | Learns interpretable latent codes from pretrained protein embeddings (e.g., ESM). |
| **ML models** | Predict properties like binding scores or stability from sequences or latent codes. |
| **Agentic Lab** | Orchestrates iterative experiments through autonomous agent loops. (Skeleton only) |
| **Scoring** | Evaluates synthetic sequences via internal ML models (e.g., Random Forest regressors) and metrics. |
| **Diagnostics** | Provides metrics, run tracking, and codirectionality analyses. |
| **Model Scout (submodule)** | Benchmarks ML models automatically to find optimal predictors and encodings. |

---

## ⚙️ Installation

```bash
git clone --recurse-submodules https://github.com/icyjayd/sae-protein-design.git
cd sae-protein-design
conda create -n sae python=3.10
conda activate sae
```

If you’ve already cloned without submodules:
```bash
git submodule update --init --recursive
```

---

## 🚀 Environment Setup (Recommended)

After activating your environment, run the appropriate setup script for your operating system to automatically:

- Install all Poetry dependencies  
- Detect and install the correct **PyTorch** variant (CUDA / MPS / ROCm / CPU)  
- Clone and install **interPLM**  
- Install the **model-scout** submodule in editable mode  

### 🪟 **Windows (PowerShell)**
```powershell
.\setup.ps1
```

### 🐧 **macOS / Linux (bash/zsh)**
```bash
./setup.sh
```

These scripts are functionally equivalent and execute:

```bash
poetry install
poetry run python -m install_hooks
```

to build and configure the full environment automatically.

---

### 🧩 Manual Alternative
If you prefer not to use the setup scripts, you can run the same steps manually:

```bash
pip install poetry
poetry install
poetry run python -m install_hooks
```

---

## 🧠 Conceptual Overview

### 1. **Learning Latent Representations**
The `sae/` module trains sparse autoencoders on embeddings from pretrained protein models (e.g., ESM).  
These SAEs discover interpretable latent axes that often correspond to functional or structural protein features.

### 2. **Latent Steering & Generation**
Perturb latent dimensions and decode to sequence space to explore causal property directions.

### 3. **Predictive Scoring**
The `ml_models/` package trains lightweight predictors (e.g., random forests, ridge regression) for scoring protein properties.

### 4. **Agentic Experimentation**
The `agentic_lab/` module runs autonomous feedback loops that test and optimize sequences.

### 5. **Diagnostics & Tracking**
All experiments log structured metadata under `/runs`, enabling full reproducibility and comparison.

---

## 🧩 Model Scout Submodule

### 🔹 Overview
[`model-scout`](https://github.com/icyjayd/model-scout) is an independent benchmarking toolkit integrated as a submodule.  
It systematically tests combinations of models, encodings, and sample sizes to determine which predictive setup performs best on a given dataset.

### 🔹 Core features
- Unified CLI (`model-scout.exe`) for running full model sweeps  
- Automatic handling of encodings (`aac`, `kmer`, `onehot`, etc.)  
- Parallelized multi-model testing (`--jobs N`)  
- Built-in Spearman correlation metrics with intelligent warning handling  
- Easy integration into any ML or protein design workflow



This will:
- Preview the first 10 sequences and labels  
- Train all combinations of models and encodings  
- Report Spearman ρ, p-value, and timing per run  
- Save results automatically under `/runs/model-scout/`

### 🔹 Updating the submodule
To pull the latest version of Model Scout:
```bash
git submodule update --remote --merge
```

---

## 🧪 Current Research Focus
- Validating **codirectionality** between latent perturbations and predicted binding scores  
- Testing whether SAEs encode **causal** features for controllable property optimization  


### 🔹 Example usage
The integrated pipeline now also supports direct model benchmarking.

From the project root:
```bash
model-scout data/GB1_sequences.npy --labels data/GB1_labels.npy --models rf ridge svr --encodings aac kmer --n-samples 1000 5000 all --jobs 5
```

## Project Status (v0.10)

- **InterPLM SAE integrated:** operational with ESM2 interface.
- **Feature Atlas completed:** computes latent–property correlations and outputs CSV compatible with PoC pipeline.
- **Next phase:** implement and test latent-effect evaluation (perturb → decode → measure property delta).

**Next Steps**
1. Add `evaluate_latent_effects.py` and unit tests.
2. Integrate perturbation results into PoC pipeline.
3. Prepare for cross-family validation and plotting utilities.


## 🧑‍🔬 Citation

If you use or extend this work, please cite:

> Irizarry-Cole, J. (2025). *SAE Protein Design: Agentic latent-space control for interpretable protein engineering.*
