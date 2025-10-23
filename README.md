
# 🧬 SAE Protein Design

### **AI-driven discovery in protein sequence space**

**SAE Protein Design** is an experimental research framework for exploring and steering protein representations using *Sparse Autoencoders (SAEs)* and *agentic AI orchestration*.  
It integrates latent-space manipulation, property prediction, and experiment automation into one cohesive system that learns how to generate, evaluate, and refine synthetic proteins.

The goal: **controllable protein engineering** — guiding sequence generation through interpretable latent features, with human or AI agents proposing and validating hypotheses in a continuous feedback loop.

---

## 🚀 Quick Summary

| Component | Purpose |
|------------|----------|
| **SAE module** | Learns interpretable latent codes from pretrained protein embeddings (e.g., ESM). |
| **ML models** | Predict properties like binding scores or stability from sequences or latent codes. |
| **Agentic Lab** | Orchestrates iterative experiments through autonomous agent loops. |
| **Scoring pipeline** | Evaluates synthetic sequences via internal ML models (e.g., Random Forest regressors). |
| **Diagnostics** | Provides metrics, run tracking, and codirectionality analyses. |
| **Model Scout (submodule)** | Benchmarks ML models automatically to find optimal predictors and encodings. |

---

## ⚙️ Installation

```bash
git clone --recurse-submodules https://github.com/icyjayd/sae-protein-design.git
cd sae-protein-design
conda create -n sae python=3.10
conda activate sae
pip install -r requirements.txt
```

If you’ve already cloned without submodules:
```bash
git submodule update --init --recursive
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

### 🔹 Example usage
From the project root:
```bash
model-scout data/GB1_sequences.npy --labels data/GB1_labels.npy --models rf ridge svr --encodings aac kmer --n-samples 1000 5000 all --jobs 5
```

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
- Using **Model Scout** to discover optimal predictive models for new protein datasets

---

## 📊 Output Artifacts

| Path | Description |
|------|--------------|
| `runs/*/memory.json` | Experiment metadata and metrics |
| `sae/outputs/` | Generated codes and reconstructions |
| `diagnostics/` | Analysis and plots |
| `proteng_model_scout/` | Submodule containing Model Scout package |

---

## 🧰 CLI and Integration
Many scripts can be run standalone or orchestrated via agents:
```bash
python agentic_lab/run_demo.py
```

The integrated pipeline now also supports direct model benchmarking:
```bash
model-scout data/GB1_sequences.npy --labels data/GB1_labels.npy --jobs 5
```

---

## 🧭 Roadmap

- [ ] Integrate structure prediction for 3D validation  
- [ ] Extend scoring to wet-lab-compatible ΔΔG metrics  
- [ ] Enhance agentic control loop for closed-loop optimization  
- [ ] Visualize Model Scout benchmarks in a live dashboard  

---

## 🧑‍🔬 Citation

If you use or extend this work, please cite:

> Irizarry-Cole, J. (2025). *SAE Protein Design: Agentic latent-space control for interpretable protein engineering.*
