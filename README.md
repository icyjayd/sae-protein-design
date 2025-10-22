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

---

## ⚙️ Installation

```bash
git clone https://github.com/icyjayd/sae-protein-design.git
cd sae-protein-design
conda create -n sae python=3.10
conda activate sae
pip install -r requirements.txt
```

---

## 🧠 Conceptual Overview

### 1. **Learning Latent Representations**
The `sae/` module trains sparse autoencoders on embeddings from pretrained protein models (e.g., ESM).  
These SAEs discover interpretable latent axes that often correspond to functional or structural protein features.

- `sae/train.py` — trains the SAE model.  
- `sae/extract_from_hf_model.py` — extracts ESM activations for downstream use.  
- `sae/extract_codes.py` — encodes sequences into sparse latent vectors.  

### 2. **Latent Steering & Generation**
By perturbing specific latent dimensions and decoding back to sequence space, the system can test whether a direction correlates with a property change (like binding energy).

- `sae/latent_traversal.py` — generates perturbations along chosen latents.  
- `sae/optimize_latent.py` — performs targeted optimization or traversal.  
- `sae/steer_latent_tool.py` — provides a CLI-style interface for automated experiments.

### 3. **Predictive Scoring**
The `ml_models/` package contains lightweight ML pipelines that predict protein properties directly from sequences or latent codes.

- `ml_models/train.py` — trains property predictors (e.g., random forest regressors).  
- `ml_models/encoding.py` — handles sequence embeddings.  
- `ml_models/metrics.py` — evaluates predictive performance.  
- `ml_models/cli_train.py` — provides a command-line training interface.

### 4. **Agentic Experimentation**
The `agentic_lab/` module is an early prototype of a self-directed experimental loop.  
It can run iterative optimization routines that test, score, and adjust candidate sequences using agents that plan, critique, and improve runs.

- `agentic_lab/agents.py` — defines modular AI agents for planning and analysis.  
- `agentic_lab/loop.py` — orchestrates agent interactions and state transitions.  
- `agentic_lab/run_demo.py` — runs a minimal working demo of the full loop.  

### 5. **Diagnostics & Tracking**
The `diagnostics/` and `runs/` directories store experiment logs and output artifacts.  
Each run records metrics, predictions, and latent statistics in structured JSONs (`memory.json`) for reproducibility and post-hoc analysis.

---

## 🧩 Example Workflow

### 🔹 Step 1 — Extract embeddings
```bash
python sae/extract_from_hf_model.py --model esm2_t33_650M_UR50D --sequences data/gb1.fasta
```

### 🔹 Step 2 — Train a sparse autoencoder
```bash
python sae/train.py --input activations.npy --output sae_model.pt
```

### 🔹 Step 3 — Train a property predictor
```bash
python ml_models/train.py --data dataset.csv --labels binding_scores.csv
```

### 🔹 Step 4 — Perturb latent features and generate sequences
```bash
python sae/latent_traversal.py --model sae_model.pt --latents 12 42 85
```

### 🔹 Step 5 — Score synthetic sequences
```bash
python scoring/evaluate_sequences.py --model rf_model.pkl --input generated_sequences.fasta
```

---

## 🧪 Current Research Focus

- Validating **codirectionality** between latent perturbations and model-predicted binding scores.  
- Testing whether SAEs encode **causal** features that can guide controllable property optimization.  
- Developing **agentic orchestration** that automatically proposes and evaluates new latent experiments.

---

## 📊 Output Artifacts
| Path | Description |
|------|--------------|
| `runs/*/memory.json` | Tracks all parameters, metrics, and agent decisions per run. |
| `sae/outputs/` | Generated latent codes, reconstructions, and visualizations. |
| `diagnostics/` | Plots and metrics for model debugging. |

---

## 🧰 CLI and Integration
Many scripts can be run standalone or through an integrated agentic workflow:

```bash
python agentic_lab/run_demo.py
```

The demo agent trains, tests, and scores models autonomously, logging its reasoning and results in `/runs`.

---

## 🧭 Roadmap

- [ ] Integrate structure prediction (e.g., via AlphaFold or ESMFold) for 3D validation.  
- [ ] Extend scoring to wet-lab-compatible ΔΔG and stability metrics.  
- [ ] Refine agentic control loop for closed-loop protein optimization.  
- [ ] Implement visualization dashboard for latent and property space trajectories.  

---

## 🧑‍🔬 Citation

If you use or extend this work, please cite it as:

> Irizarry-Cole, J. (2025). *SAE Protein Design: Agentic latent-space control for interpretable protein engineering.*
