# ML Models Package - Complete Guide

Complete documentation for the ML models training pipeline.

---

## Table of Contents

1. [Package Overview](#package-overview)
2. [Training Models](#training-models)
3. [Model Outputs & Metrics](#model-outputs--metrics)
4. [Understanding Spearman Correlation](#understanding-spearman-correlation)
5. [Sample Limiting](#sample-limiting)
6. [Quick Reference](#quick-reference)

---

## Package Overview

### Directory Structure

```
ml_models/
├── __init__.py
├── train.py           # Core training logic
├── cli_train.py       # Command-line interface
├── encoding.py        # Feature encoding (onehot, kmer, ESM)
├── metrics.py         # Evaluation metrics
└── models.py          # Model builders (RF, XGBoost, etc.)
```

### Available Models

**Regression:**
- `linear` - Linear Regression
- `ridge` - Ridge Regression
- `lasso` - Lasso Regression
- `rf` - Random Forest
- `gb` - Gradient Boosting
- `svr` - Support Vector Regression
- `mlp` - Multi-Layer Perceptron
- `xgb` - XGBoost (if installed)
- `lgbm` - LightGBM (if installed)

**Classification:**
- `logreg` - Logistic Regression
- `svm` - Support Vector Machine
- `rf` - Random Forest
- `gb` - Gradient Boosting
- `mlp` - Multi-Layer Perceptron
- `xgb` - XGBoost (if installed)
- `lgbm` - LightGBM (if installed)

### Encoding Methods

- `onehot` - One-hot encoding (requires `--max-len`)
- `kmer` - K-mer frequency encoding (requires `--k`)
- `aac` - Amino acid composition
- `esm` - ESM embeddings (requires ESM model)

---

## Training Models

### Basic Usage

```bash
python -m ml_models.cli_train SEQUENCES_FILE \
    --labels LABELS_FILE \
    --model MODEL_NAME \
    --encoding ENCODING_METHOD \
    --task {regression|classification} \
    --outdir OUTPUT_DIR
```

### Example: Random Forest Regression

```bash
python -m ml_models.cli_train temp/sequences.npy \
    --labels temp/labels.npy \
    --model rf \
    --encoding onehot \
    --max-len 100 \
    --task regression \
    --outdir runs/rf_onehot \
    --seed 42
```

### All Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequences` | str | Required | Path to sequences (.npy or .csv) |
| `--labels` | str | None | Path to labels (.npy or .csv) |
| `--model` | str | `"xgb"` | Model name |
| `--encoding` | str | `"onehot"` | Encoding method |
| `--task` | str | Auto-detect | `"regression"` or `"classification"` |
| `--seed` | int | `42` | Random seed |
| `--outdir` | str | `"runs/ml_model"` | Output directory |
| `--test-size` | float | `0.2` | Test set fraction |
| `--n-samples` | int | None | Limit to N samples |
| `--k` | int | `3` | K-mer size (for kmer encoding) |
| `--max-len` | int | `512` | Max sequence length (for onehot) |
| `--split` | str | None | Reuse existing split file |
| `--no-save-split` | flag | False | Don't save split indices |
| `--stratify` | str | `"auto"` | Stratify split (`auto`, `yes`, `no`) |

---

## Model Outputs & Metrics

### Files Created

Every training run creates these files in `outdir/`:

```
runs/rf_onehot/
├── model.pkl       # Trained model (joblib)
├── scaler.pkl      # Feature scaler (joblib)
├── split.json      # Train/test indices
├── metrics.json    # Evaluation metrics
└── labels.pkl      # Label encoder (classification only)
```

### Regression Metrics (metrics.json)

```json
{
  "r2": 0.8234,
  "rmse": 0.4521,
  "mae": 0.3102,
  "spearman_rho": 0.8891,
  "spearman_p": 1.234e-15
}
```

**Metrics explained:**
- **`r2`** - R² coefficient (0-1, higher is better)
  - 1.0 = perfect predictions
  - 0.0 = model as good as predicting mean
  - > 0.7 = good model
  
- **`rmse`** - Root Mean Squared Error (lower is better)
  - Same units as target variable
  - Penalizes large errors
  
- **`mae`** - Mean Absolute Error (lower is better)
  - Average absolute difference
  - More robust to outliers
  
- **`spearman_rho`** - Spearman correlation (-1 to +1)
  - Measures rank-order correlation
  - > 0.8 = excellent ranking
  - > 0.7 = good ranking
  
- **`spearman_p`** - Statistical significance
  - < 0.001 = highly significant
  - < 0.05 = significant

### Classification Metrics (metrics.json)

```json
{
  "accuracy": 0.9200,
  "f1": 0.9150,
  "roc_auc": 0.9450
}
```

**Metrics explained:**
- **`accuracy`** - Fraction of correct predictions (0-1)
- **`f1`** - F1 score weighted average (0-1)
- **`roc_auc`** - ROC AUC score (binary classification only, 0-1)

### Accessing Results

**Load metrics:**
```python
import json

with open('runs/rf_onehot/metrics.json') as f:
    metrics = json.load(f)

print(f"R²: {metrics['r2']:.4f}")
print(f"Spearman ρ: {metrics['spearman_rho']:.4f}")
```

**Load model for predictions:**
```python
import joblib
import numpy as np
from ml_models.encoding import encode_onehot

# Load model and scaler
model = joblib.load('runs/rf_onehot/model.pkl')
scaler = joblib.load('runs/rf_onehot/scaler.pkl')

# Prepare sequences
sequences = np.array(["ACDEFGHIKLMNPQRSTVWY"], dtype=object)
X = encode_onehot(sequences, max_len=100)
X_scaled = scaler.transform(X)

# Predict
predictions = model.predict(X_scaled)
print(f"Prediction: {predictions[0]}")
```

---

## Understanding Spearman Correlation

### Why Spearman Matters

**Spearman correlation** measures monotonic (rank-order) relationships, making it ideal for protein engineering:

1. **Robust to non-linear scaling** - Captures trends even if predictions are in different units
2. **Measures what matters** - Relative ordering of sequences, not absolute values
3. **Critical for codirectionality** - Validates that model can rank sequences correctly

### Example: R² vs Spearman

**Scenario: Non-linear but Monotonic Model**
```
True values:    [1.0, 2.0, 3.0, 4.0, 5.0]
Predictions:    [1.0, 2.5, 4.5, 7.0, 10.0]

R² = 0.85             (good but not perfect)
Spearman ρ = 1.00     (perfect rank order!)
```

The model perfectly ranks sequences even though absolute predictions have non-linear scaling.

**Scenario: Poor Model**
```
True values:    [1.0, 2.0, 3.0, 4.0, 5.0]
Predictions:    [2.3, 1.8, 3.2, 2.9, 3.5]

R² = 0.12             (poor)
Spearman ρ = 0.40     (weak)
```

Both metrics agree the model is unreliable.

### Quality Thresholds

**For protein engineering applications:**

| Spearman ρ | Quality | Suitable for Use? |
|-----------|---------|-------------------|
| > 0.8 | Excellent | ✅ Yes, highly reliable |
| 0.7 - 0.8 | Good | ✅ Yes, reliable |
| 0.5 - 0.7 | Moderate | ⚠️ Proceed with caution |
| < 0.5 | Poor | ❌ Not recommended |

**p-value should be < 0.001** (highly significant)

### Using Spearman for Model Selection

```python
import json
from pathlib import Path

models = ['rf', 'xgb', 'lgbm']
results = []

for model in models:
    with open(f'runs/{model}_model/metrics.json') as f:
        metrics = json.load(f)
        metrics['model'] = model
        results.append(metrics)

# Sort by Spearman correlation (best ranking ability)
results.sort(key=lambda x: x['spearman_rho'], reverse=True)

print("Model Ranking by Spearman Correlation:")
for i, r in enumerate(results, 1):
    print(f"{i}. {r['model']}: ρ={r['spearman_rho']:.4f}, p={r['spearman_p']:.2e}")
```

---

## Sample Limiting

### Using --n-samples

Limit training to a subset of data for faster iteration:

```bash
# Use only first 30 samples
python -m ml_models.cli_train temp/sequences.npy \
    --labels temp/labels.npy \
    --model rf \
    --encoding onehot \
    --n-samples 30 \
    --outdir runs/rf_test
```

### Use Cases

**1. Fast Prototyping**
```bash
# Test different models quickly
for model in rf xgb lgbm; do
    python -m ml_models.cli_train sequences.npy --labels labels.npy \
        --model $model --n-samples 30 --outdir runs/${model}_test
done
```

**2. Learning Curves**
```bash
# Evaluate performance vs dataset size
for n in 20 50 100 200; do
    python -m ml_models.cli_train sequences.npy --labels labels.npy \
        --model rf --n-samples $n --outdir runs/rf_${n}samples
done
```

**3. Resource-Constrained Training**
```bash
# Train on subset if full dataset is too large
python -m ml_models.cli_train large_dataset.npy --labels labels.npy \
    --model rf --n-samples 1000 --outdir runs/rf_subset
```

### How It Works

```python
# In train.py:
if n_samples is not None and n_samples < len(df):
    df = df.iloc[:n_samples].copy()
    print(f"[INFO] Limited dataset to {n_samples} samples")
```

- Uses **first N samples** from dataset
- Applied **before** train/test split
- Works with both `.npy` and `.csv` files

---

## Quick Reference

### Common Workflows

**Quick test (fast):**
```bash
python -m ml_models.cli_train sequences.npy --labels labels.npy \
    --model rf --encoding onehot --max-len 100 --n-samples 30 \
    --outdir runs/quick_test
```

**Full training (production):**
```bash
python -m ml_models.cli_train sequences.npy --labels labels.npy \
    --model rf --encoding onehot --max-len 100 --task regression \
    --outdir runs/rf_production --seed 42
```

**Compare models:**
```bash
for model in rf xgb lgbm; do
    python -m ml_models.cli_train sequences.npy --labels labels.npy \
        --model $model --encoding onehot --max-len 100 \
        --outdir runs/${model}_model --seed 42
done

# Compare metrics
for model in rf xgb lgbm; do
    echo "=== $model ==="
    cat runs/${model}_model/metrics.json
done
```

### Quality Checks

Before using a model for downstream tasks:

```python
import json

with open('runs/rf_onehot/metrics.json') as f:
    m = json.load(f)

# Quality gates
checks = {
    "R² > 0.5": m['r2'] > 0.5,
    "Spearman ρ > 0.7": m['spearman_rho'] > 0.7,
    "p-value < 0.001": m['spearman_p'] < 0.001
}

print("Quality Checks:")
for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"  {status} {check}")

if all(checks.values()):
    print("\n✅ Model ready for use!")
else:
    print("\n⚠️ Consider improving model")
```

### Troubleshooting

**Low R² but high Spearman:**
- Model captures trends but has non-linear scaling
- ✅ Still useful for ranking sequences
- Consider: feature engineering, different model

**Low Spearman:**
- Model cannot reliably rank sequences
- ❌ Not suitable for optimization tasks
- Try: more training data, different encoding, different model

**High p-value (> 0.05):**
- Correlation not statistically significant
- Might be due to: small dataset, noisy labels, poor model
- Need: more data or better model

---

## Additional Resources

**For testing:**
- See `docs/testing/` for test suite documentation
- Run tests: `pytest tests/ml_models/test_model_output_consistency.py -v`

**For integration:**
- See `docs/INTEGRATION_CHECKLIST.md` for step-by-step guide

**For examples:**
- See individual test functions in `tests/ml_models/`
- See CLI usage in `ml_models/cli_train.py`

---

## Support

For issues or questions:
1. Check test outputs: `pytest tests/ml_models/ -v`
2. Review metrics in `metrics.json`
3. Verify data format with `encoding.py` functions
4. Check model outputs in `outdir/`

---

**Version:** 1.0  
**Last Updated:** 2025-10-22
