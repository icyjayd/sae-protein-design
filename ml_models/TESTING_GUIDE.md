# ML Models Testing - Complete Guide

Complete documentation for the ML models test suite.

---

## Table of Contents

1. [Test Overview](#test-overview)
2. [Running Tests](#running-tests)
3. [Test Architecture](#test-architecture)
4. [Testing with Real Data](#testing-with-real-data)
5. [What Gets Tested](#what-gets-tested)
6. [Troubleshooting](#troubleshooting)

---

## Test Overview

### Purpose

The test suite validates that all ML models:
1. ✅ Train without errors
2. ✅ Produce consistent output structures
3. ✅ Generate correct prediction shapes
4. ✅ Save all required artifacts
5. ✅ Handle edge cases properly

### Test Files

```
tests/ml_models/
├── conftest.py                          # Pytest configuration & fixtures
├── test_model_output_consistency.py     # Main test suite (31 tests)
└── README.md                            # This file
```

### Models Tested

**Regression:** `linear`, `ridge`, `lasso`, `rf`, `gb`, `svr`, `mlp`, `xgb`, `lgbm`  
**Classification:** `logreg`, `svm`, `rf`, `gb`, `mlp`, `xgb`, `lgbm`  
**Encodings:** `onehot`, `kmer`

---

## Running Tests

### Basic Usage

**Run all tests (synthetic data):**
```bash
pytest tests/ml_models/test_model_output_consistency.py -v
```

**Run specific test:**
```bash
# Test only Random Forest
pytest tests/ml_models/test_model_output_consistency.py -k "rf" -v

# Test only one-hot encoding
pytest tests/ml_models/test_model_output_consistency.py -k "onehot" -v

# Test n_samples functionality
pytest tests/ml_models/test_model_output_consistency.py -k "n_samples" -v
```

**Run with real data:**
```bash
pytest tests/ml_models/test_model_output_consistency.py \
    --sequences temp/sequences.npy \
    --labels temp/labels.npy \
    -v
```

**Run with limited samples:**
```bash
pytest tests/ml_models/test_model_output_consistency.py \
    --sequences temp/sequences.npy \
    --labels temp/labels.npy \
    --n-samples 30 \
    -v
```

### Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--sequences` | str | None | Path to sequences file (.npy or .csv) |
| `--labels` | str | None | Path to labels file (.npy or .csv) |
| `--n-samples` | int | None | Number of samples to use (default: all) |

### Expected Output

```
tests/ml_models/test_model_output_consistency.py::test_regression_model_consistency[linear-onehot] PASSED
tests/ml_models/test_model_output_consistency.py::test_regression_model_consistency[linear-kmer] PASSED
tests/ml_models/test_model_output_consistency.py::test_regression_model_consistency[rf-onehot] PASSED
...
tests/ml_models/test_model_output_consistency.py::test_n_samples_parameter PASSED
tests/ml_models/test_model_output_consistency.py::test_n_samples_edge_cases PASSED

====== 31 passed in 12.37s ======
```

---

## Test Architecture

### Test Categories

#### 1. Regression Model Tests (14 tests)
Tests each regression model with both encodings:
- `test_regression_model_consistency[MODEL-ENCODING]`
- Models: linear, ridge, lasso, rf, gb, svr, mlp
- Encodings: onehot, kmer

**What's validated:**
- Model trains successfully
- Predictions have shape `(n_samples,)`
- Predictions are floats
- Predictions are finite (no NaN/inf)

#### 2. Classification Model Tests (10 tests)
Tests each classification model with both encodings:
- `test_classification_model_consistency[MODEL-ENCODING]`
- Models: logreg, svm, rf, gb, mlp
- Encodings: onehot, kmer

**What's validated:**
- Model trains successfully
- Class predictions in {0, 1}
- Probabilities sum to 1.0
- Probabilities in range [0, 1]
- Correct output shapes

#### 3. Optional Model Tests (4 tests)
Tests XGBoost and LightGBM if installed:
- `test_optional_regression_models[xgb]`
- `test_optional_regression_models[lgbm]`
- `test_optional_classification_models[xgb]`
- `test_optional_classification_models[lgbm]`

Skipped if packages not installed.

#### 4. Split Consistency Test (1 test)
Validates that split files work correctly:
- `test_all_models_same_split`

**What's validated:**
- Models can reuse split files
- Different seeds don't affect split when file provided
- New splits are created when no file provided

#### 5. Sample Limiting Tests (2 tests)
Validates `--n-samples` parameter:
- `test_n_samples_parameter`
- `test_n_samples_edge_cases`

**What's validated:**
- Dataset correctly limited to N samples
- Train/test split correct on limited data
- Edge cases handled (None, oversized, minimum)

### Fixtures

#### `regression_data`
Creates or loads regression dataset:
- Synthetic: 50 random protein sequences with continuous labels
- Real: Loads from `--sequences` and `--labels`
- Returns: `(seq_path, label_path, n_samples, task="regression")`

#### `classification_data`
Creates or loads classification dataset:
- Synthetic: 50 random protein sequences with binary labels
- Real: Loads from `--sequences` and `--labels`
- Returns: `(seq_path, label_path, n_samples, task="classification")`

### Helper Functions

#### `check_model_outputs(outdir, n_samples, task)`
Validates output files and returns loaded artifacts:
```python
model, scaler, split = check_model_outputs(outdir, n_samples, task)
```

**Checks:**
- `model.pkl` exists and loads
- `scaler.pkl` exists and loads
- `split.json` exists and valid
- Split indices sum to n_samples
- No overlap between train/test

#### `_generate_regression_data(tmp_path, n_samples)`
Generates synthetic regression data for testing.

#### `_generate_classification_data(tmp_path, n_samples)`
Generates synthetic classification data for testing.

---

## Testing with Real Data

### Why Test with Real Data?

1. **Validate on actual sequences** - Ensures models work with your specific data format
2. **Realistic edge cases** - Real data may have unusual sequence lengths, characters, etc.
3. **Performance benchmarking** - See how models perform on your actual task
4. **Integration testing** - End-to-end validation with production data

### Setup

**Prepare data files:**
```bash
# Option 1: .npy files (recommended)
sequences.npy  # numpy array of strings
labels.npy     # numpy array of floats (regression) or ints (classification)

# Option 2: .csv files
sequences.csv  # must have 'sequence' column
labels.csv     # must have 'label' column
```

**Run tests:**
```bash
# Full dataset
pytest tests/ml_models/test_model_output_consistency.py \
    --sequences temp/sequences.npy \
    --labels temp/labels.npy \
    -v

# Subset for faster testing
pytest tests/ml_models/test_model_output_consistency.py \
    --sequences temp/sequences.npy \
    --labels temp/labels.npy \
    --n-samples 30 \
    -v
```

### Real Data Workflow

```bash
# 1. Quick validation (30 samples, fast)
pytest tests/ml_models/test_model_output_consistency.py \
    --sequences temp/sequences.npy \
    --labels temp/labels.npy \
    --n-samples 30 \
    -v

# 2. If passes, test full dataset
pytest tests/ml_models/test_model_output_consistency.py \
    --sequences temp/sequences.npy \
    --labels temp/labels.npy \
    -v

# 3. Test specific models only
pytest tests/ml_models/test_model_output_consistency.py \
    --sequences temp/sequences.npy \
    --labels temp/labels.npy \
    -k "rf or xgb" \
    -v
```

### Data Format Requirements

**Sequences (.npy):**
```python
import numpy as np

# Create from list
sequences = np.array([
    "ACDEFGHIKLMNPQRSTVWY",
    "MKTLLILAVITAIAAGALA",
    "ACACACACACACACACACAC"
], dtype=object)

np.save('sequences.npy', sequences)
```

**Labels (.npy):**
```python
import numpy as np

# Regression: continuous values
labels = np.array([1.2, -0.5, 3.4])

# Classification: integers
labels = np.array([0, 1, 0])

np.save('labels.npy', labels)
```

**Sequences (.csv):**
```csv
sequence
ACDEFGHIKLMNPQRSTVWY
MKTLLILAVITAIAAGALA
ACACACACACACACACACAC
```

**Labels (.csv):**
```csv
label
1.2
-0.5
3.4
```

---

## What Gets Tested

### File Structure Validation

Every test verifies these files are created:
```
outdir/
├── model.pkl       ✓ Exists, can be loaded
├── scaler.pkl      ✓ Exists, can be loaded
├── split.json      ✓ Exists, valid JSON
└── metrics.json    ✓ Exists, valid metrics (if generated)
```

### Split Validation

```python
# split.json structure
{
  "train_idx": [0, 2, 4, ...],  # Training indices
  "test_idx": [1, 3, 5, ...]    # Test indices
}
```

**Checks:**
- ✓ No overlap between train/test
- ✓ All indices < n_samples
- ✓ Sum of lengths = n_samples
- ✓ Approximately correct ratio (e.g., 80/20)

### Prediction Validation

**Regression:**
```python
predictions = model.predict(X_test)

# Checks:
assert predictions.shape == (n_test,)
assert np.issubdtype(predictions.dtype, np.floating)
assert np.all(np.isfinite(predictions))
```

**Classification:**
```python
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Checks:
assert predictions.shape == (n_test,)
assert all(p in [0, 1] for p in predictions)
assert probabilities.shape == (n_test, 2)
assert np.allclose(probabilities.sum(axis=1), 1.0)
assert np.all((probabilities >= 0) & (probabilities <= 1))
```

### Edge Cases Tested

#### n_samples Parameter
- ✓ `n_samples=None` uses all data
- ✓ `n_samples < total` limits correctly
- ✓ `n_samples > total` uses all data (no error)
- ✓ `n_samples=5` minimum viable size works

#### Data Edge Cases
- ✓ Short sequences (20 amino acids)
- ✓ Long sequences (50+ amino acids)
- ✓ Random sequence composition
- ✓ Continuous labels (regression)
- ✓ Binary labels (classification)

---

## Troubleshooting

### Common Issues

#### Issue: "Fixture 'real_or_synthetic_data' called directly"
**Cause:** Fixtures cannot call other fixtures  
**Fix:** Already fixed in current version

#### Issue: "ValueError: no option named '--task-type'"
**Cause:** Old conftest.py version  
**Fix:** Use updated conftest.py (option removed)

#### Issue: "ValueError: too many values to unpack"
**Cause:** Test expecting 3 values but fixture returns 4  
**Fix:** Update test to unpack: `seq_path, label_path, n_samples, task = fixture`

#### Issue: Tests are slow
**Solutions:**
```bash
# Use --n-samples to limit data
pytest tests/ml_models/test_model_output_consistency.py --n-samples 30 -v

# Test specific models only
pytest tests/ml_models/test_model_output_consistency.py -k "rf" -v

# Parallelize tests (if pytest-xdist installed)
pytest tests/ml_models/test_model_output_consistency.py -n auto
```

#### Issue: XGBoost/LightGBM tests fail
**Cause:** Packages not installed  
**Expected:** Tests should be skipped, not failed  
**Check:** Look for `SKIPPED` not `FAILED`

#### Issue: Model produces NaN predictions
**Possible causes:**
- Data contains NaN values
- Feature scaling issues
- Model convergence problems

**Debug:**
```python
import numpy as np

# Check input data
sequences = np.load('sequences.npy', allow_pickle=True)
labels = np.load('labels.npy')

print(f"Sequences: {len(sequences)}")
print(f"Labels: {len(labels)}")
print(f"Label range: [{labels.min()}, {labels.max()}]")
print(f"Any NaN: {np.any(np.isnan(labels))}")
```

### Debug Mode

Run tests with verbose output:
```bash
# Show print statements
pytest tests/ml_models/test_model_output_consistency.py -v -s

# Show full tracebacks
pytest tests/ml_models/test_model_output_consistency.py -v --tb=long

# Stop on first failure
pytest tests/ml_models/test_model_output_consistency.py -v -x

# Run specific test with debugging
pytest tests/ml_models/test_model_output_consistency.py::test_n_samples_parameter -v -s
```

### Verify Installation

```bash
# Check pytest
pytest --version

# Check sklearn
python -c "import sklearn; print(sklearn.__version__)"

# Check optional packages
python -c "import xgboost; print('XGBoost:', xgboost.__version__)" 2>/dev/null || echo "XGBoost not installed"
python -c "import lightgbm; print('LightGBM:', lightgbm.__version__)" 2>/dev/null || echo "LightGBM not installed"

# Check test discovery
pytest --collect-only tests/ml_models/
```

---

## Test Development

### Adding New Tests

**Template:**
```python
def test_new_feature(regression_data, tmp_path):
    """Test description"""
    seq_path, label_path, n_samples, task = regression_data
    outdir = tmp_path / "test_output"
    
    # Train model
    metrics = train_model(
        seq_file=str(seq_path),
        labels_file=str(label_path),
        model_name="rf",
        encoding="onehot",
        task=task,
        outdir=str(outdir),
    )
    
    # Validate
    assert (outdir / "model.pkl").exists()
    assert isinstance(metrics, dict)
    # Add more assertions...
```

### Best Practices

1. **Use fixtures** - Don't create data in tests
2. **Use tmp_path** - All outputs to temporary directories
3. **Clean assertions** - One concept per assertion
4. **Descriptive names** - `test_model_handles_short_sequences` not `test_1`
5. **Add docstrings** - Explain what and why

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Test ML Models

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
          
      - name: Run tests
        run: |
          pytest tests/ml_models/test_model_output_consistency.py -v
          
      - name: Test with real data (if available)
        run: |
          if [ -f data/sequences.npy ]; then
            pytest tests/ml_models/test_model_output_consistency.py \
              --sequences data/sequences.npy \
              --labels data/labels.npy \
              --n-samples 50 \
              -v
          fi
```

---

## Summary

**Test coverage:**
- ✅ 31 tests covering all models and encodings
- ✅ Regression and classification tasks
- ✅ Both synthetic and real data
- ✅ Edge cases and error conditions
- ✅ Output structure consistency

**Run tests:**
```bash
# Quick validation
pytest tests/ml_models/test_model_output_consistency.py -v

# With your data
pytest tests/ml_models/test_model_output_consistency.py \
    --sequences YOUR_SEQUENCES.npy \
    --labels YOUR_LABELS.npy \
    -v
```

**All tests passing = ready for production!** ✅

---

**Version:** 1.0  
**Last Updated:** 2025-10-22
