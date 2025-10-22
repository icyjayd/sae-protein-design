# Integration Checklist

Use this checklist to integrate the test suite with your actual `sae-protein-design` codebase.

## Phase 1: Setup (5 minutes)

- [ ] Copy all test files to your project root directory
  ```bash
  cp test_ml_models.py pytest.ini requirements_testing.txt your-project/
  ```

- [ ] Install testing dependencies
  ```bash
  pip install -r requirements_testing.txt --break-system-packages
  ```

- [ ] Verify installation
  ```bash
  python validate_test_suite.py
  ```

## Phase 2: Understand Current Structure (10 minutes)

- [ ] Review your current codebase structure
  ```bash
  tree -L 3 sae-protein-design/
  ```

- [ ] Identify where models are saved
  - [ ] Random Forest location: _______________
  - [ ] SAE encoder location: _______________
  - [ ] SAE decoder location: _______________

- [ ] Identify where results are saved
  - [ ] Predictions location: _______________
  - [ ] Metrics location: _______________
  - [ ] Analysis location: _______________

- [ ] Document your CLI commands
  ```bash
  # Training RF:
  # _______________________________________________
  
  # Training SAE:
  # _______________________________________________
  
  # Running perturbation:
  # _______________________________________________
  ```

## Phase 3: Adapt Test Suite (30 minutes)

### Update File Paths

- [ ] Open `test_ml_models.py`
- [ ] Find `ModelOutputStructure` class (around line 250)
- [ ] Update file names to match your actual outputs:

```python
class ModelOutputStructure:
    RANDOM_FOREST = {
        'model_file': 'YOUR_ACTUAL_FILENAME.pkl',  # Update this
        'predictions_file': 'YOUR_PREDICTIONS.csv',  # Update this
        'metrics_file': 'YOUR_METRICS.json',        # Update this
        # ... etc
    }
```

### Update Data Configuration

- [ ] Find `DummyDataConfig` class (around line 30)
- [ ] Update to match your data:

```python
@dataclass
class DummyDataConfig:
    n_sequences: int = 100          # Your training set size
    sequence_length: int = 56       # Your protein length (GB1 = 56)
    latent_dim: int = 128          # Your SAE latent dimension
    # ... etc
```

### Update Expected Columns

- [ ] Check your actual CSV outputs
- [ ] Update expected columns in tests:

```python
# Example: Update RF predictions columns
'predictions_columns': ['sequence_id', 'true_score', 'predicted_score', 'residual']
# Change to match your actual columns
```

## Phase 4: Test with Dummy Data (15 minutes)

- [ ] Run data generation tests
  ```bash
  pytest test_ml_models.py -k TestDataGeneration -v
  ```
  Expected: All pass ✅

- [ ] Run file structure tests
  ```bash
  pytest test_ml_models.py -k TestDataFiles -v
  ```
  Expected: All pass ✅

- [ ] If any fail, debug and fix

## Phase 5: Train Models with Test Data (30 minutes)

- [ ] Generate test data using the test suite
  ```python
  from test_ml_models import DummyDataGenerator
  generator = DummyDataGenerator()
  sequences, scores = generator.generate_dataset()
  generator.save_dataset_csv('data/test_data.csv', sequences, scores)
  ```

- [ ] Train your Random Forest on test data
  ```bash
  # Use your actual CLI command here
  python -m sae_protein_design.ml_models.train --data data/test_data.csv ...
  ```

- [ ] Run RF tests
  ```bash
  pytest test_ml_models.py -k TestRandomForestModel -v
  ```

- [ ] Fix any issues:
  - [ ] File not found? → Update `model_file` path
  - [ ] Wrong columns? → Update `predictions_columns`
  - [ ] Wrong metrics? → Update `required_metrics`

- [ ] Train your SAE on test data
  ```bash
  # Use your actual CLI command here
  python -m sae_protein_design.sae.train --data data/test_sequences.fasta ...
  ```

- [ ] Run SAE tests
  ```bash
  pytest test_ml_models.py -k TestSAEModel -v
  ```

- [ ] Fix any issues (same process as RF)

## Phase 6: Test Perturbation Pipeline (20 minutes)

- [ ] Run perturbation experiment with test models
  ```bash
  # Use your actual CLI command here
  python -m sae_protein_design.experiments.perturbation ...
  ```

- [ ] Run perturbation tests
  ```bash
  pytest test_ml_models.py -k TestLatentPerturbation -v
  ```

- [ ] Fix any issues

## Phase 7: Full Pipeline Test (10 minutes)

- [ ] Run all tests together
  ```bash
  pytest test_ml_models.py -v
  ```

- [ ] Document any remaining failures
  ```
  Failing test: _________________
  Reason: _______________________
  Fix needed: ___________________
  ```

- [ ] Fix issues and re-run until all pass

## Phase 8: Real Data Integration (Optional)

If you want to test with real data instead of dummy data:

- [ ] Create a new fixture using real data
  ```python
  @pytest.fixture
  def real_gb1_data():
      df = pd.read_csv('path/to/real/gb1_data.csv')
      return {
          'sequences': df['sequence'].tolist(),
          'scores': df['binding_score'].values
      }
  ```

- [ ] Update tests to use real data fixture
- [ ] Run tests with real data
- [ ] Compare results to dummy data tests

## Phase 9: Agentic Integration (30 minutes)

- [ ] Review `integration_example.py`
- [ ] Adapt `ModelTrainingPipeline` class to your CLI
- [ ] Test programmatic execution:
  ```python
  from integration_example import ModelTrainingPipeline
  pipeline = ModelTrainingPipeline()
  results = pipeline.run_full_pipeline()
  ```

- [ ] Implement `AgenticTestRunner` in your workflow
  ```python
  from test_ml_models import AgenticTestRunner
  AgenticTestRunner.run_all_tests()
  ```

## Phase 10: Documentation (15 minutes)

- [ ] Document your specific setup in a project README
- [ ] Note any custom modifications to tests
- [ ] Document expected file structure for your team
- [ ] Add example commands for your specific pipeline

## Success Criteria

You're done when:

✅ All tests pass with dummy data
✅ All tests pass with trained models
✅ You can run tests programmatically via `AgenticTestRunner`
✅ CLI commands are documented and working
✅ Team members can run tests independently

## Troubleshooting Common Issues

### Issue: FileNotFoundError
**Problem**: Tests can't find model outputs
**Solution**: 
1. Check output directory structure
2. Update paths in `ModelOutputStructure`
3. Ensure your models save to the expected locations

### Issue: KeyError in CSV/JSON tests
**Problem**: Expected columns/keys missing
**Solution**:
1. Print actual columns: `df.columns.tolist()`
2. Update expected columns in test
3. Or modify your model to output expected format

### Issue: Shape mismatch in NumPy tests
**Problem**: Array dimensions don't match
**Solution**:
1. Check actual shape: `array.shape`
2. Update expected shape in test
3. Or verify your model is using correct dimensions

### Issue: Tests pass but models don't work
**Problem**: Tests validate structure, not functionality
**Solution**:
1. Tests validate form, not content - this is intentional
2. For functionality, need actual experimental validation
3. Tests ensure pipeline mechanics work correctly

## Next Steps After Integration

1. **Set up CI/CD**
   - Add tests to GitHub Actions
   - Run on every commit
   - Block merges if tests fail

2. **Expand test coverage**
   - Add tests for edge cases
   - Add tests for error handling
   - Add tests for data validation

3. **Integrate with agentic system**
   - Use `AgenticTestRunner` in automation
   - Generate reports automatically
   - Trigger experiments based on test results

4. **Share with team**
   - Document process
   - Train team members
   - Establish testing standards

## Estimated Time

- **Minimum viable integration**: 1-2 hours
- **Full integration with real data**: 3-4 hours
- **Agentic automation setup**: 2-3 hours
- **Total**: 6-9 hours for complete integration

## Questions During Integration

Use this space to note questions that arise:

1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

## Notes and Modifications

Document any changes you make to the test suite:

1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

---

**Remember**: The goal is to validate that your pipeline produces consistent, well-structured outputs. The tests don't need to pass on first try - they're a tool to help you identify and fix structural issues in your pipeline.
