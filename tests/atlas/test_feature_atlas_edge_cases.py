import numpy as np
import pandas as pd
import pytest
from sae.analysis.feature_atlas import aggregate_per_sequence, compute_correlations

def test_aggregate_handles_empty_and_singleton():
    z1 = np.array([[1,2]])   # L=1, D=2
    z2 = np.array([[0,0],[0,0]])  # zeros
    Z = aggregate_per_sequence([z1, z2], method="mean")
    assert Z.shape == (2,2)
    np.testing.assert_allclose(Z[0], np.array([1.0,2.0]))
    np.testing.assert_allclose(Z[1], np.array([0.0,0.0]))

def test_compute_correlations_constant_property():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((5,3))
    y = np.ones(5)  # constant property
    df = compute_correlations(Z, y, method="spearman")
    # corr should be 0 and pval=1 (or near-1) when y is constant
    assert all(abs(df["spearman"]) < 1e-8)
    assert all(df["p_value"] >= 0.99)

def test_compute_correlations_with_nans():
    Z = np.array([[1.0, np.nan, 0.5],
                  [2.0, 0.3,   0.2],
                  [3.0, 0.2,   np.nan]])
    y = np.array([0.0, 1.0, 2.0])
    df = compute_correlations(Z, y, method="pearson")
    assert len(df) == 3
    # columns with many NaNs should still produce finite outputs or be handled gracefully
    assert set(df.columns) >= {"latent_index","spearman","p_value","n"}
    assert (df["n"] <= 3).all() and (df["n"] >= 1).all()
