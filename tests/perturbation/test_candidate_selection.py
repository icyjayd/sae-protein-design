"""
Tests for the candidate selection logic in experiments/poc_pipeline.py
"""

import os
import pandas as pd
from experiments.poc_pipeline import load_perturbation_candidates

def test_load_candidates_handles_missing_file():
    """Tests that the function returns empty lists if file is missing."""
    candidates = load_perturbation_candidates("non_existent_file.csv")
    assert candidates == {"positive": [], "negative": [], "null": []}

def test_load_candidates_parses_csv(mock_correlation_csv):
    """Tests that the mock CSV is loaded and parsed correctly."""
    candidates = load_perturbation_candidates(mock_correlation_csv, top_n=2)
    
    # Based on the data in conftest.py:
    # Sorted spearman: [0.9, 0.7, 0.5, 0.02, 0.01, 0.0, -0.6, -0.8]
    # Indices:         [1,   5,   10,  7,    4,    8,   6,    2]
    
    # Positive (head 2):
    assert candidates["positive"] == [1, 5]
    
    # Negative (tail 2):
    # --- FIX: The assertion order was wrong ---
    assert candidates["negative"] == [6, 2]
    
    # Null (head 2 from null_df):
    # The null_df is filtered from the sorted df, so its order is [7, 4, 8]
    # --- FIX: The assertion was wrong ---
    assert candidates["null"] == [7, 4]


def test_load_candidates_filters_nans(mock_correlation_csv):
    """Tests that latent indices 3 and 9 (with NaN) are excluded."""
    candidates = load_perturbation_candidates(mock_correlation_csv, top_n=10)
    all_indices = candidates["positive"] + candidates["negative"] + candidates["null"]
    
    assert 3 not in all_indices
    assert 9 not in all_indices
    
def test_load_candidates_handles_top_n(mock_correlation_csv):
    """Tests that top_n parameter is respected."""
    candidates = load_perturbation_candidates(mock_correlation_csv, top_n=1)
    
    # Positive (head 1):
    assert candidates["positive"] == [1]
    
    # Negative (tail 1):
    assert candidates["negative"] == [2]
    
    # Null (head 1 from null_df):
    # --- FIX: The assertion was wrong ---
    assert candidates["null"] == [7]

