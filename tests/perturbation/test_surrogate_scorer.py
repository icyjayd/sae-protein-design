"""
Tests for the SurrogateScorer class in scoring/surrogate_models.py
"""

import pytest
import numpy as np
from unittest.mock import patch
from scoring.surrogate_models import SurrogateScorer

def test_scorer_initialization(mock_surrogate_model):
    """Tests that the scorer class loads the mock model file."""
    model_path = mock_surrogate_model
    
    scorer = SurrogateScorer(model_path, encoding_config={"encoding": "onehot"})
    assert scorer.model is not None
    # Check that the loaded model has the .predict method
    assert hasattr(scorer.model, "predict")

def test_scorer_handles_missing_model_file():
    """Tests that the scorer initializes with model=None if file is missing."""
    scorer = SurrogateScorer("non_existent_model.joblib", encoding_config={})
    assert scorer.model is None

def test_scorer_predicts_score(mock_surrogate_model, mocker):
    """
    Tests that the .score() method calls the encoding function
    and the model's .predict() method.
    """
    model_path = mock_surrogate_model
    config = {"encoding": "onehot", "max_len": 100}
    scorer = SurrogateScorer(model_path, config)
    
    # Mock the encode_sequences function from the ml_models.encoding module
    mock_encoder = mocker.patch(
        "scoring.surrogate_models.encode_sequences", 
        return_value=np.array([[0, 1, 0]]))
    
    # --- FIX: Spy on the *loaded model instance* ---
    # We spy on the 'predict' method of the model object *inside* the scorer
    predict_spy = mocker.spy(scorer.model, "predict")
    
    test_sequences = ["TESTSEQ"]
    scores = scorer.score(test_sequences)
    
    # 1. Assert encode_sequences was called correctly
    mock_encoder.assert_called_once_with(test_sequences, **config)
    
    # 2. Assert model.predict was called with the output of the encoder
    predict_spy.assert_called_once()
    # Check the argument passed to the spied 'predict' method
    np.testing.assert_array_equal(
        predict_spy.call_args[0][0], 
        np.array([[0, 1, 0]])
    )
    
    # 3. Assert the final score is what the mock model returned
    assert scores == np.array([1.23])

def test_scorer_raises_error_if_no_model(mocker):
    """Tests that .score() raises a RuntimeError if the model wasn't loaded."""
    scorer = SurrogateScorer("non_existent_model.joblib", encoding_config={})
    assert scorer.model is None
    
    # Mock the encoder just to get past that step
    mocker.patch(
        "scoring.surrogate_models.encode_sequences", 
        return_value=np.array([[0, 1, 0]]))
    
    with pytest.raises(RuntimeError, match="model is not loaded"):
        scorer.score(["TESTSEQ"])

