"""
Pytest fixtures for the perturbation pipeline tests.
These fixtures create mock objects and temporary files
to simulate the behavior of real models and data.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import MagicMock

@pytest.fixture(scope="module")
def mock_correlation_csv(tmp_path_factory):
    """
    Creates a temporary CSV file with mock latent correlation data.
    This file is used to test the candidate selection logic.
    """
    # Create a temporary directory for this module's tests
    tmp_dir = tmp_path_factory.mktemp("data")
    csv_path = tmp_dir / "mock_correlations.csv"
    
    # Define mock data, including NaNs, positive, negative, and nulls
    data = {
        "latent_index": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "spearman": [0.9, -0.8, np.nan, 0.01, 0.7, -0.6, 0.02, 0.0, np.nan, 0.5],
        "p_value": [0.001, 0.002, 0.9, 0.8, 0.01, 0.02, 0.7, 0.9, 0.5, 0.03]
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path

# --- FIX: Replaced MagicMock with a real, pickle-able class ---
class SimpleMockModel:
    """A simple, pickle-able class to mock a scikit-learn model."""
    def predict(self, X):
        # Return a deterministic value for testing
        return np.array([1.23])

@pytest.fixture
def mock_surrogate_model(tmp_path):
    """
    Creates a simple, pickle-able mock model object,
    saves it to a .joblib file, and returns the path.
    """
    # Create an instance of our simple mock model
    mock_model_instance = SimpleMockModel()
    
    # Save the instance to a temporary file
    model_path = tmp_path / "mock_ridge_onehot.joblib"
    joblib.dump(mock_model_instance, model_path)
    
    # Return just the path. The test will load this file.
    return model_path

@pytest.fixture
def mock_esm_model():
    """Mocks the ESM (PLM) model utils."""
    mock = MagicMock()
    # 1. .embed() takes a sequence, returns a fake tensor
    mock_embedding = np.random.rand(1, 128).astype(np.float32)
    mock.embed.return_value = mock_embedding
    
    # 2. .decode() takes an embedding, returns a fake sequence
    mock.decode.return_value = "PERTURBEDSEQ"
    return mock

@pytest.fixture
def mock_sae_model():
    """Mocks the SAE model (encoder/decoder)."""
    mock = MagicMock()
    
    # 1. .encoder() takes an embedding, returns a fake latent vector
    # We return a simple numpy array. The .clone() call is
    # now fixed with np.copy() in the implementation.
    mock_latent = np.random.rand(1, 8192).astype(np.float32)
    mock.encoder.return_value = mock_latent
    
    # 2. .decoder() takes a latent, returns a fake embedding
    mock_embedding = np.random.rand(1, 128).astype(np.float32)
    mock.decoder.return_value = mock_embedding
    return mock

