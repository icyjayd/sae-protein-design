"""
Fixtures for the pipeline integration tests.

Fixtures defined here are available to all tests within
the 'tests/pipeline/' directory.
"""

import pytest
import torch
import os
import sys
from pathlib import Path

# --- FIX: Import the correct loaders ---
from sae.utils.esm_utils import load_esm2_model # From the file in your Canvas

# --- FIX: Add interplm to path and import SAE loader ---
repo_root = Path(__file__).resolve().parent.parent.parent
interplm_path = repo_root / "interplm"
if str(interplm_path) not in sys.path:
    sys.path.insert(0, str(interplm_path))

try:
    from interplm.sae.inference import load_sae_from_hf
except ImportError:
    print(f"Could not import from interplm at {interplm_path}. Tests will fail.")
    print("Please ensure the 'interplm' submodule is initialized.")
    load_sae_from_hf = None # Will cause a failure

# --- Configuration (Aligned with agentic_adapter.py) ---
SAE_HF_NAME = "esm2-8m" # The name of the SAE on Hugging Face
SAE_HF_LAYER = 6
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D" # The base ESM model
TEST_SEQUENCES_FILE = "tests/random_sequences.txt"

# --- Session-Scoped Fixtures ---

@pytest.fixture(scope="session")
def device():
    """Provides the device (CUDA or CPU) for the test session."""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="session")
def esm_model_and_tokenizer(device):
    """
    Loads the ESM model and tokenizer once per test session.
    """
    print(f"\nLoading ESM model ({ESM_MODEL_NAME}) for test session...")
    # This now correctly loads EsmForMaskedLM
    model, tokenizer = load_esm2_model(ESM_MODEL_NAME, device=device)
    return model, tokenizer

@pytest.fixture(scope="session")
def sae_model(device):
    """
    Loads the SAE model from HF via interplm once per test session.
    This follows the logic from agentic_adapter.py.
    """
    print(f"\nLoading InterPLM SAE ({SAE_HF_NAME}, layer {SAE_HF_LAYER}) for test session...")
    if load_sae_from_hf is None:
        pytest.fail("Failed to import load_sae_from_hf. Check interplm submodule.")
        
    model = load_sae_from_hf(SAE_HF_NAME, plm_layer=SAE_HF_LAYER)
    model.to(device)
    model.eval()
    return model

@pytest.fixture(scope="session")
def test_sequences():
    """
    Loads a list of sequences from the test file.
    """
    print(f"Loading test sequences from {TEST_SEQUENCES_FILE}...")
    
    # Use repo_root to build absolute path
    abs_seq_file = repo_root / TEST_SEQUENCES_FILE
    
    if not abs_seq_file.exists():
        pytest.fail(f"Test sequences file not found at {abs_seq_file}")
        
    with open(abs_seq_file, 'r') as f:
        # Read, strip whitespace, and filter out empty lines
        sequences = [line.strip() for line in f if line.strip()]
    
    if not sequences:
         pytest.fail(f"No sequences found in {abs_seq_file}")
         
    return sequences

