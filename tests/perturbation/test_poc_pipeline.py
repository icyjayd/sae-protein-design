"""
Tests the end-to-end logic of the "Encode -> Perturb -> Decode -> Score"
pipeline defined in experiments/poc_pipeline.py
"""

import numpy as np
import pytest
from experiments.poc_pipeline import run_single_perturbation
from scoring.surrogate_models import SurrogateScorer

@pytest.mark.parametrize("strength, expected_score", [
    (5.0, 1.23),  # Test positive perturbation
    (-5.0, -0.5)  # Test negative perturbation
])
def test_pipeline_integration_logic(
    mock_esm_model, 
    mock_sae_model, 
    mock_surrogate_model, 
    mocker,
    strength,         # Parameterized strength
    expected_score    # Parameterized expected score
):
    """
    Tests that all mock components are called in the correct order
    and that the latent vector is perturbed correctly.
    
    This test is parameterized to run with both positive and
    negative perturbation strengths.
    """
    
    # --- 1. Load the scorer correctly ---
    model_path = mock_surrogate_model 
    config = {"encoding": "onehot", "max_len": 100}
    # We still need a real scorer instance, even though we will mock its method
    mock_scorer = SurrogateScorer(model_path, config)
    
    
    # --- 2. FIX: Mock the .score() method directly ---
    # This test is for the *pipeline*, not the scorer's implementation.
    # We patch the 'score' method to return the parameterized score.
    mock_score_method = mocker.patch(
        "scoring.surrogate_models.SurrogateScorer.score",
        return_value=np.array([expected_score]) # Return the parameterized score
    )
    # --- End of Fix ---

    # --- Test Parameters ---
    baseline_sequence = "WILDTYPESEQ"
    target_index = 42
    # The 'strength' parameter is now injected by pytest
    
    # Get the original latent vector to compare against
    original_latent = mock_sae_model.encoder(
        mock_esm_model.embed(baseline_sequence)
    )

    # --- Run the Pipeline ---
    result = run_single_perturbation(
        baseline_sequence=baseline_sequence,
        latent_index=target_index,
        strength=strength,      # Use parameterized strength
        esm_model=mock_esm_model,
        sae_model=mock_sae_model,
        scorer_model=mock_scorer # Pass the scorer instance
    )

    # --- Assertions ---
    
    # 1. Assert ENCODE step
    mock_esm_model.embed.assert_called_with(baseline_sequence)
    mock_sae_model.encoder.assert_called_with(
        mock_esm_model.embed.return_value
    )
    
    # 2. Assert PERTURB step
    # Get the latent vector that was passed to the decoder
    perturbed_latent = mock_sae_model.decoder.call_args[0][0]
    
    # Check that the perturbation was applied correctly
    expected_latent = np.copy(original_latent) # Use np.copy
    expected_latent[0, target_index] += strength
    
    np.testing.assert_array_equal(perturbed_latent, expected_latent)
    
    # 3. Assert DECODE step
    mock_sae_model.decoder.assert_called_with(perturbed_latent)
    mock_esm_model.decode.assert_called_with(
        mock_sae_model.decoder.return_value
    )
    
    # 4. Assert SCORE step
    # Check that our mocked .score() method was called
    # The mock ESM decoder in conftest.py returns "PERTURBEDSEQ"
    mock_score_method.assert_called_once_with(["PERTURBEDSEQ"])
    
    # 5. Assert final result dictionary is correct
    assert result == {
        "latent_index": target_index,
        "strength": strength,         # Assert parameterized strength
        "new_sequence": "PERTURBEDSEQ",
        "new_score": expected_score # Assert parameterized score
    }

