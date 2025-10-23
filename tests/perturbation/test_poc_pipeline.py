"""
Tests the end-to-end logic of the "Encode -> Perturb -> Decode -> Score"
pipeline defined in experiments/poc_pipeline.py
"""

import numpy as np
from experiments.poc_pipeline import run_single_perturbation
from scoring.surrogate_models import SurrogateScorer # <-- 1. Add import

def test_pipeline_integration_logic(
    mock_esm_model, 
    mock_sae_model, 
    mock_surrogate_model, 
    mocker  # <-- 2. Add mocker
):
    """
    Tests that all mock components are called in the correct order
    and that the latent vector is perturbed correctly.
    """
    
    # --- 3. FIX: Load the scorer correctly ---
    # The fixture now *only* returns the path
    model_path = mock_surrogate_model 
    
    # Define the config our mock model expects
    config = {"encoding": "onehot", "max_len": 100}
    
    # Load the real scorer class, which will load our pickle-able mock model
    mock_scorer = SurrogateScorer(model_path, config)
    
    # We can spy on the *model inside* the scorer to be 100% sure
    mocker.spy(mock_scorer.model, "predict")
    # --- End of Fix ---

    # --- Test Parameters ---
    baseline_sequence = "WILDTYPESEQ"
    target_index = 42
    strength = 5.0
    
    # Get the original latent vector to compare against
    original_latent = mock_sae_model.encoder(
        mock_esm_model.embed(baseline_sequence)
    )

    # --- Run the Pipeline ---
    result = run_single_perturbation(
        baseline_sequence=baseline_sequence,
        latent_index=target_index,
        strength=strength,
        esm_model=mock_esm_model,
        sae_model=mock_sae_model,
        scorer_model=mock_scorer # Pass the real scorer instance
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
    # Check that our spied-upon .predict method was called
    mock_scorer.model.predict.assert_called_once()
    
    # 5. Assert final result dictionary is correct
    # --- 4. FIX: Check for the *real* mock score ---
    # Our SimpleMockModel in conftest.py always returns 1.23
    assert result == {
        "latent_index": target_index,
        "strength": strength,
        "new_sequence": "PERTURBEDSEQ",
        "new_score": 1.23 
    }

