"""
Integration test for the full per-token reconstruction pipeline.

This test verifies that the `perturb_and_decode` function
can reconstruct sequences with high fidelity when no perturbation is applied.
"""

import pytest
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

# --- Import function to be tested ---
from sae.utils.esm_utils import perturb_and_decode

# --- Import grading utility ---
from sae.utils.grade_reconstructions import mean_grade, grade_pair

# --- Test Constants ---
RECONSTRUCTION_SIMILARITY_THRESHOLD = 0.95 
NUM_SEQUENCES_TO_TEST = 20 
STRONG_PERTURBATION_DELTA = 50.0
# We expect a strong perturbation to make the sequence < 50% similar
PERTURBATION_SIMILARITY_THRESHOLD = 0.5 

def test_reconstruction_similarity(
    esm_model_and_tokenizer,  # Fixture from tests/pipeline/conftest.py
    sae_model,                # Fixture from tests/pipeline/conftest.py
    test_sequences            # Fixture from tests/pipeline/conftest.py
):
    """
    Tests that the end-to-end pipeline (ESM-encode -> SAE-encode ->
    SAE-decode -> ESM-decode) can reconstruct sequences with high similarity
    when no perturbation (delta=0.0) is applied.
    """
    esm_model, tokenizer = esm_model_and_tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nTesting reconstruction on {NUM_SEQUENCES_TO_TEST} sequences...")
    
    pairs = [] 
    all_metrics = [] 
    
    sequences_to_test = test_sequences[:NUM_SEQUENCES_TO_TEST]

    for seq in tqdm(sequences_to_test, desc="Grading Reconstructions"):
        
        # --- Call with no perturbations ---
        recon_seq = perturb_and_decode(
            sequence=seq,
            perturbations=None, # <-- No perturbations
            esm_model=esm_model,
            tokenizer=tokenizer,
            sae_model=sae_model,
            device=device
        )
        
        pairs.append((seq, recon_seq))
        metrics = grade_pair(seq, recon_seq)
        all_metrics.append(metrics)

    # --- Grade the results ---
    print("\nCalculating mean similarity scores...")
    _ = mean_grade(pairs, csv_path="runs/test_reports/reconstruction_report.csv")
    
    metrics_df = pd.DataFrame(all_metrics)
    mean_similarity = metrics_df['similarity'].mean()
    mean_identity = metrics_df['identity'].mean()

    print(f"Mean Reconstruction Similarity: {mean_similarity * 100:.2f}%")
    print(f"Mean Reconstruction Identity: {mean_identity * 100:.2f}%")

    assert mean_similarity >= RECONSTRUCTION_SIMILARITY_THRESHOLD, \
        f"Mean similarity {mean_similarity:.4f} is below the {RECONSTRUCTION_SIMILARITY_THRESHOLD} threshold."


# --- UPDATED TEST ---
def test_strong_perturbation_changes_sequence(
    esm_model_and_tokenizer,  # Fixture from tests/pipeline/conftest.py
    sae_model,                # Fixture from tests/pipeline/conftest.py
    test_sequences            # Fixture from tests/pipeline/conftest.py
):
    """
    Tests that applying a very strong perturbation to *all* latent
    dimensions results in a sequence that is *significantly different*
    from the original.
    """
    esm_model, tokenizer = esm_model_and_tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seq = test_sequences[0]
    print(f"\nTesting strong perturbation on sequence: {seq[:30]}...")

    # --- Create a perturbation dict for *all* latents ---
    # num_latents = sae_model.d_sae
    num_latents = 10240
    perturbations = {i: STRONG_PERTURBATION_DELTA for i in range(num_latents)}
    
    print(f"Applying delta={STRONG_PERTURBATION_DELTA} to all {num_latents} latents...")

    # --- Run the function with a STRONG perturbation ---
    recon_seq_perturbed = perturb_and_decode(
        sequence=seq,
        perturbations=perturbations, # <-- Pass the full dict
        esm_model=esm_model,
        tokenizer=tokenizer,
        sae_model=sae_model,
        device=device
    )

    # --- Grade the result ---
    metrics = grade_pair(seq, recon_seq_perturbed)
    similarity = metrics['similarity']

    print(f"Original Sequence (len {len(seq)}):   {seq[:60]}...")
    print(f"Perturbed Sequence (len {len(recon_seq_perturbed)}): {recon_seq_perturbed[:60]}...")
    print(f"Resulting Similarity: {similarity * 100:.2f}%")

    # --- Assert that the similarity is LOW ---
    assert similarity < PERTURBATION_SIMILARITY_THRESHOLD, \
        f"Similarity {similarity:.4f} is *above* the {PERTURBATION_SIMILARITY_THRESHOLD} threshold. " \
        "The strong perturbation did not significantly change the sequence."

