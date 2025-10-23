"""
New file to hold the core Proof-of-Concept (PoC) pipeline logic.
This will be tested by tests/perturbation/
"""

import pandas as pd
import numpy as np

def load_perturbation_candidates(
    csv_path: str, 
    top_n: int = 5, 
    null_threshold: float = 0.05
) -> dict:
    """
    Loads the correlation CSV and selects the best candidates 
    for the codirectionality test.

    Args:
        csv_path (str): Path to the latent_property_correlation_mono.csv file.
        top_n (int): Number of positive/negative/null candidates to select.
        null_threshold (float): Absolute spearman's rho to be considered "null".

    Returns:
        dict: A dictionary with 'positive', 'negative', and 'null' candidate indices.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Warning: Candidate CSV not found at {csv_path}. Returning empty dict.")
        return {"positive": [], "negative": [], "null": []}
    
    # Filter out invalid (NaN) latents, as you noted
    df = df.dropna(subset=['spearman', 'p_value'])
    
    # Ensure latent_index is integer
    df['latent_index'] = df['latent_index'].astype(int)
    
    # Sort by correlation strength
    df = df.sort_values(by='spearman', ascending=False)
    
    # 1. Select Top Positive Candidates (Hypothesis Test)
    positive_candidates = df.head(top_n)['latent_index'].tolist()
    
    # 2. Select Top Negative Candidates (Hypothesis Test)
    negative_candidates = df.tail(top_n)['latent_index'].tolist()
    
    # 3. Select Null Candidates (Control Group)
    # Find latents with near-zero correlation
    null_df = df[
        (df['spearman'].abs() < null_threshold)
    ]
    
    # Ensure we don't pick from the positive/negative lists
    null_df = null_df[
        ~null_df['latent_index'].isin(positive_candidates + negative_candidates)
    ]
    
    null_candidates = null_df.head(top_n)['latent_index'].tolist()
    
    return {
        "positive": positive_candidates,
        "negative": negative_candidates,
        "null": null_candidates
    }


def run_single_perturbation(
    baseline_sequence: str,
    latent_index: int,
    strength: float,
    esm_model,  # Mocked PLM (e.g., from sae/utils/esm_utils.py)
    sae_model,  # Mocked SAE (e.g., from sae/utils/model_utils.py)
    scorer_model # Mocked SurrogateScorer
) -> dict:
    """
    Runs the full "Encode -> Perturb -> Decode -> Score" pipeline
    for a single data point.
    
    This function is designed to be testable with mock objects.
    """
    
    # 1. ENCODE
    # Get baseline embedding from PLM
    baseline_embedding = esm_model.embed(baseline_sequence)
    
    # Get baseline latent code from SAE encoder
    baseline_latent = sae_model.encoder(baseline_embedding)
    
    # 2. PERTURB
    # Create a new latent vector by perturbing the target index
    # --- FIX: Use np.copy() instead of .clone() ---
    new_latent = np.copy(baseline_latent) 
    new_latent[0, latent_index] += strength # Assuming batch dim of 1
    
    # 3. DECODE
    # Get new embedding from SAE decoder
    perturbed_embedding = sae_model.decoder(new_latent)
    
    # Get new sequence from PLM decoder
    new_sequence = esm_model.decode(perturbed_embedding)
    
    # 4. SCORE
    # Score the new sequence with the independent oracle
    score = scorer_model.score([new_sequence])[0] # score returns an array
    
    return {
        "latent_index": latent_index,
        "strength": strength,
        "new_sequence": new_sequence,
        "new_score": score
    }

