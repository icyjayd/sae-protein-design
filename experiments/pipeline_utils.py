"""
Utilities for running pipeline experiments.

Includes:
- Loading candidates from the correlation CSV.
- Finding "hotspot" positions from an activation matrix.
"""

import pandas as pd
from typing import List, Dict
import torch

def load_perturb_candidates(
    csv_path: str, 
    top_n: int = 5, 
    null_threshold: float = 0.05
) -> Dict[str, List[int]]:
    """
    Loads the correlation CSV and selects the best candidates 
    for the codirectionality test.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Warning: Candidate CSV not found at {csv_path}. Returning empty dict.")
        return {"positive": [], "negative": [], "null": []}
    
    # Filter out invalid (NaN) latents
    df = df.dropna(subset=['spearman', 'p_value'])
    
    # Ensure latent_index is int
    df['latent_index'] = df['latent_index'].astype(int)
    
    # Sort by correlation strength
    df = df.sort_values(by='spearman', ascending=False)
    
    # 1. Select Top Positive Candidates (Hypothesis Test)
    positive_candidates = df.head(top_n)['latent_index'].tolist()
    
    # 2. Select Top Negative Candidates (Hypothesis Test)
    negative_candidates = df.tail(top_n)['latent_index'].tolist()
    
    # 3. Select Null Candidates (Control Group)
    null_df = df[
        (df['spearman'].abs() < null_threshold) & 
        (df['p_value'] > 0.1)
    ]
    null_candidates = null_df.head(top_n)['latent_index'].tolist()
    
    return {
        "positive": positive_candidates,
        "negative": negative_candidates,
        "null": null_candidates
    }

def find_hotspots(
    activation_matrix: torch.Tensor,
    latent_index: int,
    top_k: int = 5
) -> List[int]:
    """
    Finds the top_k positions (token indices) with the highest
    activation for a given latent feature.
    
    Args:
        activation_matrix: (L-2, D_SAE) tensor from get_activation_matrix.
        latent_index: The latent feature we care about.
        top_k: The number of hotspot positions to return.
        
    Returns:
        A list of token indices (e.g., [3, 12, 50]) corresponding
        to the *1-based* indices in the original token tensor.
    """
    if latent_index >= activation_matrix.shape[1]:
        print(f"Warning: latent_index {latent_index} is out of bounds.")
        return []
        
    # 1. Get the activation vector for this one latent
    # shape: (L-2,)
    latent_activations = activation_matrix[:, latent_index]
    
    # 2. Find the top_k values and their indices
    # We ask for min(top_k, num_tokens) in case sequence is short
    k = min(top_k, len(latent_activations))
    _, top_indices = torch.topk(latent_activations, k) # shape: (k,)
    
    # 3. Convert to a sorted list of Python ints
    top_indices_list = sorted([idx.item() for idx in top_indices])
    
    # 4. CRITICAL: Adjust indices.
    # The activation matrix is 0-indexed and maps to tokens [1:-1].
    # An index 'i' in the matrix (0 to L-3) corresponds to
    # token 'i+1' in the original sequence (1 to L-2).
    # The perturb_and_decode function's loop also starts at 1.
    surgical_positions = [i + 1 for i in top_indices_list]
    
    return surgical_positions

