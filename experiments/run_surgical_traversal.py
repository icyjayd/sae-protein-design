"""
Surgical vs. Global Perturbation Traversal

This experiment tests the core hypothesis:
"Is perturbing a latent feature *only* at its most active
positions (surgical) more effective than perturbing it
everywhere (global)?"

This script will:
1. Load correlated latent candidates (e.g., #4092).
2. For each candidate:
    a. Find its "hotspot" positions in the baseline sequence.
    b. Run 3 tests:
        i.   Baseline (no perturbation)
        ii.  Global (perturb all positions)
        iii. Surgical (perturb only hotspot positions)
    c. Score all 3 results and save them.
"""

import pandas as pd
from tqdm import tqdm
import torch
import os 
import sys
from pathlib import Path
import numpy as np

# --- FIX: Add project root to sys.path ---
# This ensures that we can find the 'experiments', 'scoring', and 'sae' modules
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
# --- End Fix ---

# --- FIX: Use absolute import from project root ---
from experiments.pipeline_utils import (
    load_perturb_candidates,
    find_hotspots
)
# --- End Fix ---

from scoring.surrogate_models import SurrogateScorer
from sae.utils.esm_utils import (
    load_esm2_model, 
    perturb_and_decode,
    get_activation_matrix  # <-- NEW
)

# --- Add interplm to path and import SAE loader ---
# repo_root is already defined above
interplm_path = repo_root / "interplm"
if str(interplm_path) not in sys.path:
    sys.path.insert(0, str(interplm_path))

try:
    from interplm.sae.inference import load_sae_from_hf
except ImportError:
    print(f"Could not import from interplm at {interplm_path}.")
    print("Please ensure the 'interplm' submodule is initialized.")
    sys.exit(1)

# --- Configuration ---
SAE_HF_NAME = "esm2-8m"
SAE_HF_LAYER = 6
ORACLE_MODEL_PATH = r"C:\Users\juani\Documents\Repos\proteng\model_scout_outs\models\ridge_onehot_10000.joblib"
CANDIDATE_CSV_PATH = "outputs/latent_property_correlation_mono.csv"
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

BASELINE_SEQUENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE" # GB1 Wildtype
STRENGTHS_TO_TEST = [-15.0, -10.0, -5.0, 5.0, 10.0, 15.0]
CANDIDATES_PER_GROUP = 5 # Test top 5 positive, top 5 negative
TOP_K_HOTSPOTS = 5       # Define "surgical" as top 5 positions
OUTPUT_CSV_PATH = "runs/surgical_vs_global_results.csv"

# --- FIX: Define the max_len the model was trained with ---
SCORER_MODEL_MAX_LEN = 512

# --- Main Execution ---

def main():
    """
    Runs the full Surgical vs. Global experiment.
    """
    print("Starting Surgical vs. Global Traversal Experiment...")
    
    # --- 1. Load Candidates ---
    print(f"Loading candidates from {CANDIDATE_CSV_PATH}")
    candidate_map = load_perturb_candidates( 
        CANDIDATE_CSV_PATH, 
        top_n=CANDIDATES_PER_GROUP
    )
    # We only test the positive/negative candidates for this experiment
    all_candidates = candidate_map["positive"] + candidate_map["negative"]
    all_candidates = [int(c) for c in all_candidates]
    print(f"Loaded {len(all_candidates)} candidates to test.")
    
    # --- 2. Load Models ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models onto device: {device}")
    
    sae_model = load_sae_from_hf(SAE_HF_NAME, plm_layer=SAE_HF_LAYER)
    sae_model.to(device).eval()
    
    esm_model, esm_tokenizer = load_esm2_model(ESM_MODEL_NAME, device=device)
    
    # --- FIX: Use the correct SCORER_MODEL_MAX_LEN ---
    scorer_config = {"encoding": "onehot", "max_len": SCORER_MODEL_MAX_LEN}
    scorer_model = SurrogateScorer(ORACLE_MODEL_PATH, scorer_config)
    
    print("All models loaded.")

    # --- 3. Get Baseline Score & Activation Matrix ---
    print("Calculating baseline score and activation matrix...")
    
    # Get Baseline Score (no perturbation)
    baseline_seq = perturb_and_decode(
        BASELINE_SEQUENCE, sae_model, esm_model, esm_tokenizer, 
        surgical_perturbations=None, device=device
    )
    print("Scorer model score:", scorer_model.score)
    baseline_score = scorer_model.score([baseline_seq])[0]
    print(f"Baseline (reconstructed) score: {baseline_score:.4f}")

    # Get Activation Matrix
    activation_matrix = get_activation_matrix(
        BASELINE_SEQUENCE, sae_model, esm_model, esm_tokenizer, device=device
    )
    print(f"Activation matrix shape: {activation_matrix.shape}") # (L-2, 10240)

    # --- 4. Run Experiment Loop ---
    results = []
    
    # Add the baseline result
    results.append({
        "latent_index": -1, "strength": 0.0, "test_type": "baseline",
        "new_sequence": baseline_seq, "new_score": baseline_score,
        "delta_score": 0.0, "hotspots": "[]"
    })

    pbar = tqdm(total=len(all_candidates) * len(STRENGTHS_TO_TEST), desc="Running Traversals")
    
    seq_len = len(BASELINE_SEQUENCE)
    all_positions = list(range(1, seq_len + 1)) # Token indices are 1 to L-2 (== 1 to seq_len)

    for latent_index in all_candidates:
        
        # Find hotspots *once* per latent
        hotspot_positions = find_hotspots(activation_matrix, latent_index, top_k=TOP_K_HOTSPOTS)
        hotspots_str = str(hotspot_positions)
        
        for strength in STRENGTHS_TO_TEST:
            
            # --- FIX: Build the surgical_perturbation dictionaries ---
            
            # 1. Global: { 1: {latent: delta}, 2: {latent: delta}, ... }
            pert_global = {
                pos: {latent_index: strength} for pos in all_positions
            }
            
            # 2. Surgical: { 12: {latent: delta}, 34: {latent: delta}, ... }
            pert_surgical = {
                pos: {latent_index: strength} for pos in hotspot_positions
            }
            
            # --- Test A: Global Perturbation ---
            seq_global = perturb_and_decode(
                BASELINE_SEQUENCE, sae_model, esm_model, esm_tokenizer,
                surgical_perturbations=pert_global, # <-- Pass global map
                device=device
            )
            score_global = scorer_model.score([seq_global])[0]
            
            results.append({
                "latent_index": latent_index, "strength": strength, "test_type": "global",
                "new_sequence": seq_global, "new_score": score_global,
                "delta_score": score_global - baseline_score, "hotspots": hotspots_str
            })

            # --- Test B: Surgical Perturbation ---
            seq_surgical = perturb_and_decode(
                BASELINE_SEQUENCE, sae_model, esm_model, esm_tokenizer,
                surgical_perturbations=pert_surgical, # <-- Pass surgical map
                device=device
            )
            score_surgical = scorer_model.score([seq_surgical])[0]
            
            results.append({
                "latent_index": latent_index, "strength": strength, "test_type": "surgical",
                "new_sequence": seq_surgical, "new_score": score_surgical,
                "delta_score": score_surgical - baseline_score, "hotspots": hotspots_str
            })
            
            pbar.update(1)
            
    pbar.close()
    
    # --- 5. Save Results ---
    if not results:
        print("No results generated. Exiting.")
        return

    print(f"Experiment complete. Saving {len(results)} results to {OUTPUT_CSV_PATH}")
    results_df = pd.DataFrame(results)
    
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()

