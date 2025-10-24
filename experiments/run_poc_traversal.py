"""
Main Proof-of-Concept (PoC) Script

This script runs the full "Encode -> Perturb -> Decode -> Score" experiment.
It now uses the *per-token* perturbation logic.

This script will:
1. Load correlated latent candidates from the CSV.
2. Load the real SAE model.
3. Load the real ESM model (EsmForMaskedLM).
4. Load the real surrogate scorer oracle (ridge-onehot).
5. Loop through candidates/strengths, calling the new stateless perturb_and_decode.
6. Save all results to a new CSV in the 'runs/' directory.
"""

import pandas as pd
from tqdm import tqdm
import torch
import os 

# --- FIX: Import only candidate loader from pipeline_utils ---
# We no longer need pipeline_utils, we can use the function from esm_utils
from experiments.pipeline_utils import load_perturb_candidates
from scoring.surrogate_models import SurrogateScorer

# Import the *real* model loading utilities
from sae.utils.model_utils import load_config, load_model
# --- FIX: Import our new stateless function ---
from sae.utils.esm_utils import (
    load_esm2_model, 
    perturb_and_decode # --- FIX: Renamed function
)


# --- Configuration ---

# 1. Paths to real models and data
# --- FIX: Using interplm, we don't need local paths ---
# SAE_CHECKPOINT_PATH = "models/sae_16k_l1_1.0/checkpoint.pt"
# SAE_CONFIG_PATH = "models/sae_16k_l1_1.0/config.json" 
SAE_HF_NAME = "esm2-8m" # The name of the SAE on Hugging Face
SAE_HF_LAYER = 6
ORACLE_MODEL_PATH = "models/ridge_onehot_rho0.953.joblib" # Assumed path
CANDIDATE_CSV_PATH = "latent_property_correlation_mono.csv"
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# 2. Experiment parameters
BASELINE_SEQUENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE" # GB1 Wildtype
STRENGTHS_TO_TEST = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]
CANDIDATES_PER_GROUP = 5 # Test top 5 pos, top 5 neg, 5 null
OUTPUT_CSV_PATH = "runs/poc_traversal_results_per_token.csv" # Renamed output


# --- FIX: Add interplm to path and import SAE loader ---
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
interplm_path = repo_root / "interplm"
if str(interplm_path) not in sys.path:
    sys.path.insert(0, str(interplm_path))

try:
    from interplm.sae.inference import load_sae_from_hf
except ImportError:
    print(f"Could not import from interplm at {interplm_path}.")
    print("Please ensure the 'interplm' submodule is initialized.")
    sys.exit(1) # Exit if we can't load the SAE

# --- Main Execution ---

def main():
    """
    Runs the full PoC traversal experiment.
    """
    print("Starting Proof-of-Concept Traversal Experiment (Per-Token)...")
    
    # --- 1. Load Candidates ---
    print(f"Loading candidates from {CANDIDATE_CSV_PATH}")
    candidate_map = load_perturb_candidates( 
        CANDIDATE_CSV_PATH, 
        top_n=CANDIDATES_PER_GROUP
    )
    all_candidates = (
        candidate_map["positive"] + 
        candidate_map["negative"] + 
        candidate_map["null"]
    )
    # --- FIX: Ensure candidates are integers ---
    all_candidates = [int(c) for c in all_candidates]
    print(f"Loaded {len(all_candidates)} total candidates.")
    
    # --- 2. Load Models ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models onto device: {device}")
    
    # Load SAE model (from interplm)
    print(f"Loading InterPLM SAE ({SAE_HF_NAME}, layer {SAE_HF_LAYER})...")
    sae_model = load_sae_from_hf(SAE_HF_NAME, plm_layer=SAE_HF_LAYER)
    sae_model.to(device)
    sae_model.eval()
    
    # Load ESM (EsmForMaskedLM)
    print(f"Loading ESM model ({ESM_MODEL_NAME})...")
    esm_model, esm_tokenizer = load_esm2_model(ESM_MODEL_NAME, device=device)
    
    # Load Oracle Scorer
    print(f"Loading Oracle Scorer from {ORACLE_MODEL_PATH}...")
    oracle_config = {"encoding": "onehot", "max_len": len(BASELINE_SEQUENCE)}
    scorer_model = SurrogateScorer(ORACLE_MODEL_PATH, oracle_config)
    
    print("All models loaded.")
    
    # --- 3. Run Experiment Loop ---
    results = []
    
    pbar = tqdm(total=len(all_candidates) * len(STRENGTHS_TO_TEST), desc="Running Traversals")
    
    for latent_index in all_candidates:
        for strength in STRENGTHS_TO_TEST:
            
            try:
                # --- FIX: Call our new stateless function directly ---
                
                # 1. ENCODE -> PERTURB -> DECODE (Per-Token)
                new_sequence = perturb_and_decode( # --- FIX: Renamed function
                    sequence=BASELINE_SEQUENCE,
                    dim=latent_index,
                    delta=strength,
                    esm_model=esm_model,
                    tokenizer=esm_tokenizer,
                    sae_model=sae_model,
                    device=device
                )
                
                # 2. SCORE
                score = scorer_model.score([new_sequence])[0] # Get single score
                
                # 3. RECORD
                result = {
                    "latent_index": latent_index,
                    "strength": strength,
                    "new_sequence": new_sequence,
                    "new_score": score
                }
                
                # Add metadata to the result
                if latent_index in candidate_map["positive"]:
                    result["group"] = "positive"
                elif latent_index in candidate_map["negative"]:
                    result["group"] = "negative"
                else:
                    result["group"] = "null"
                
                results.append(result)

            except Exception as e:
                print(f"\nError during traversal (idx={latent_index}, str={strength}): {e}")
                # Optionally, re-raise to stop: raise e
            
            pbar.update(1)
            
    pbar.close()
    
    # --- 4. Save Results ---
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

