"""
Main Proof-of-Concept (PoC) Script

This script runs the full "Encode -> Perturb -> Decode -> Score" experiment.
It uses the logic from poc_pipeline.py (which we tested) and applies it
to the real, pre-trained models.

This script will:
1. Load correlated latent candidates from the CSV.
2. Load the real SAE model.
3. Load the real ESM model.
4. Load the real surrogate scorer oracle (ridge-onehot).
5. Loop through candidates and strengths, calling run_single_perturbation().
6. Save all results to a new CSV in the 'runs/' directory.
"""

import pandas as pd
from tqdm import tqdm
import torch
import os # <-- FIX: Added import

# Import the functions we built and tested
from experiments.poc_pipeline import (
    load_perturbation_candidates, 
    run_single_perturbation
)
from scoring.surrogate_models import SurrogateScorer

# Import the *real* model loading utilities
# --- FIX: Importing the correct functions from model_utils ---
from sae.utils.model_utils import load_config, load_model
# --- FIX: Importing the real, existing functions from esm_utils ---
from sae.utils.esm_utils import get_esm_model_and_tokenizer, get_sequence_tokens, get_embeddings


# --- Configuration ---

# 1. Paths to real models and data
SAE_CHECKPOINT_PATH = "models/sae_16k_l1_1.0/checkpoint.pt"
# --- FIX: SAE Config path is also needed ---
SAE_CONFIG_PATH = "models/sae_16k_l1_1.0/config.json" # Assumed path
ORACLE_MODEL_PATH = "models/ridge_onehot_rho0.953.joblib" # Assumed path
CANDIDATE_CSV_PATH = "latent_property_correlation_mono.csv"
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# 2. Experiment parameters
BASELINE_SEQUENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE" # GB1 Wildtype
STRENGTHS_TO_TEST = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]
CANDIDATES_PER_GROUP = 5 # Test top 5 pos, top 5 neg, 5 null
OUTPUT_CSV_PATH = "runs/poc_traversal_results.csv"

# --- FIX: Define a wrapper class to match the pipeline's expectations ---
class EsmUtilsWrapper:
    """
    A wrapper class to bridge the gap between our tested pipeline
    (which expects .embed and .decode) and the *actual* repo code
    in sae/utils/esm_utils.py.
    """
    def __init__(self, esm_model, esm_tokenizer, device='cpu', layer=30):
        self.model = esm_model
        self.tokenizer = esm_tokenizer
        self.device = device
        self.layer = layer # ESM-2 t6 has 30 layers

    def embed(self, sequence: str) -> torch.Tensor:
        """
        Uses the *real* functions from esm_utils.py to get an embedding.
        """
        # 1. Get tokens
        tokens = get_sequence_tokens(self.tokenizer, sequence)
        # 2. Get embeddings
        #    Note: get_embeddings returns a list, we take the first item
        embedding = get_embeddings(
            self.model, 
            tokens, 
            self.device, 
            self.layer
        )[0]
        
        # Add a batch dimension to match what the SAE expects
        return embedding.unsqueeze(0)

    def decode(self, embedding: torch.Tensor) -> str:
        """
        CRITICAL: This function does not exist in the repo.
        ESM2 is an ENCODER-only model and cannot decode embeddings
        back into sequences.
        """
        raise NotImplementedError(
            "CRITICAL: The 'Decode' step of the pipeline is not implemented. "
            "sae/utils/esm_utils.py has no decode function, and the ESM2 "
            "model is an encoder-only architecture."
        )
# --- End of Fix ---


# --- Main Execution ---

def main():
    """
    Runs the full PoC traversal experiment.
    """
    print("Starting Proof-of-Concept Traversal Experiment...")
    
    # --- 1. Load Candidates ---
    print(f"Loading candidates from {CANDIDATE_CSV_PATH}")
    candidate_map = load_perturbation_candidates(
        CANDIDATE_CSV_PATH, 
        top_n=CANDIDATES_PER_GROUP
    )
    all_candidates = (
        candidate_map["positive"] + 
        candidate_map["negative"] + 
        candidate_map["null"]
    )
    print(f"Loaded {len(all_candidates)} total candidates.")
    
    # --- 2. Load Models ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models onto device: {device}")
    
    # --- FIX: Load SAE model using the *correct* functions ---
    print(f"Loading SAE config from {SAE_CONFIG_PATH}")
    sae_config = load_config(SAE_CONFIG_PATH)
    print(f"Loading SAE model from {SAE_CHECKPOINT_PATH}")
    sae_model = load_model(sae_config, SAE_CHECKPOINT_PATH, device=device)
    sae_model.eval() # Set to evaluation mode
    
    # Load ESM
    esm_model, esm_tokenizer = get_esm_model_and_tokenizer(ESM_MODEL_NAME, device=device)
    # --- FIX: Instantiate our new wrapper class ---
    # We assume the t6 model (30 layers) for embedding
    esm_utils_wrapper = EsmUtilsWrapper(esm_model, esm_tokenizer, device=device, layer=30)
    
    # Load Oracle Scorer
    # We assume the oracle model was trained to use the max_len of the GB1 protein
    oracle_config = {"encoding": "onehot", "max_len": len(BASELINE_SEQUENCE)}
    scorer_model = SurrogateScorer(ORACLE_MODEL_PATH, oracle_config)
    
    print("All models loaded.")
    
    # --- 3. Run Experiment Loop ---
    results = []
    
    # Create a progress bar for the experiment
    pbar = tqdm(total=len(all_candidates) * len(STRENGTHS_TO_TEST), desc="Running Traversals")
    
    for latent_index in all_candidates:
        for strength in STRENGTHS_TO_TEST:
            
            try:
                # This is the function we unit-tested
                result = run_single_perturbation(
                    baseline_sequence=BASELINE_SEQUENCE,
                    latent_index=latent_index,
                    strength=strength,
                    esm_model=esm_utils_wrapper,  # Pass the real wrapper
                    sae_model=sae_model,          # Pass the real SAE model
                    scorer_model=scorer_model     # Pass the real Scorer
                )
                
                # Add metadata to the result
                if latent_index in candidate_map["positive"]:
                    result["group"] = "positive"
                elif latent_index in candidate_map["negative"]:
                    result["group"] = "negative"
                else:
                    result["group"] = "null"
                
                results.append(result)

            except NotImplementedError as e:
                print(f"\nFATAL ERROR: {e}")
                print("Stopping experiment. The pipeline cannot continue without a decode function.")
                pbar.close()
                return # Exit main
            except Exception as e:
                print(f"\nError during traversal (idx={latent_index}, str={strength}): {e}")
            
            pbar.update(1)
            
    pbar.close()
    
    # --- 4. Save Results ---
    if not results:
        print("No results generated. Exiting.")
        return

    print(f"Experiment complete. Saving {len(results)} results to {OUTPUT_CSV_PATH}")
    results_df = pd.DataFrame(results)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    # Save to CSV
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()

