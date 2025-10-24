# scoring/surrogate_models.py
from __future__ import annotations
from typing import Dict
import math

# --- New Imports for Oracle Scorer ---\
import joblib
import numpy as np

# --- FIX: Removed the try/except block ---
# This forces a clean import. If this fails, we have a PYTHONPATH
# issue, but pytest should handle this by adding the root dir.
from ml_models.encoding import encode_sequences


AA = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWYV")
CHARGED_POS = set("KRH")
CHARGED_NEG = set("DE")
AROMATIC = set("FWY")

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _fraction(seq: str, bag: set[str]) -> float:
    n = len(seq)
    if n == 0:
        return 0.0
    return sum(c in bag for c in seq) / n

def _max_run(seq: str) -> int:
    if not seq:
        return 0
    max_run = 0
    current_run = 0
    for char in seq:
        if char in AROMATIC:
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 0
    max_run = max(max_run, current_run)
    return max_run

def heuristic_surrogate_score(seq: str) -> float:
    """
    A simple, interpretable surrogate model for a hypothetical
    protein property that balances stability (hydrophobicity)
    and solubility (charge), while penalizing aggregation (aromatic runs).
    """
    if not seq or not all(c in AA for c in seq):
        return 0.0  # Invalid sequence

    # 1. Stability (Hydrophobicity)
    # Target: 30-40% hydrophobic residues
    f_hydro = _fraction(seq, HYDROPHOBIC)
    # Score is 1.0 at 35%, 0.0 at 15% and 55%
    stability_score = _clamp(1.0 - abs(f_hydro - 0.35) / 0.2)

    # 2. Solubility (Net Charge)
    # Target: More charged residues, but not extremely skewed
    f_pos = _fraction(seq, CHARGED_POS)
    f_neg = _fraction(seq, CHARGED_NEG)
    f_charged = f_pos + f_neg
    net_charge = abs(f_pos - f_neg)
    # Score is 1.0 at 25% charged, 0.0 at 0%
    charge_amount_score = _clamp(f_charged / 0.25)
    # Score is 1.0 at 0 net charge, 0.0 at 20% net charge
    charge_balance_score = _clamp(1.0 - net_charge / 0.2)
    solubility_score = (charge_amount_score + charge_balance_score) / 2.0

    # 3. Aggregation Penalty (Aromatic Runs)
    # Penalize runs of aromatic residues (e.g., FFF, WYW)
    max_aromatic_run = _max_run(seq)
    # No penalty for <= 2, max penalty at 5
    aggregation_penalty = _clamp((max_aromatic_run - 2) / 3.0)
    
    # Final Score
    # Stability and solubility are key. Aggregation is a penalty.
    base_score = 0.6 * stability_score + 0.4 * solubility_score
    final_score = base_score * (1.0 - 0.5 * aggregation_penalty) # 50% max penalty
    
    # Scale to a 0-10 range for more dynamic output
    return round(final_score * 10.0, 4)


class SurrogateScorer:
    """
    Loads and runs a pre-trained, data-driven sequence oracle
    (e.g., the ridge on onehot features from results.csv).
    
    This is distinct from the heuristic `heuristic_surrogate_score` function above.
    """
    
    def __init__(self, model_path: str, encoding_config: dict):
        """
        Loads the pre-trained model.

        Args:
            model_path (str): Path to the .joblib model file.
            encoding_config (dict): Params for the encoder, 
                                    e.g., {"encoding": "onehot", "max_len": 512}
        """
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Warning: SurrogateScorer model file not found at {model_path}. Using None.")
            self.model = None # Allows for testing without a real model
        self.config = encoding_config

    def score(self, sequences: list[str]) -> np.ndarray:
        """
        Encodes and scores one or more sequences.

        Args:
            sequences (list[str]): A list of protein sequences.

        Returns:
            np.ndarray: A numpy array of predicted scores.
        """
        if self.model is None:
            raise RuntimeError("SurrogateScorer model is not loaded.")
            
        # Call the encoder
        X = encode_sequences(sequences, **self.config)
        
        # Predict scores
        scores = self.model.predict(X)
        return scores

