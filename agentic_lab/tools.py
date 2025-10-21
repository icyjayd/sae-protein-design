# agentic_lab/tools.py
from __future__ import annotations
from typing import List, Tuple
import sys, os
import numpy as np 
# -----------------------------------------------------------------------------
#  Path setup — ensures sibling packages like `sae/` are importable
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# -----------------------------------------------------------------------------
#  Real SAE + Scoring Imports
# -----------------------------------------------------------------------------
from agentic_adapter import RealSAE, score_sequence
from sae.utils.grade_reconstructions import grade_pair

# -----------------------------------------------------------------------------
#  SAE + Latent Operations
# -----------------------------------------------------------------------------
sae = RealSAE(model_name="esm2-8m", layer=6)

def encode_sequence(seq: str, latent_dim: int) -> List[float]:
    """Encode a protein sequence into latent space using the real SAE."""
    return sae.encode(seq, latent_dim)

def perturb_latent(latent: List[float], dim: int, delta: float) -> List[float]:
    """Apply a directional perturbation to one latent dimension."""
    v = latent[:]  # shallow copy
    if 0 <= dim < len(v):
        v[dim] += delta
    return v

def decode_latent(latent, base_seq="MKTLLILAVITAIAAGALA"):
    # identify which dimension changed
    if hasattr(decode_latent, "prev_latent"):
        diffs = latent - decode_latent.prev_latent
        dim = int(np.argmax(np.abs(diffs)))
        delta = float(diffs[dim])
    else:
        dim, delta = 0, 0.0
    decode_latent.prev_latent = latent.copy()
    return sae.perturb_and_decode(base_seq, dim=dim, delta=delta)

# -----------------------------------------------------------------------------
#  Scoring and Evaluation
# -----------------------------------------------------------------------------
def score_sequence_wrapper(seq: str) -> Tuple[float, float, float]:
    """Compute (stability, folding, plausibility) for a sequence."""
    return score_sequence(seq)

def sequence_similarity(seq_a: str, seq_b: str) -> float:
    """
    Compute overall alignment similarity between two sequences.
    Uses grade_reconstructions.grade_pair(), which combines:
      - identity %
      - BLOSUM62 similarity
      - normalized alignment
      - Levenshtein similarity
      - weighted composite final score
    """
    result = grade_pair(seq_a, seq_b)
    return result.get("final_score", 0.0)

# -----------------------------------------------------------------------------
#  Optional convenience functions
# -----------------------------------------------------------------------------
def compare_sequences(seq_a: str, seq_b: str) -> dict:
    """
    Return the full comparison dictionary from grade_pair(),
    including identity, similarity, norm_align, lev_sim, final_score.
    """
    return grade_pair(seq_a, seq_b)
