from __future__ import annotations
from typing import List, Tuple
import hashlib
import math
import random

# -------------------- Mock scientific tools --------------------
# These are designed to be *drop-in replaced* later with real models.
# No external dependencies are used here.

AMINO_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

def _seed_from_seq(seq: str) -> int:
    # Deterministic seed from sequence content
    h = hashlib.sha256(seq.encode()).hexdigest()
    return int(h[:16], 16)

def encode_sequence(seq: str, latent_dim: int) -> List[float]:
    """Mock encoder: pseudo-random but deterministic vector from the sequence."""
    rnd = random.Random(_seed_from_seq(seq))
    return [rnd.uniform(-1.0, 1.0) for _ in range(latent_dim)]

def perturb_latent(latent: List[float], dim: int, delta: float) -> List[float]:
    v = latent[:]  # copy
    if 0 <= dim < len(v):
        v[dim] += delta
    return v

def decode_latent(latent: List[float]) -> str:
    """Mock decoder: maps latent back to a plausible amino string.
    This is intentionally simplistic: thresholds select an amino acid index.
    """
    n = max(30, min(200, len(latent) * 3))
    # fold latent values into indices deterministically
    seq = []
    for i in range(n):
        x = latent[i % len(latent)]
        idx = int(abs(x) * 997) % len(AMINO_ALPHABET)
        seq.append(AMINO_ALPHABET[idx])
    return "".join(seq)

def score_sequence(seq: str) -> Tuple[float, float, float]:
    """Return (stability, folding, plausibility) in [0, 1]. Deterministic.
    We engineer simple signals from composition and motifs to simulate behavior.
    """
    # Composition features
    length = len(seq) or 1
    hydrophobic = set("AILMFWYV")
    charged = set("DEHKR")
    hyd_count = sum(aa in hydrophobic for aa in seq)
    charged_count = sum(aa in charged for aa in seq)
    gly_count = seq.count("G")
    pro_count = seq.count("P")

    # Toy metrics
    hyd_ratio = hyd_count / length
    charge_balance = 1.0 - abs((charged_count / length) - 0.2)  # prefer ~20% charged
    gly_pro_penalty = max(0.0, 1.0 - (gly_count + pro_count) / (0.25 * length + 1e-9))

    # Motif rewards (simulate 'function-like' structure)
    motif_bonus = 0.0
    for motif in ("AAXA", "VIL", "YWY", "STST", "RKD"):
        if motif in seq:
            motif_bonus += 0.05

    # Map to scores
    stability = max(0.0, min(1.0, 0.3 * hyd_ratio + 0.4 * charge_balance + 0.2 * gly_pro_penalty + motif_bonus))
    folding = max(0.0, min(1.0, 0.5 * hyd_ratio + 0.3 * gly_pro_penalty + 0.1 * charge_balance + motif_bonus))
    plausibility = max(0.0, min(1.0, 0.4 * charge_balance + 0.3 * (1.0 - abs(hyd_ratio - 0.45)) + 0.2 * gly_pro_penalty + motif_bonus))

    return (stability, folding, plausibility)

def sequence_similarity(a: str, b: str) -> float:
    """Simple similarity: identity over aligned length (truncate to min len)."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n
