from __future__ import annotations
from typing import Dict

AA = set("ACDEFGHIKLMNPQRSTVWY")

HELIX_PROP = {
    # Rough helix propensities
    "A": 1.45, "L": 1.34, "M": 1.20, "Q": 1.17, "E": 1.51,
    "K": 1.23, "R": 0.79, "H": 1.0,  "I": 1.0,  "V": 1.06,
    "F": 1.12, "Y": 0.61, "W": 1.0,  "C": 0.79, "T": 0.82,
    "S": 0.79, "N": 0.73, "D": 0.98, "G": 0.53, "P": 0.34,
}

BETA_PROP = {
    # Rough beta-strand propensities
    "V": 1.70, "I": 1.60, "Y": 1.47, "F": 1.38, "W": 1.19,
    "T": 1.20, "C": 1.30, "L": 1.22, "M": 1.05, "A": 0.97,
    "H": 1.00, "R": 0.93, "Q": 1.10, "E": 0.26, "K": 0.74,
    "D": 0.80, "N": 0.65, "S": 0.72, "G": 0.81, "P": 0.31,
}

POS = set("KRH")
NEG = set("DE")

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _frac(seq: str, bag: set[str]) -> float:
    if not seq:
        return 0.0
    return sum(c in bag for c in seq)/len(seq)

def _mean_prop(seq: str, table: Dict[str, float]) -> float:
    if not seq:
        return 0.0
    vals = [table.get(c, 1.0) for c in seq]
    return sum(vals) / len(vals)

def predict_stability(seq: str) -> float:
    """
    Stability heuristic in [0,1].
    Combines helix/beta propensities, disulfide potential, and charge moderation.
    """
    if not seq:
        return 0.0

    # Composition sanity
    valid_frac = sum(c in AA for c in seq) / len(seq)
    if valid_frac < 1.0:
        return _clamp(0.4 * valid_frac)

    # Secondary structure propensities
    helix = _mean_prop(seq, HELIX_PROP) / 1.6  # normalize to ~[0.2,1]
    beta = _mean_prop(seq, BETA_PROP) / 1.7

    # Balance of charges (avoid extreme net charge)
    pos = _frac(seq, POS)
    neg = _frac(seq, NEG)
    charge_total = pos + neg
    charge_bal = 1.0 - abs(pos - neg)  # symmetric charges are better
    charge_level = 1.0 - abs(charge_total - 0.18)  # moderate overall charge

    # Disulfide potential: reward if enough Cys and reasonable length
    has_ss = (seq.count("C") >= 2 and len(seq) >= 60)
    ss_bonus = 0.08 if has_ss else 0.0

    L = len(seq)
    length_ok = _clamp((L - 30) / 120.0)  # harsher than plausibility

    score = (
        0.28 * _clamp(helix) +
        0.22 * _clamp(beta) +
        0.16 * _clamp(charge_bal) +
        0.14 * _clamp(charge_level) +
        0.12 * length_ok +
        ss_bonus
    )
    return _clamp(score)

def predict_folding(seq: str) -> float:
    """
    Folding confidence proxy in [0,1].
    Encourages mixed composition, adequate length, and moderate complexity.
    """
    if not seq:
        return 0.0

    valid_frac = sum(c in AA for c in seq) / len(seq)
    if valid_frac < 1.0:
        return _clamp(0.4 * valid_frac)

    hyd = _frac(seq, set("AILMFWYV"))
    pol = _frac(seq, set("STNQ"))
    gly_pro = _frac(seq, set("GP"))

    # Favor mixed hydrophobic and polar content
    mix = 1.0 - abs(hyd - 0.42) + 0.6 * (1.0 - abs(pol - 0.18))
    mix *= 0.5  # scale to [0,1]-ish

    # Penalize too many gly/pro (structure breakers)
    gp_pen = _clamp(1.0 - max(0.0, gly_pro - 0.12) / 0.20)

    L = len(seq)
    length_ok = _clamp((L - 40) / 160.0)

    return _clamp(0.45 * _clamp(mix) + 0.35 * gp_pen + 0.20 * length_ok)

