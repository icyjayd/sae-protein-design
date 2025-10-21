# scoring/surrogate_models.py
from __future__ import annotations
from typing import Dict
import math

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
    best = cur = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best

def _shannon_entropy(seq: str) -> float:
    if not seq:
        return 0.0
    n = len(seq)
    counts: Dict[str, int] = {}
    for c in seq:
        counts[c] = counts.get(c, 0) + 1
    probs = [c / n for c in counts.values()]
    H = -sum(p * math.log2(p) for p in probs)
    # Normalize by max possible entropy (log2 of alphabet size actually used)
    Hmax = math.log2(min(len(AA), len(counts))) if counts else 1.0
    return 0.0 if Hmax == 0 else _clamp(H / Hmax)

def surrogate_score(seq: str) -> float:
    """
    Plausibility heuristic in [0,1].
    Combines composition balance, charge mix, aromaticity, entropy, and repeat penalty.
    Deterministic and fast; no network calls.
    """
    if not seq:
        return 0.0

    # Valid letters check
    valid_frac = sum(c in AA for c in seq) / len(seq)
    if valid_frac < 1.0:
        # Heavy penalty for non-canonical tokens
        return _clamp(0.5 * valid_frac)

    hyd = _fraction(seq, HYDROPHOBIC)
    pos = _fraction(seq, CHARGED_POS)
    neg = _fraction(seq, CHARGED_NEG)
    arom = _fraction(seq, AROMATIC)
    ent = _shannon_entropy(seq)
    run = _max_run(seq)

    # Target bands based on broad proteome statistics
    comp = 1.0 - abs(hyd - 0.45)            # hydrophobicity balance
    ch_mix = 1.0 - abs((pos + neg) - 0.18)  # ~18% charged overall
    ch_bal = 1.0 - abs(pos - neg)           # charge symmetry
    aro_ok = 1.0 - abs(arom - 0.10)         # ~10% aromatic
    rep_pen = _clamp(1.0 - max(0, run - 4) / 6.0)  # penalize long homopolymers

    # Length sanity (very short sequences unlikely to fold independently)
    L = len(seq)
    length_ok = _clamp((L - 20) / 80.0)  # 0 at 20aa, ~1 by 100aa

    # Weighted combination
    score = (
        0.22 * comp +
        0.18 * ch_mix +
        0.12 * ch_bal +
        0.12 * aro_ok +
        0.16 * ent +
        0.12 * rep_pen +
        0.08 * length_ok
    )
    return _clamp(score)
