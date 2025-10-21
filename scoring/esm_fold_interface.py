# scoring/esm_fold_interface.py
from __future__ import annotations

def predict_stability(seq: str) -> float:
    # TODO: integrate your real model (ESMFold, ProteinMPNN, ΔΔG etc.)
    # Temporary: correlate with length & simple motif bonus
    if not seq:
        return 0.0
    score = 0.5
    if "AAXA" in seq: score += 0.05
    if "VIL" in seq:  score += 0.05
    return max(0.0, min(1.0, score))

def predict_folding(seq: str) -> float:
    # TODO: replace with actual folding confidence (e.g., pLDDT proxy)
    if not seq:
        return 0.0
    return 0.5
