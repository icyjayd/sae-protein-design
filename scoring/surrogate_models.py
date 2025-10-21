# scoring/surrogate_models.py
from __future__ import annotations
from typing import Dict

# Simple composition-based plausibility as a placeholder.
HYDROPHOBIC = set("AILMFWYV")
CHARGED = set("DEHKR")

def surrogate_score(seq: str) -> float:
    if not seq:
        return 0.0
    n = len(seq)
    hyd = sum(c in HYDROPHOBIC for c in seq)/n
    charge = sum(c in CHARGED for c in seq)/n
    # prefer moderate hydrophobicity and ~20% charged
    plaus = 0.4*(1.0-abs(hyd-0.45)) + 0.6*(1.0-abs(charge-0.2))
    return max(0.0, min(1.0, plaus))
