# scoring/__init__.py
from .surrogate_models import surrogate_score
from .esm_fold_interface import predict_stability, predict_folding

# Reuse the real metrics implementation from SAE
from sae.utils.grade_reconstructions import grade_pair

# For convenience, provide an alias so tools can call sequence_similarity()
def sequence_similarity(a: str, b: str) -> float:
    return grade_pair(a, b)["final_score"]
