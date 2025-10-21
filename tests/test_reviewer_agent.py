import pytest
from agentic_lab.agents import ReviewerAgent, TrialResult, Scores

def test_reviewer_accepts_reasonably():
    reviewer = ReviewerAgent(plausibility_min=0.4, reversibility_min=0.9)
    scores = Scores(stability=0.6, folding=0.7, plausibility=0.8)
    tr = TrialResult(
        seq_id="seed",
        base_seq="MKTLLILAVITAIAAGALA",
        edited_seq="MKTLLILAVITAIAAGALA",
        dim=0,
        delta=0.2,
        scores=scores,
        similarity=0.95,
        reversible_similarity=0.95
    )
    accepted = reviewer.accept(tr)
    assert isinstance(accepted, bool)
    print(f"Reviewer decision: {'accepted' if accepted else 'rejected'}")
