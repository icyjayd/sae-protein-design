import pytest
import json
from pathlib import Path
from agentic_lab.loop import run_experiment_for_test

@pytest.mark.slow
def test_agentic_loop_smoke():
    """
    Run one agentic loop and verify that memory.json
    is generated with valid trial information.
    """
    memory_path = run_experiment_for_test(n_steps=1)
    assert memory_path.exists(), "memory.json not generated"

    with open(memory_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list) and len(data) > 0, "Memory file empty or wrong format"
    record = data[0]

    # --- Core structure validation ---
    expected_top = {"dim", "delta", "edited_seq", "scores"}
    missing = expected_top - record.keys()
    assert not missing, f"Missing expected top-level fields: {missing}"

    # --- Verify nested score fields ---
    scores = record["scores"]
    for key in ("stability", "folding", "plausibility"):
        assert key in scores, f"Missing score field: {key}"
        assert 0.0 <= scores[key] <= 1.0, f"Score {key} out of range: {scores[key]}"

    # --- Sequence and metadata checks ---
    seq = record["edited_seq"]
    assert isinstance(seq, str) and len(seq) > 0, "Edited sequence missing or invalid"
    assert set(seq).issubset(set("ACDEFGHIKLMNPQRSTVWY")), "Edited sequence has invalid amino acids"

    print(f"\n✅ Agentic loop smoke test passed. Memory saved at: {memory_path}")
