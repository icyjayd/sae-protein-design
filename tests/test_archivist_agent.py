import pytest
import json
from agentic_lab.agents import ArchivistAgent, TrialResult, Scores
from agentic_lab.schema import Memory, ExperimentConfig, SequenceRecord

def test_archivist_records_and_flushes(tmp_path):
    out_dir = tmp_path / "runs"
    cfg = ExperimentConfig(
        run_name="archivist_test",
        latent_dim=64,
        step_values=[-0.2, 0.2],
        candidate_dims=[0],
        sequences=[SequenceRecord(id="seed", sequence="MKTLLILAVITAIAAGALA")],
        out_dir=out_dir
    )
    memory = Memory()
    archivist = ArchivistAgent(cfg, memory)

    tr = TrialResult(
        seq_id="seed",
        base_seq="MKTLLILAVITAIAAGALA",
        edited_seq="MKTLLILAVITAIAAGALA",
        dim=0,
        delta=0.2,
        scores=Scores(stability=0.5, folding=0.6, plausibility=0.7),
        similarity=0.9,
        reversible_similarity=0.9
    )
    archivist.record(tr, accepted=True)
    archivist.flush()

    mem_path = out_dir / "archivist_test" / "memory.json"
    assert mem_path.exists(), "Archivist failed to save memory"
    with open(mem_path) as f:
        data = json.load(f)
    assert isinstance(data, list) and len(data) > 0
    print("Archivist wrote record:", data[0])
