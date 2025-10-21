import pytest
from agentic_lab.agents import EngineerAgent, EditPlan, SequenceRecord, TrialResult

def test_engineer_agent_run_trial(cfg_and_memory):
    cfg, _ = cfg_and_memory
    engineer = EngineerAgent(cfg)

    seq = SequenceRecord(id="seed", sequence="MKTLLILAVITAIAAGALA")
    plan = EditPlan(dim=0, delta=0.2)
    result = engineer.run_trial(seq, plan)

    assert isinstance(result, TrialResult)
    assert isinstance(result.edited_seq, str)
    assert len(result.edited_seq) > 0
    print("Engineer output sequence:", result.edited_seq[:40])
