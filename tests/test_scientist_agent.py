import pytest
from agentic_lab.agents import ScientistAgent, EditPlan

def test_scientist_agent_proposes(cfg_and_memory):
    cfg, memory = cfg_and_memory
    scientist = ScientistAgent(cfg, memory)
    plans = scientist.propose_plans()

    assert isinstance(plans, list) and len(plans) > 0, "Scientist produced no plans"
    assert all(isinstance(p, EditPlan) for p in plans), "Plans must be EditPlan objects"
    print(f"Scientist produced {len(plans)} edit plans. Example: {plans[0]}")
