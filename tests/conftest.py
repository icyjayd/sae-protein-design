import pytest
from agentic_lab.schema import ExperimentConfig, SequenceRecord, Memory

@pytest.fixture(scope="module")
def cfg_and_memory(tmp_path_factory):
    cfg = ExperimentConfig(
        run_name="agent_unit_tests",
        latent_dim=64,
        step_values=[-0.2, 0.2],
        candidate_dims=[0, 1],
        sequences=[SequenceRecord(id="seed", sequence="MKTLLILAVITAIAAGALA")],
        out_dir=tmp_path_factory.mktemp("runs")
    )
    memory = Memory()
    return cfg, memory
