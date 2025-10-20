from __future__ import annotations
from pathlib import Path
from .schema import ExperimentConfig, SequenceRecord, Memory
from .agents import ScientistAgent, EngineerAgent, ReviewerAgent, ArchivistAgent, CoordinatorAgent

def run_experiment(cfg: ExperimentConfig):
    memory = Memory()
    scientist = ScientistAgent(cfg, memory)
    engineer = EngineerAgent(cfg)
    reviewer = ReviewerAgent(plausibility_min=0.45, reversibility_min=0.9)
    archivist = ArchivistAgent(cfg, memory)
    coord = CoordinatorAgent(cfg, scientist, engineer, reviewer, archivist)
    coord.run(max_trials=80)
