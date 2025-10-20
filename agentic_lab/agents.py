from __future__ import annotations
from dataclasses import asdict
from typing import Dict, List, Any
from pathlib import Path
from .schema import SequenceRecord, ExperimentConfig, EditPlan, Scores, TrialResult, Memory
from . import tools

class ScientistAgent:
    """Analyzes results and proposes next perturbations."""
    def __init__(self, cfg: ExperimentConfig, memory: Memory):
        self.cfg = cfg
        self.memory = memory

    def propose_plans(self) -> List[EditPlan]:
        # Simple exploration-first strategy: try each candidate dim with each step
        plans = []
        for d in self.cfg.candidate_dims:
            for step in self.cfg.step_values:
                plans.append(EditPlan(dim=d, delta=step))
        return plans

class EngineerAgent:
    """Executes encode → perturb → decode → score."""
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

    def run_trial(self, seq: SequenceRecord, plan: EditPlan) -> TrialResult:
        base_latent = tools.encode_sequence(seq.sequence, self.cfg.latent_dim)
        edited_latent = tools.perturb_latent(base_latent, plan.dim, plan.delta)
        edited_seq = tools.decode_latent(edited_latent)

        # Scores
        stab, fold, plaus = tools.score_sequence(edited_seq)
        scores = Scores(stability=stab, folding=fold, plausibility=plaus)

        # Similarity and reversibility check
        sim = tools.sequence_similarity(seq.sequence, edited_seq)
        reversed_latent = tools.perturb_latent(edited_latent, plan.dim, -plan.delta)
        reversed_seq = tools.decode_latent(reversed_latent)
        rev_sim = tools.sequence_similarity(seq.sequence, reversed_seq)

        return TrialResult(
            seq_id=seq.id,
            base_seq=seq.sequence,
            edited_seq=edited_seq,
            dim=plan.dim,
            delta=plan.delta,
            scores=scores,
            similarity=sim,
            reversible_similarity=rev_sim,
        )

class ReviewerAgent:
    """Filters results using plausibility and reversibility guards."""
    def __init__(self, plausibility_min: float = 0.4, reversibility_min: float = 0.9):
        self.plausibility_min = plausibility_min
        self.reversibility_min = reversibility_min

    def accept(self, tr: TrialResult) -> bool:
        return (tr.scores.plausibility >= self.plausibility_min) and (tr.reversible_similarity >= self.reversibility_min)

class ArchivistAgent:
    """Logs results and maintains append-only memory."""
    def __init__(self, cfg: ExperimentConfig, memory: Memory):
        self.cfg = cfg
        self.memory = memory
        self.out_dir = (self.cfg.out_dir / self.cfg.run_name)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def record(self, tr: TrialResult, accepted: bool):
        row = {
            **asdict(tr),
            "accepted": accepted,
        }
        self.memory.log(event="trial", **row)

    def flush(self):
        # Persist JSON memory and a CSV-like JSONL for analysis
        mem_path = self.out_dir / "memory.json"
        self.memory.save(mem_path)

class CoordinatorAgent:
    """High-level loop coordinator."""
    def __init__(self, cfg: ExperimentConfig, scientist: ScientistAgent, engineer: EngineerAgent, reviewer: ReviewerAgent, archivist: ArchivistAgent):
        self.cfg = cfg
        self.scientist = scientist
        self.engineer = engineer
        self.reviewer = reviewer
        self.archivist = archivist

    def run(self, max_trials: int = 60):
        plans = self.scientist.propose_plans()
        trials_done = 0
        for plan in plans:
            for seq in self.cfg.sequences:
                tr = self.engineer.run_trial(seq, plan)
                accepted = self.reviewer.accept(tr)
                self.archivist.record(tr, accepted)
                trials_done += 1
                if trials_done >= max_trials:
                    self.archivist.flush()
                    return
        self.archivist.flush()
