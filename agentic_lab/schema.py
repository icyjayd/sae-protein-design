from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import time

@dataclass
class SequenceRecord:
    id: str
    sequence: str

@dataclass
class ExperimentConfig:
    run_name: str
    latent_dim: int = 64
    step_values: List[float] = field(default_factory=lambda: [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3])
    candidate_dims: List[int] = field(default_factory=lambda: list(range(0, 16)))
    sequences: List[SequenceRecord] = field(default_factory=list)
    out_dir: Path = Path("runs")

@dataclass
class EditPlan:
    dim: int
    delta: float

@dataclass
class Scores:
    stability: float
    folding: float
    plausibility: float

@dataclass
class TrialResult:
    seq_id: str
    base_seq: str
    edited_seq: str
    dim: int
    delta: float
    scores: Scores
    similarity: float
    reversible_similarity: float

@dataclass
class Memory:
    # Simple append-only "lab notebook"
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, **kwargs):
        entry = dict(timestamp=time.time(), **kwargs)
        self.entries.append(entry)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.entries, indent=2))

def dump_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
