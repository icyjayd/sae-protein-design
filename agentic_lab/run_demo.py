from __future__ import annotations
from pathlib import Path
from .schema import ExperimentConfig, SequenceRecord
from .loop import run_experiment

# A few toy seed sequences (replace with your proteins)
SEQS = [
    SequenceRecord(id="seqA", sequence="MKTLLILAVITAIAAGALA" * 3),
    SequenceRecord(id="seqB", sequence="GGHHPPAAVVLLRRKKSSTT" * 2),
    SequenceRecord(id="seqC", sequence="VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPH"),
]

def main():
    cfg = ExperimentConfig(
        run_name="demo_run",
        latent_dim=64,
        step_values=[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3],
        candidate_dims=list(range(0, 12)),
        sequences=SEQS,
        out_dir=Path("runs"),
    )
    run_experiment(cfg)
    print("Done. See outputs under ./runs/demo_run/")

if __name__ == "__main__":
    main()
