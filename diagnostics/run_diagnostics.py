#!/usr/bin/env python3
"""
Diagnostic dashboard for the Protein Engineering pipeline.

Runs a short end-to-end experiment cycle and prints:
- per-agent runtimes
- acceptance rate
- score distributions (mean, stdev, min, max)
- output location
"""

import time
import json
from statistics import mean, stdev
from pathlib import Path
from agentic_lab.schema import ExperimentConfig, SequenceRecord, Memory
from agentic_lab.agents import ScientistAgent, EngineerAgent, ReviewerAgent, ArchivistAgent, CoordinatorAgent

def main(n_trials: int = 2):
    print("=== Protein Engineering Diagnostics ===")

    # --- Setup configuration ---
    cfg = ExperimentConfig(
        run_name="diagnostic_run",
        latent_dim=64,
        step_values=[-0.2, 0.2],
        candidate_dims=list(range(10)),
        sequences=[SequenceRecord(id="seed", sequence="MKTLLILAVITAIAAGALA")],
        out_dir=Path("runs/diagnostic_run")
    )
    memory = Memory()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize agents ---
    print("[INIT] Creating agents...")
    scientist = ScientistAgent(cfg, memory)
    engineer = EngineerAgent(cfg)
    reviewer = ReviewerAgent(plausibility_min=0.4, reversibility_min=0.8)
    archivist = ArchivistAgent(cfg, memory)
    coord = CoordinatorAgent(cfg, scientist, engineer, reviewer, archivist)

    # --- Measure per-agent runtimes ---
    print(f"[RUN] Executing {n_trials} trials...")
    start_all = time.time()
    start = time.time(); scientist.propose_plans(); t_sci = time.time() - start
    start = time.time(); engineer.run_trial(cfg.sequences[0], scientist.propose_plans()[0]); t_eng = time.time() - start
    start = time.time(); reviewer.accept(engineer.run_trial(cfg.sequences[0], scientist.propose_plans()[0])); t_rev = time.time() - start
    start = time.time(); coord.run(max_trials=n_trials); t_coord = time.time() - start
    total = time.time() - start_all
    # --- Run coordinator ---
    start = time.time()
    coord.run(max_trials=n_trials)
    t_coord = time.time() - start

    # --- Force flush memory to disk (ensures file exists) ---
    archivist.flush()

    # --- Load memory output ---
    mem_path = out_dir / "memory.json"
    if not mem_path.exists():
        print("[WARN] No memory.json found — writing empty report.")
        return
    with open(mem_path, "r") as f:
        data = json.load(f)

    accepted = [d for d in data if d.get("accepted")]
    scores = [d["scores"]["plausibility"] for d in data if "scores" in d]

    # --- Print diagnostics summary ---
    print("\n=== Diagnostic Summary ===")
    print(f"Trials completed: {len(data)}")
    print(f"Accepted: {len(accepted)} ({len(accepted)/len(data)*100:.1f}%)")
    print(f"Mean plausibility: {mean(scores):.3f} ± {stdev(scores) if len(scores)>1 else 0:.3f}")
    print(f"Min/Max plausibility: {min(scores):.3f}/{max(scores):.3f}")
    print(f"Output directory: {mem_path.parent.resolve()}")

    print("\n=== Runtime (seconds) ===")
    print(f"ScientistAgent: {t_sci:.3f}")
    print(f"EngineerAgent: {t_eng:.3f}")
    print(f"ReviewerAgent: {t_rev:.3f}")
    print(f"Coordinator (full loop): {t_coord:.3f}")
    print(f"Total runtime: {total:.3f}")
    print("=============================")

if __name__ == "__main__":
    main()
