
# SAE Protein Design — HF + GPU + Finetune + Presets

## Quick start
```bash
pip install -r requirements.txt

# Toy run
python generate_activations.py
python run_pipeline.py --mode both --retrain

# Use pretrained InterPLM SAE (auto-installs interplm). Layer defaults via preset.
python run_pipeline.py --from_hf_model esm2-8m

# Fine-tune pretrained SAE; freeze decoder; small LR on frozen; 10 epochs
python run_pipeline.py --from_hf_model esm2-8m --layer 4 --finetune --freeze decoder --lr-mult 0.1 --epochs 10

# List available pretrained presets
python run_pipeline.py --list-hf-presets
```
Outputs appear in ./outputs (named checkpoints/configs, metrics, and `pipeline_report.pdf`).
