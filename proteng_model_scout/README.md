# proteng-scout (modular)

Parallel **model scouting** for protein ML pipelines. It evaluates multiple model/encoding/sample-size
combinations, ranks by Spearman ρ (with p-value), and generates plots + an HTML report.

This package **auto-detects your repo root** at runtime to import `ml_models` from your existing project.

## Install (editable)

```bash
pip install -e ./proteng_scout_pkg
```

## Usage

```bash
model-scout path/to/sequences.csv --labels path/to/labels.npy   --models ridge,rf,xgb --encodings aac,dpc,kmer --samples 2000,5000,10000   --out runs/model_scout/model_scout_results.json --jobs 8
```

Outputs:

```
runs/model_scout/
├── model_scout_results.json
├── model_scout_results.csv
├── plots/
│   ├── heatmap_rho.png
│   ├── rho_vs_samples.png
│   └── runtime_per_model.png
└── reports/
    └── summary.html
```
