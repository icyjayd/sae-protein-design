# Extended metrics and tools

# Pareto plot and uncertainty are in analysis_metrics.py (run after extract/training)
python analysis_metrics.py

# Latent traversal (prototype decoding)
python latent_traversal.py --suffix mono --latent-dim-index 0 --steps 9 --step-size 1.0

# Latent-space optimization (evolutionary, uncertainty-aware)
python optimize_latent.py --suffix mono --iters 50 --pop 256 --elite 32 --sigma 0.5 --risk 0.25
