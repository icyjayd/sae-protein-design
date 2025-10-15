#!/usr/bin/env python3
import json, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

OUT = Path("outputs")

def load_metadata():
    meta_path = OUT / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}

def spearman_safe(a, b):
    try:
        rho, _ = spearmanr(a, b)
        return rho if np.isfinite(rho) else 0.0
    except Exception:
        return 0.0

def add_metadata_text(meta):
    fields = []
    if "gene" in meta:
        fields.append(f"Gene: {meta['gene']}")
    if "property" in meta:
        fields.append(f"Property: {meta['property']}")
    if "model" in meta:
        fields.append(f"Model: {meta['model']}")
    if "layer" in meta:
        fields.append(f"Layer: {meta['layer']}")
    if "date" in meta:
        fields.append(f"Date: {meta['date']}")
    return " | ".join(fields)

def run_for_suffix(sfx, sequences, y, meta):
    results = {}
    path = OUT / f"sparse_codes_{sfx}.npy"
    if not path.exists():
        print(f"[WARN] {path.name} not found; skipping.")
        return results

    Z = np.load(path)
    if Z.shape[0] == 0:
        print(f"[WARN] No data found for {sfx}; skipping.")
        return results

    corrs = [spearman_safe(Z[:, i], y) for i in range(Z.shape[1])]
    results["latent_property_corr_mean"] = float(np.mean(np.abs(corrs)))

    # Correlation Plot
    plt.figure(figsize=(8, 4))
    plt.scatter(np.arange(len(corrs)), corrs, alpha=0.6)
    plt.title(f"Latent-Property Correlations ({sfx})")
    plt.xlabel("Latent Dimension")
    plt.ylabel("Spearman ρ")
    plt.suptitle(add_metadata_text(meta), fontsize=8, y=0.95)
    plt.tight_layout()
    plt.savefig(OUT / f"latent_corr_{sfx}.png")
    plt.close()

    # Diversity
    diffs = np.mean([np.linalg.norm(Z[i] - Z[j]) for i in range(len(Z)) for j in range(i+1, len(Z))])
    results["diversity"] = float(diffs)

    # Surrogate Fit
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    n_train = int(0.8 * len(Z))
    rf.fit(Z[:n_train], y[:n_train])
    preds = rf.predict(Z[n_train:])
    results["r2"] = float(r2_score(y[n_train:], preds))
    results["fitness_mean"] = float(np.mean(preds))

    # Plot predicted vs actual
    plt.figure(figsize=(5, 5))
    plt.scatter(y[n_train:], preds, alpha=0.6)
    plt.xlabel("True Property")
    plt.ylabel("Predicted Property")
    plt.title(f"Surrogate Fit ({sfx}) R²={results['r2']:.3f}")
    plt.suptitle(add_metadata_text(meta), fontsize=8, y=0.95)
    plt.tight_layout()
    plt.savefig(OUT / f"surrogate_fit_{sfx}.png")
    plt.close()

    return results

def create_pdf_report(summary, meta):
    pdf_path = OUT / "pipeline_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path))
    styles = getSampleStyleSheet()
    story = [Paragraph("Protein SAE Analysis Report", styles["Title"]),
             Spacer(1, 12)]

    if meta:
        story.append(Paragraph("Metadata", styles["Heading2"]))
        for k, v in meta.items():
            story.append(Paragraph(f"{k.capitalize()}: {v}", styles["Normal"]))
        story.append(Spacer(1, 12))

    if summary:
        story.append(Paragraph("Metric Summary", styles["Heading2"]))
        for key, val in summary.items():
            story.append(Paragraph(f"{key}: {val:.4f}", styles["Normal"]))
        story.append(Spacer(1, 12))
    else:
        story.append(Paragraph("No metrics found.", styles["Normal"]))

    for plot in sorted(OUT.glob("*.png")):
        story.append(Paragraph(plot.name, styles["Heading3"]))
        story.append(Image(str(plot), width=400, height=250))
        story.append(Spacer(1, 12))

    doc.build(story)
    print(f"[INFO] PDF report created at {pdf_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", action="store_true")
    args = ap.parse_args()

    prop_path = OUT / "labels.npy"
    seq_path = OUT / "sequences.npy"
    y = np.load(prop_path)
    sequences = np.load(seq_path, allow_pickle=True) if seq_path.exists() else np.array([])

    meta = load_metadata()
    summary = {}
    for sfx in ["regular", "mono"]:
        res = run_for_suffix(sfx, sequences, y, meta)
        for k, v in res.items():
            summary[f"{sfx}_{k}"] = v

    with open(OUT / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    create_pdf_report(summary, meta)
    print("[INFO] Analysis complete.")

if __name__ == "__main__":
    main()
