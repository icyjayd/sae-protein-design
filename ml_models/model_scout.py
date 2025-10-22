"""
Parallel model scout for protein design ML pipeline.

Evaluates multiple model / encoding / sample-size combinations,
computes Spearman rho + p-value, ranks them, and auto-generates plots.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed, cpu_count
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from .data_utils import load_data
from .encoding import encode_sequences
from .models import build_model


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DEFAULT_MODELS = [
    "ridge", "lasso", "enet", "rf", "gb", "xgb", "lgbm", "svr", "mlp",
]
DEFAULT_ENCODINGS = ["aac", "dpc", "kmer"]
DEFAULT_SAMPLE_GRID = [2000, 5000, 10000, 20000, 30000]
ALPHA = 0.01
N_JOBS = max(cpu_count() - 1, 1)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _spearman(y_true, y_pred):
    rho, p = spearmanr(y_true, y_pred)
    if np.isnan(rho): rho = 0.0
    if np.isnan(p): p = 1.0
    return float(rho), float(p)


def _run_single(model_name, encoding, n_samples, df, task, seed, test_size, stratify):
    try:
        df_sub = df.sample(n=min(n_samples, len(df)), random_state=seed)
        X = encode_sequences(df_sub["sequence"], encoding=encoding, k=3)
        y = df_sub["label"].values

        scaler = StandardScaler(with_mean=False)
        X = scaler.fit_transform(X)
        strat_y = y if (task == "classification" and stratify == "auto") else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=strat_y
        )

        model = build_model(task, model_name)
        t0 = time.time()
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        seconds = round(time.time() - t0, 3)
        rho, p = _spearman(yte, ypred)

        return {
            "model": model_name,
            "encoding": encoding,
            "n_samples": int(n_samples),
            "rho": rho,
            "p": p,
            "seconds": seconds,
            "status": "ok",
        }
    except Exception as e:
        return {
            "model": model_name,
            "encoding": encoding,
            "n_samples": int(n_samples),
            "rho": 0.0,
            "p": 1.0,
            "seconds": 0,
            "status": f"error: {type(e).__name__}: {e}",
        }


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def _make_plots(df_results, outdir):
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(outdir, exist_ok=True)
    sns.set(style="whitegrid")

    # --- Heatmap of best rho per model/encoding ---
    heat = (
        df_results.groupby(["model", "encoding"])["rho"]
        .max()
        .unstack(fill_value=0)
        .sort_index()
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Best Spearman ρ per model/encoding")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "heatmap_rho.png"))
    plt.close()

    # --- Line plot of rho vs sample size for each encoding ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_results,
        x="n_samples",
        y="rho",
        hue="model",
        style="encoding",
        markers=True,
        dashes=False,
    )
    plt.title("Spearman ρ vs sample size")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "rho_vs_samples.png"))
    plt.close()

    # --- Runtime plot ---
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_results.groupby("model", as_index=False)["seconds"].mean(),
        x="model",
        y="seconds",
        palette="crest",
    )
    plt.title("Average runtime per model (seconds)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "runtime_per_model.png"))
    plt.close()


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------
def run_scout(
    seq_file,
    labels_file=None,
    models=DEFAULT_MODELS,
    encodings=DEFAULT_ENCODINGS,
    sample_grid=DEFAULT_SAMPLE_GRID,
    alpha=ALPHA,
    seed=42,
    test_size=0.2,
    stratify="none",
    outpath="runs/model_scout/model_scout_results.json",
    n_jobs=N_JOBS,
):
    print(f"[INFO] Starting model scout using {n_jobs} parallel jobs")

    df = load_data(seq_file, labels_file)
    if not np.issubdtype(df["label"].dtype, np.number):
        df["label"] = LabelEncoder().fit_transform(df["label"])
        task = "classification"
    else:
        task = "regression"

    combos = [
        (m, e, n)
        for m in models
        for e in encodings
        for n in sample_grid
        if len(df) >= n
    ]
    print(f"[INFO] Total runs: {len(combos)}")

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_run_single)(m, e, n, df, task, seed, test_size, stratify)
        for (m, e, n) in combos
    )

    outdir = os.path.dirname(outpath) or "runs/model_scout"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Save raw results
    json.dump(results, open(outpath, "w"), indent=2)
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(outdir, "model_scout_results.csv"), index=False)

    # Filter significant results
    df_valid = df_results[df_results["p"] <= alpha].copy()
    if len(df_valid) == 0:
        print("[WARN] No models with significant p ≤ α; using all results.")
        df_valid = df_results.copy()

    # Aggregate best per model/encoding
    grouped = (
        df_valid.groupby(["model", "encoding"], as_index=False)
        .agg({"rho": "max", "n_samples": "min", "p": "min"})
        .sort_values("rho", ascending=False)
    )

    print("\nTop configurations (Spearman ρ):")
    print(grouped.head(10).to_string(index=False))

    # --- Plots ---
    plot_dir = os.path.join(outdir, "plots")
    _make_plots(df_results, plot_dir)
    print(f"[INFO] Plots saved to: {plot_dir}")

    # --- Combined output ---
    final = {
        "alpha": alpha,
        "models_tested": models,
        "encodings_tested": encodings,
        "sample_grid": sample_grid,
        "n_jobs": n_jobs,
        "task": task,
        "ranked_results": grouped.to_dict(orient="records"),
        "all_results": results,
        "plot_dir": plot_dir,
    }
    with open(outpath, "w") as f:
        json.dump(final, f, indent=2)

    print(f"[INFO] Results saved to: {outpath}")
    return final


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parallel model scout with automatic plots.")
    ap.add_argument("sequences", type=str)
    ap.add_argument("--labels", type=str, default=None)
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    ap.add_argument("--encodings", type=str, default=",".join(DEFAULT_ENCODINGS))
    ap.add_argument("--samples", type=str, default=",".join(map(str, DEFAULT_SAMPLE_GRID)))
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--stratify", type=str, default="none", choices=["none", "auto"])
    ap.add_argument("--out", type=str, default="runs/model_scout/model_scout_results.json")
    ap.add_argument("--jobs", type=int, default=N_JOBS)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    encodings = [e.strip() for e in args.encodings.split(",") if e.strip()]
    sample_grid = [int(x) for x in args.samples.split(",") if x.strip()]

    run_scout(
        seq_file=args.sequences,
        labels_file=args.labels,
        models=models,
        encodings=encodings,
        sample_grid=sample_grid,
        alpha=args.alpha,
        test_size=args.test_size,
        stratify=args.stratify,
        outpath=args.out,
        n_jobs=args.jobs,
    )
