import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from .data_utils import load_data
from .models import build_model
from .metrics import compute_metrics
from .encoding import encode_sequences

# --------------------------------------------------
# Helper to auto-generate an output folder
# --------------------------------------------------
def _auto_outdir(model_name, encoding, n_samples, prefix=None, base="runs/ml_model"):
    n_str = str(n_samples) if n_samples else "all"
    parts = []
    if prefix:
        parts.append(prefix)
    parts += [model_name, encoding, n_str]
    run_name = "_".join(parts)
    return os.path.join(base, run_name)


# --------------------------------------------------
# Main training entry
# --------------------------------------------------
def train_model(
    seq_file,
    labels_file=None,
    task=None,
    model_name="rf",
    encoding="aac",
    seed=42,
    outdir=None,
    prefix=None,
    n_samples=None,
    test_size=0.2,
    stratify="auto",
    k=3,
    max_len=512,
    split=None,
    save_split=True,
):
    np.random.seed(seed)

    # ----- Load data -----
    df = load_data(seq_file, labels_file)
    if n_samples:
        df = df.sample(n=min(n_samples, len(df)), random_state=seed)
    n_samples = n_samples or "all"

    # ----- Task detection -----
    if task is None:
        task = "regression" if np.issubdtype(df["label"].dtype, np.number) else "classification"

    # ----- Encode sequences -----
    X = encode_sequences(df["sequence"], encoding, k=k, max_len=max_len)

    y = df["label"].values

    # ----- Scale features -----
    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)

    # ----- Label encoding for classification -----
    label_encoder = None
    if task == "classification" and not np.issubdtype(y.dtype, np.number):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # ----- Split -----
    if split and os.path.exists(split):
        split_data = json.load(open(split))
        train_idx, test_idx = split_data["train_idx"], split_data["test_idx"]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        stratify_y = y if (task == "classification" and stratify == "auto") else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify_y
        )
        if save_split:
            split_data = {
                "train_idx": np.arange(len(y_train)).tolist(),
                "test_idx": np.arange(len(y_test)).tolist(),
            }
        else:
            split_data = None

    # ----- Model -----
    model = build_model(task, model_name)
    model.fit(X_train, y_train)

    # ----- Predict & Metrics -----
    y_pred = model.predict(X_test)
    metrics = compute_metrics(task, y_test, y_pred)

    # ----- Output directory -----
    if outdir is None:
        outdir = _auto_outdir(model_name, encoding, None if n_samples == "all" else n_samples, prefix)
    os.makedirs(outdir, exist_ok=True)

    # ----- Save artifacts -----
    import joblib
    joblib.dump(model, os.path.join(outdir, "model.pkl"))
    joblib.dump(scaler, os.path.join(outdir, "scaler.pkl"))
    if label_encoder is not None:
        joblib.dump(label_encoder, os.path.join(outdir, "labels.pkl"))
    if split_data and save_split:
        with open(os.path.join(outdir, "split.json"), "w") as f:
            json.dump(split_data, f, indent=2)

    # ----- Save metrics and config manifest -----
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "model": model_name,
        "encoding": encoding,
        "n_samples": n_samples,
        "seed": seed,
        "metrics": metrics,
        "outdir": outdir,
    }
    with open(os.path.join(outdir, "run.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics
