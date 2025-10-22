import os
import json
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from .data_utils import load_data
from .models import build_model
from .metrics import compute_metrics
from .encoding import encode_sequences

def _auto_outdir(model_name, encoding, n_samples, prefix=None, base="runs/ml_model"):
    n_str = str(n_samples) if n_samples else "all"
    parts = []
    if prefix:
        parts.append(prefix)
    parts += [model_name, encoding, n_str]
    run_name = "_".join(parts)
    return os.path.join(base, run_name)

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
    split_file=None,
    save_split=True,
):
    np.random.seed(seed)

    seq_file = str(seq_file)
    labels_file = str(labels_file) if labels_file is not None else None
    outdir = str(outdir) if outdir is not None else None
    split_path = split_file or split
    split_path = str(split_path) if split_path is not None else None

    df = load_data(seq_file, labels_file)
    if n_samples:
        df = df.sample(n=min(n_samples, len(df)), random_state=seed)
    n_samples = n_samples or "all"

    if task is None:
        task = "regression" if np.issubdtype(df["label"].dtype, np.number) else "classification"

    X = encode_sequences(df["sequence"], encoding, k=k, max_len=max_len)
    y = df["label"].values

    scaler = StandardScaler(with_mean=False)
    X = scaler.fit_transform(X)

    label_encoder = None
    if task == "classification" and not np.issubdtype(y.dtype, np.number):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    idx = np.arange(len(y))
    if split_path and os.path.exists(split_path):
        split_data = json.load(open(split_path))
        train_idx = np.asarray(split_data["train_idx"], dtype=int)
        test_idx = np.asarray(split_data["test_idx"], dtype=int)
    else:
        stratify_y = y if (task == "classification" and stratify == "auto") else None
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=stratify_y
        )
        split_data = {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()} if save_split else None

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = build_model(task, model_name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = compute_metrics(task, y_test, y_pred)

    if outdir is None:
        outdir = _auto_outdir(model_name, encoding, None if n_samples == "all" else n_samples, prefix)
    os.makedirs(outdir, exist_ok=True)

    import joblib
    joblib.dump(model, os.path.join(outdir, "model.pkl"))
    joblib.dump(scaler, os.path.join(outdir, "scaler.pkl"))
    if label_encoder is not None:
        joblib.dump(label_encoder, os.path.join(outdir, "labels.pkl"))

    if split_data and save_split:
        split_out = split_path or os.path.join(outdir, "split.json")
        with open(split_out, "w") as f:
            json.dump(split_data, f, indent=2)

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "model": model_name,
        "encoding": encoding,
        "n_samples": n_samples,
        "seed": seed,
        "metrics": metrics,
        "outdir": str(outdir),
    }
    with open(os.path.join(outdir, "run.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return metrics
