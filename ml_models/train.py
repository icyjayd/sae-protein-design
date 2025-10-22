import numpy as np,pandas as pd,joblib,json,os
from pathlib import Path
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from .data_utils import load_data
from .encoding import encode_aac,encode_kmer,encode_onehot,encode_esm
from .models import build_model
from .metrics import compute_metrics

def _load_split_indices(split_path: str, n: int):
    with open(split_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    tr = d.get("train_idx", [])
    te = d.get("test_idx", [])
    if not isinstance(tr, list) or not isinstance(te, list):
        raise ValueError("Split file must contain 'train_idx' and 'test_idx' lists")
    tr = np.array(tr, dtype=int)
    te = np.array(te, dtype=int)
    if tr.min(initial=0) < 0 or te.min(initial=0) < 0 or tr.max(initial=-1) >= n or te.max(initial=-1) >= n:
        raise ValueError("Split indices out of range for current dataset")
    if len(set(tr.tolist()).intersection(set(te.tolist()))) > 0:
        raise ValueError("Train/test indices overlap in split file")
    return tr, te


def _save_split_indices(outdir: str | Path, train_idx, test_idx):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"train_idx": list(map(int, train_idx)), "test_idx": list(map(int, test_idx))}
    with open(outdir / "split.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train_model(
    seq_file,
    labels_file=None,
    model_name="xgb",
    encoding="kmer",
    task=None,
    seed=42,
    outdir="runs/ml_model",
    split_file=None,
    save_split=True,
    test_size=0.2,
    stratify="auto",
    n_samples=None,
    **kwargs,
):
    Path(outdir).mkdir(parents=True,exist_ok=True)
    df=load_data(seq_file,labels_file)
    
    # Limit to n_samples if specified
    if n_samples is not None and n_samples < len(df):
        df = df.iloc[:n_samples].copy()
        print(f"[INFO] Limited dataset to {n_samples} samples")
    
    if task is None:
        try: pd.to_numeric(df["label"]); task="regression"
        except: task="classification"
    if encoding=="aac": X=encode_aac(df.sequence)
    elif encoding=="kmer": X=encode_kmer(df.sequence,k=kwargs.get("k",3))
    elif encoding=="onehot": X=encode_onehot(df.sequence,max_len=kwargs.get("max_len",512))
    elif encoding=="esm": X=encode_esm(df.sequence,device=kwargs.get("device","cpu"))
    else: raise ValueError("Bad encoding")
    pre=StandardScaler(with_mean=False); X=pre.fit_transform(X)
    y=df["label"]; le=None
    if task=="classification" and not np.issubdtype(y.dtype,np.number):
        le=LabelEncoder(); y=le.fit_transform(y)
    else: y=pd.to_numeric(y)
    # Determine or reuse split (work with explicit index arrays for robustness)
    n = len(df)
    if split_file and os.path.exists(split_file):
        tr_idx, te_idx = _load_split_indices(split_file, n=n)
    else:
        idx = np.arange(n)
        # Determine stratification behavior
        is_classification = (task == "classification")
        can_stratify = is_classification and (len(np.unique(y)) > 1)
        if stratify == "yes" and can_stratify:
            strat = y
        elif stratify == "no":
            strat = None
        else:  # auto
            strat = y if can_stratify else None

        tr_idx, te_idx = train_test_split(
            idx, test_size=float(test_size), random_state=seed, stratify=strat
        )
        if save_split:
            _save_split_indices(outdir, tr_idx, te_idx)

    # Apply indices
    Xtr, Xte = X[tr_idx], X[te_idx]
    if isinstance(y, pd.Series):
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
    else:
        ytr, yte = y[tr_idx], y[te_idx]
    m=build_model(task,model_name,seed); m.fit(Xtr,ytr)
    yp=m.predict(Xte); yp_prob=m.predict_proba(Xte) if hasattr(m,"predict_proba") else None
    met=compute_metrics(task,yte,yp,yp_prob)
    joblib.dump(m,f"{outdir}/model.pkl"); joblib.dump(pre,f"{outdir}/scaler.pkl")
    if le: joblib.dump(le,f"{outdir}/labels.pkl")
    return met