import pandas as pd, numpy as np, re
from sklearn.model_selection import train_test_split

def normalize_sequence(seq: str) -> str:
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq.upper())

def load_data(seq_file: str, labels_file: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(seq_file, sep=None, engine="python")
    if labels_file:
        df_labels = pd.read_csv(labels_file, sep=None, engine="python")
        df = pd.merge(df, df_labels, on="id", how="inner")
    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError("Must include 'sequence' and 'label'.")
    df["sequence"] = df["sequence"].astype(str).map(normalize_sequence)
    return df[["sequence","label"]]

def split_data(df, seed=42, test_size=0.2, task="regression"):
    strat = df["label"] if (task=="classification" and df["label"].nunique()>1) else None
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
