import pandas as pd, numpy as np, re
from sklearn.model_selection import train_test_split

def normalize_sequence(seq: str) -> str:
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq.upper())

def load_data(sequences_file, labels_file=None, **kwargs):
    """
    Load sequences and labels from CSV/NPY combinations and normalize sequences.
    Supported:
      - CSV sequences (+ optional CSV/NPY labels)
      - NPY sequences (+ required CSV/NPY labels)
      - CSV+CSV with merge on 'id' if present, else row-aligned if labels-only column exists
      - NPY+CSV row-aligned if CSV has only 'label' (or we can coerce to that)
    """
    sequences_file = str(sequences_file)
    labels_file = str(labels_file) if labels_file else None

    def _to_seq_list(arr: np.ndarray):
        arr = np.asarray(arr)
        # structured array with 'sequence' or 'seq'
        if hasattr(arr, "dtype") and getattr(arr.dtype, "names", None):
            names = list(arr.dtype.names)
            if "sequence" in names:
                return [str(x) for x in arr["sequence"].tolist()]
            if "seq" in names:
                return [str(x) for x in arr["seq"].tolist()]
        # 1D array of strings
        if arr.ndim == 1:
            return [str(x) for x in arr.tolist()]
        # 2D: take first column
        if arr.ndim >= 2 and arr.shape[1] >= 1:
            return [str(x) for x in arr[:, 0].tolist()]
        raise ValueError("Unrecognized NPY sequence array shape")

    # ---- NPY sequences ----
    if sequences_file.lower().endswith(".npy"):
        seqs = np.load(sequences_file, allow_pickle=True)
        seq_list = _to_seq_list(seqs)
        df = pd.DataFrame({"sequence": seq_list})

        if labels_file:
            if labels_file.lower().endswith(".npy"):
                y = np.load(labels_file, allow_pickle=True)
                y = np.asarray(y).squeeze()
                if len(y) != len(df):
                    raise ValueError("labels_file length does not match sequences length")
                df["label"] = pd.Series(y).reset_index(drop=True)
            else:
                df_labels = pd.read_csv(labels_file, sep=None, engine="python")
                # accept a single-column CSV and coerce the column name to 'label' if needed
                if "label" not in df_labels.columns and df_labels.shape[1] == 1:
                    df_labels.columns = ["label"]
                if "label" in df_labels.columns and len(df_labels) == len(df):
                    df["label"] = df_labels["label"].reset_index(drop=True)
                else:
                    raise ValueError("For NPY sequences, CSV labels must have only 'label' and match length.")
        else:
            raise ValueError("Must include labels when sequences are provided as NPY. Provide labels_file.")

        # normalize and return
        df["sequence"] = df["sequence"].astype(str).map(normalize_sequence)
        return df[["sequence", "label"]]

    # ---- CSV sequences ----
    if sequences_file.lower().endswith(".csv"):
        df = pd.read_csv(sequences_file, sep=None, engine="python")
        if labels_file:
            if labels_file.lower().endswith(".npy"):
                y = np.load(labels_file, allow_pickle=True)
                y = np.asarray(y).squeeze()
                if len(y) != len(df):
                    raise ValueError("labels_file length does not match sequences length")
                df["label"] = pd.Series(y).reset_index(drop=True)
            else:
                df_labels = pd.read_csv(labels_file, sep=None, engine="python")
                if "id" in df.columns and "id" in df_labels.columns:
                    df = pd.merge(df, df_labels, on="id", how="inner")
                else:
                    # accept a single-column labels CSV without id
                    if "label" not in df_labels.columns and df_labels.shape[1] == 1:
                        df_labels.columns = ["label"]
                    if "label" in df_labels.columns and len(df_labels) == len(df):
                        df["label"] = df_labels["label"].reset_index(drop=True)
                    else:
                        raise ValueError("labels_file must have 'id' to merge or be same-length CSV with 'label'.")

        if "sequence" not in df.columns or "label" not in df.columns:
            raise ValueError("Must include 'sequence' and 'label'.")

        df["sequence"] = df["sequence"].astype(str).map(normalize_sequence)
        return df[{"sequence", "label"}] if isinstance(df, dict) else df[["sequence", "label"]]

    raise ValueError("Unsupported file format.")

def split_data(df, seed=42, test_size=0.2, task="regression"):
    strat = df["label"] if (task == "classification" and df["label"].nunique() > 1) else None
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
