import pandas as pd, numpy as np, re
from sklearn.model_selection import train_test_split

def normalize_sequence(seq: str) -> str:
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq.upper())

def load_data(seq_file: str, labels_file: str | None = None) -> pd.DataFrame:
    """
    Load sequences and labels from either CSV or NPY sources.

    Supported modes:
    - Single CSV containing columns: sequence, label (and optionally id)
    - CSV sequences + CSV labels: merged on 'id' if present; otherwise aligned by order if labels CSV only has 'label' and same length
    - CSV sequences + NPY labels: aligned by order (length must match)
    - NPY sequences + NPY labels: aligned by order (length must match)
    - NPY sequences + CSV labels: aligned by order if CSV has only 'label' and same length

    NPY sequences are expected as a 1D array of strings (object) or a 2D array where the
    first column is the sequence string. If a structured array with a 'sequence' or 'seq'
    field is provided, that field will be used.
    """

    def _to_seq_list(arr: np.ndarray) -> list[str]:
        # Handle structured array with field 'sequence' or 'seq'
        if hasattr(arr, "dtype") and getattr(arr.dtype, "names", None):
            names = list(arr.dtype.names)
            field = "sequence" if "sequence" in names else ("seq" if "seq" in names else None)
            if field is not None:
                return [str(x) for x in arr[field].tolist()]
        # 1D array
        if arr.ndim == 1:
            return [str(x) for x in arr.tolist()]
        # 2D take first column
        if arr.ndim >= 2 and arr.shape[1] >= 1:
            return [str(x) for x in arr[:, 0].tolist()]
        raise ValueError("Unrecognized NPY sequence array shape")

    if seq_file.lower().endswith(".npy"):
        # Sequences from NPY
        seqs = np.load(seq_file, allow_pickle=True)
        seq_list = _to_seq_list(np.asarray(seqs))
        df = pd.DataFrame({"sequence": seq_list})

        if labels_file:
            if labels_file.lower().endswith(".npy"):
                y = np.load(labels_file, allow_pickle=True)
                y = np.asarray(y)
                if y.ndim > 1:
                    y = y.squeeze()
                if len(y) != len(df):
                    raise ValueError("labels_file length does not match sequences length")
                df["label"] = pd.Series(y).reset_index(drop=True)
            else:
                # CSV labels: allow order-based alignment if only 'label' present
                df_labels = pd.read_csv(labels_file, sep=None, engine="python")
                if "label" in df_labels.columns and len(df_labels) == len(df):
                    df["label"] = df_labels["label"].reset_index(drop=True)
                else:
                    raise ValueError("For NPY sequences, CSV labels must have only 'label' and match length.")

        if "label" not in df.columns:
            raise ValueError("Must include labels when sequences are provided as NPY. Provide labels_file.")

    else:
        # Sequences from CSV
        df = pd.read_csv(seq_file, sep=None, engine="python")
        if labels_file:
            if labels_file.lower().endswith(".npy"):
                y = np.load(labels_file, allow_pickle=True)
                y = np.asarray(y)
                if y.ndim > 1:
                    y = y.squeeze()
                if len(y) != len(df):
                    raise ValueError("labels_file length does not match sequences length")
                df["label"] = pd.Series(y).reset_index(drop=True)
            else:
                df_labels = pd.read_csv(labels_file, sep=None, engine="python")
                if "id" in df.columns and "id" in df_labels.columns:
                    df = pd.merge(df, df_labels, on="id", how="inner")
                elif "label" in df_labels.columns and len(df_labels) == len(df):
                    # no ids; align by row order
                    df["label"] = df_labels["label"].reset_index(drop=True)
                else:
                    raise ValueError("labels_file must have 'id' to merge or be same-length CSV with 'label'.")

        if "sequence" not in df.columns or "label" not in df.columns:
            raise ValueError("Must include 'sequence' and 'label'.")

    # Normalize sequences and return
    df["sequence"] = df["sequence"].astype(str).map(normalize_sequence)
    return df[["sequence", "label"]]

def split_data(df, seed=42, test_size=0.2, task="regression"):
    strat = df["label"] if (task=="classification" and df["label"].nunique()>1) else None
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
