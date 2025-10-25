import numpy as np
import pandas as pd
from typing import List, Dict, Literal, Tuple
from scipy import stats
import os


def aggregate_per_sequence(
    z_pos_list: List[np.ndarray],
    method: Literal["mean", "median"] = "mean"
) -> np.ndarray:
    """Aggregate variable-length per-position latent activations into per-sequence vectors."""
    if not z_pos_list:
        return np.empty((0, 0))

    n_seq = len(z_pos_list)
    dim = z_pos_list[0].shape[1] if z_pos_list[0].size > 0 else 0
    Z = np.zeros((n_seq, dim), dtype=np.float64)

    for i, z in enumerate(z_pos_list):
        if z.size == 0:
            Z[i] = 0.0
        else:
            if method == "mean":
                Z[i] = np.nanmean(z, axis=0)
            elif method == "median":
                Z[i] = np.nanmedian(z, axis=0)
            else:
                raise ValueError(f"Unknown method: {method}")
    return Z


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str = "spearman") -> Tuple[float, float, int]:
    """Compute correlation robustly, returning (corr, pval, n_valid)."""
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < 3:
        return 0.0, 1.0, int(n)
    x_ = x[mask]
    y_ = y[mask]
    # Check for zero variance
    if np.nanstd(x_) == 0 or np.nanstd(y_) == 0:
        return 0.0, 1.0, int(n)
    try:
        if method == "spearman":
            r, p = stats.spearmanr(x_, y_)
        elif method == "pearson":
            r, p = stats.pearsonr(x_, y_)
        else:
            raise ValueError(f"Unknown correlation method {method}")
    except Exception:
        r, p = 0.0, 1.0
    if np.isnan(r) or np.isnan(p):
        r, p = 0.0, 1.0
    return float(r), float(p), int(n)

def compute_correlations(
    Z: np.ndarray,
    y: np.ndarray,
    method: Literal["pearson", "spearman"] = "spearman"
) -> pd.DataFrame:
    """
    Compute per-latent correlations with a sequence-level property vector.

    Returns a DataFrame compatible with experiments/poc_pipeline.py:
    columns = ['latent_index', 'spearman', 'p_value', 'n']
    """
    if Z.ndim != 2:
        raise ValueError("Z must be 2D (N,D)")
    if y.ndim != 1:
        raise ValueError("y must be 1D")

    N, D = Z.shape
    results = []
    for d in range(D):
        r, p, n = _safe_corr(Z[:, d], y, method)
        results.append((d, r, p, n))

    # Build DataFrame with exact column names expected downstream
    df = pd.DataFrame(results, columns=["latent_index", "spearman", "p_value", "n"])
    # sort by absolute correlation magnitude
    df = df.sort_values("spearman", key=lambda c: np.abs(c), ascending=False).reset_index(drop=True)
    return df

def select_candidates(
    corr_df: pd.DataFrame,
    top_n: int = 5,
    null_threshold: float = 0.05
) -> Dict[str, List[int]]:
    """Select top positive, negative, and null latent features."""
    if corr_df.empty:
        return {"positive": [], "negative": [], "null": []}

    sig_df = corr_df[corr_df["pval"] < null_threshold]
    pos = sig_df[sig_df["corr"] > 0].nlargest(top_n, "corr")["latent"].tolist()
    neg = sig_df[sig_df["corr"] < 0].nsmallest(top_n, "corr")["latent"].tolist()
    null_latents = corr_df[~corr_df["latent"].isin(pos + neg)]["latent"].tolist()
    return {"positive": pos, "negative": neg, "null": null_latents}


def save_correlation_csv(corr_df: pd.DataFrame, out_path: str) -> str:
    """Save correlation DataFrame to CSV and return the path."""
    corr_df.to_csv(out_path, index=False)
    return os.fspath(out_path)
